# 反向路径静态审计（tgai_go.train）

> 审计对象：`D:\DSH-Workspace\TGAI-对话档案\_tgai_orv`
> 方式：**只读静态审计**（未修改任何仓库文件）+ 三份 CPU 侧可复现的验证脚本（见 §8）。
> 口径：V100-32GB，Qwen2.5-0.5B 形状，fp32 + AdamW，**B=4 / S=512 / L=24 / NH=14 / KV=2 / DH=64 / FF=4864 / V=151936**，
> `GRAD_CKPT=0`、`SDPA_LOWMEM=0`（即 `docs/GAP_VS_PYTORCH.md` §9 的 base 配置：forward 0.419 s / backward 0.723 s / opt 0.024 s）。
> 实测锚点（本文引用的三个数）：`backward = 0.723 s`、`forward = 0.419 s`、`silu_mul 融合上线后 1.22 → 1.17 s`（`docs/GAP_VS_PYTORCH.md` 145–147、163–166 行）。

---

## 0. 先给结论（TL;DR）

1. **反向 0.723 s 里，矩阵乘只占约 0.39 s**（2× 前向 GEMM 0.19 s + attention 反向 0.01 s）。剩下 **约 0.33 s 是非 GEMM 的拷贝/逐元素/多余内核**，这就是本报告的目标。
   本文逐项点名 **26.9 GB/步（反向）+ 24.0 GB/步（前向）** 的访存和 **约 2400 个内核/步**；其中 **约 28 GB 与 ~1500 个内核是可省的**（本报告只主张其中前 5 条、≈340 ms，其余列在"值得排期"里）。
2. 反向里**最大的三个可避免项**：① RoPE 没接融合核（7.4 GB / 672 kernel）② softmax 反向 5 趟访存（7.0 GB）③ `p@v` 反向对 `p` 做了一次 **58.7 MB 的 `swap_last2` 真拷贝**（2.8 GB）。
3. **两条"看着该做、其实不能做"的结论**（都已实测证伪）：
   * 不能把 `T.fused_rope` 直接接到 Qwen：它是**交错式**，Qwen 是 **rotate_half**，同一份输入两者最大差 **2.478**（输入量级 3.9）⇒ **会静默算错，不报错**。
   * 不要打开 `SDPA_LOWMEM`：官方实测 1.22 s → 2.60 s（慢 2.1×）。
4. 前 5 条修复合计预期 **≈ 340 ms/步**（其中反向 **≈ 150 ms**），步时 1.166 s → **≈ 0.83 s**（PyTorch 0.677 s）。全部是"改内核/改接线"，**不需要改 autograd 契约**。

### 0.1 估算模型（后面所有 ms 都由它算出来，可复算）

```
cost(ms) = MB / 95GB/s * 1000 + kernels * 0.010 ms
```
95 GB/s 不是硬件参数，是**从两个独立约束拟合出来的等效带宽**：
* 约束 A：`silu_mul` 融合上线省了约 2 个 kernel + 约 240 MB/层（≈5.7 GB/步），对应实测 **+4.3% = 52 ms**（推算：改前 fwd 3 kernel/279 MB + bwd 3 kernel/359 MB，改后 2 kernel/199 MB + 1 kernel/199 MB，见 `__init__.py:3044-3065`）；
* 约束 B：`backward 723 ms − GEMM 底价 380 ms = 343 ms`，同期非 GEMM 流量 ≈ 26.9 GB/步。
两者给出的等效带宽分别是 **110 / 78 GB/s**，取 95 GB/s 作中值。**模型对前向偏高约 17%**（前向非 GEMM 实测 225 ms，模型给 264 ms），故本文所有数字按 ±25% 读。
另外两个"正确性"锚点（本文档 §5 复核）：`_rope_apply` 与 numpy rotate_half 逐位一致（max diff **0.0**）；checkpoint 开/关梯度 **0/39 不一致**（见 §8.3）。

---

## 1. 反向总账（按预期耗时排序）

### 1.1 反向逐项（这是"体检表"，Q1–Q5 是它的分项展开）

| # | 项 | 代码位置 | 每层 | ×24 层 | 内核/步 | 预期 ms | 可否避免 |
|---|---|---|---|---|---|---|---|
| 1 | **RoPE 反向**（未融合，见 §5） | `qwen.py:211-212` → `__init__.py:181-185` | 310 MB / 14 k | **7.44 GB** | 672 | **85** | ✅ 几乎全部 |
| 2 | **`softmax_lastdim_bwd`**（5 趟访存） | `cuda.py:3958` / 内核 `813-826` | 293.6 MB / 1 k | **7.05 GB** | 24 | **74** | ⚠️ 可压到 3 趟（-40%） |
| 3 | `silu_mul_bwd`（已融合，5 趟） | `__init__.py:3058` | 199.2 MB / 1 k | 4.78 GB | 24 | 51 | ❌ 固有 |
| 4 | **`swap_last2(p)`**（§2） | `__init__.py:892` | 117.4 MB / 1 k | **2.82 GB** | 24 | **30** | ✅ 换成 TN GEMM |
| 5 | `_acc` 累加 `add` | `__init__.py:1585` | 87 MB / 8 k | 2.09 GB | 192 | 24 | ⚠️ 可改原地（省分配） |
| 6 | `rmsnorm_bwd` ×2 | `cuda.py:5025` | 44 MB / 2 k | 1.06 GB | 48 | 12 | ❌ 固有 |
| 7 | `permute` 反向 ×4（§4） | `__init__.py:1349` | 33.6 MB / 4 k | 0.81 GB | 96 | 9 | ⚠️ 高风险 |
| 8 | **`dq/dk` 的 2 个 scale** | `__init__.py:2635-2636` | 16.8 MB / 2 k | 0.40 GB | 48 | 5 | ✅ 折进 alpha |
| 9 | **GQA `index_scatter_add` ×2**（§3） | `__init__.py:1481` | 15.7 MB / 4 操作 | 0.38 GB | 96 | 5 | ✅ 换分组求和 |
| 10 | `swap_last2(v)` | `__init__.py:890` | 2.1 MB / 1 k | 0.05 GB | 24 | 1 | ✅ 换成 NT GEMM |
| | **合计（非 GEMM）** | | | **26.9 GB** | **~1250** | **≈296** | |
| | **GEMM 底价**（2× 前向 194 ms + attention 反向） | | | | | **≈380** | |
| | **反向合计** | | | | | **≈676**（实测 723，模型低 7%） | |

> 说明：`_acc` 的"8 次/层"是解析计数（x 残差×2、h2×1、h×2、q_rope×1、k_pre×1、v×1）。
> CPU 追踪（L=1）实测 `accum=17`，但其中包含 CPU 版 RMSNorm 的 7 个分解节点；CUDA 上 RMSNorm 是单节点（`__init__.py:3279`），实际更接近 8–10。

### 1.2 前向对照（因为反向的一半浪费是前向结构的镜像）

| 项 | 位置 | ×24 层 | 内核/步 | 预期 ms |
|---|---|---|---|---|
| `softmax_lastdim`（3 趟） | `cuda.py:3914` / 内核 `796-811` | 7.05 GB | 24 | 74 |
| **RoPE 前向**（未融合） | `qwen.py:211-212` | **5.65 GB** | 480 | 64 |
| `silu_mul` 的 **clone** | `__init__.py:3053` | 1.91 GB | 24 | 20 |
| **`scores + mask`**（§2） | `__init__.py:2675` | **2.82 GB** | 24 | 30 |
| **`_attn_scores` 里的 `scale`** | `__init__.py:2627` | **2.82 GB** | 24 | 30 |
| `permute` ×4 | `qwen.py:205-207,222` | 0.81 GB | 96 | 9 |
| `broadcast_to(cos/sin)` ×4 | `__init__.py:794` | 0.70 GB | 96 | 8 |
| `rmsnorm_fwd` ×2 | `cuda.py:5014` | 0.70 GB | 48 | 8 |
| 残差 add ×2 | `qwen.py:234` / `__init__.py:757-763` | 0.70 GB | 48 | 8 |
| `+ bias` ×3 | `qwen.py:202-204` | 0.45 GB | 72 | 5 |
| GQA `index_gather` ×2 | `__init__.py:1474` | 0.40 GB | 48 | 5 |
| 合计 | | **24.0 GB** | **~1130** | **≈264**（实测非 GEMM 225，模型偏高 17%） |

---

## 2. Q1 —— `_sdpa_materialized` 与它的反向

### 2.1 前向一共做了 5 个操作（`__init__.py:2666-2677`）

| 步 | 代码 | 实际操作 | 分配 | 访存/层 |
|---|---|---|---|---|
| 1 | `2671` → `_attn_scores` `2625` | `cublasSgemmStridedBatched`(op=T)，batch=56/M=N=512/K=64。**不拷贝 k**（省掉了旧的 `k.swap_last2()`） | — | 0 |
| 2 | `2627` `rt.scale(buf, 0.125)` | 1 kernel 读 58.72 + 写 58.72 | **58.72 MB** | **117.4 MB** |
| 3 | `2675` `scores + _causal_mask(...)` | `add_broadcast`（`cuda.py:3360`，掩码按 stride=0 广播，**不物化 mask**）1 kernel | **58.72 MB** | **117.4 MB** |
| 4 | `2676` → `_softmax_cuda` `2736` | `softmax_lastdim` 1 kernel 3 趟 | **58.72 MB** | **293.6 MB** |
| 5 | `2677` `p @ v` | `bmm` → `sgemm_strided_batched`（`cuda.py:3801`），1 次调用 | 7.34 MB | 0 |

**前向：3 kernel + 2 cublas，176 MB 额外分配（3×58.72），528 MB/层（12.7 GB/步）。**
其中 **第 2、3 步纯属浪费**（235 MB/层），第 4 步可压缩一半。

### 2.2 反向一共 3 个 autograd 节点 + `_attn_scores`

| 步 | 代码 | 实际操作 | 访存/层 |
|---|---|---|---|
| A | `p@v` 的 bw `__init__.py:890` | `swap_last2(v)` **真拷贝** 1 kernel + `bmm` | 2.1 MB |
| B | `p@v` 的 bw `__init__.py:892` | `swap_last2(p)` **真拷贝 58.72 MB** + `bmm` | **117.4 MB** |
| C | softmax bw `2741` → `cuda.py:3958` | 1 kernel **2 趟**（dot、写） | **293.6 MB** |
| D | `scores+mask` 的 bw `757-763` | **0 kernel / 0 字节**：mask `requires_grad=False`，只把 `g` 透传给 `scores`；`_acc` 首次命中直接挂指针（`1585`） | **0** |
| E | `_attn_scores` bw `2631-2640` | `dq=NN`、`dk=TN` 两次 cublas（**都不拷贝**）+ 2 个 `scale` kernel | 16.8 MB |
| F | reshape / contiguous | `contiguous()` 在 CUDA 上 `return self`（`1353-1355`）；`reshape` 是视图（`cuda.py:3733-3748`） | **0** |

**反向：5 kernel + 4 cublas，430 MB/层（10.3 GB/步）。**

### 2.3 结论：哪些是可以避免的

| 判断 | 结论 |
|---|---|
| **"反向重算 scores"** | **不存在**。反向只读 `p`（softmax 输出）和 `v`，不重算 `scores`。代价是 `p` 必须活到反向（24×58.72 MB = **1.41 GB 常驻**）。 |
| **"对 mask 求梯度"** | **不存在**。`_add_cuda` 的 bw（`757-763`）对 `other.requires_grad=False` 直接跳过；mask 连 `grads` 字典都进不去（`_acc` 没被调用）⇒ **0 kernel / 0 字节**。这条可以从怀疑名单里划掉。 |
| **"多余的 contiguous/reshape 触发拷贝"** | **不存在**。`contiguous()` 是 no-op（`1353`），`reshape`/`view` 是视图（`cuda.py:3733`，`view_of=`）。真正触发拷贝的只有 `permute` 和 `swap_last2`（§4）。 |
| ✅ **可避免 1** | `swap_last2(p)` → 直接用 `sgemm_tn_strided_batched(p, g, BH, S, DH, S)`（`cuda.py:3898` 已有，语义就是 `p^T @ g`）。**省 117.4 MB/层 + 24 kernel/步 = 2.82 GB，≈30 ms** |
| ✅ **可避免 2** | `swap_last2(v)` → `sgemm_nt_strided_batched(g, v, BH, S, S, DH)`。省 2.1 MB/层 |
| ✅ **可避免 3** | `dq/dk` 的两个 `scale`：`_bmm_op` 已经支持 alpha（`cuda.py:3852`，现在硬编码 1.0），把 `scale` 传进去即可。省 16.8 MB/层 + 48 kernel/步，≈5 ms |
| ✅ **可避免 4** | 前向 `scores` 的 `scale`（117.4 MB/层）+ mask add（117.4 MB/层）→ 见 §6 的 causal-softmax 单内核 |
| ⚠️ 半可避免 | `softmax_lastdim_bwd` 5 趟 → 3 趟（smem 缓存 P 行）或 2.5 趟（再加 causal 跳过） |
| ⚠️ 只在"偏导需求"时才浪费 | `2632-2633` **无条件**算 `dq` 和 `dk`，即使只有一个需要梯度。本模型两者都要，不是问题；但 LoRA/冻结 q 的场景会白算一次 `[S,S]×[S,DH]` |

---

## 3. Q2 —— `repeat_interleave` 实现的 GQA（KV=2 → NH=14，7×）

代码：`qwen.py:214-215` → `__init__.py:1468-1483`（`_repeat_interleave_cuda`）+ 计划 `1698-1727`（`_repeat_gather_plan`）。

### 3.1 次数与字节

| 方向 | 每次 | 每层 | ×24 层 | 内核/步 |
|---|---|---|---|---|
| **前向** | `index_gather`（`cuda.py:3556`）1 kernel：写 [4,14,512,64]=**7.34 MB**、读 [4,2,512,64]=1.05 MB ⇒ **8.39 MB** | 2 次（k、v）= 16.8 MB | **403 MB** | **48** |
| **反向** | `index_scatter_add`（`cuda.py:3578`）= `zeros(1.05 MB)` 的 **memset** + 1 kernel 读 7.34 MB、按 7 路 `atomicAdd` 写回 1.05 MB ⇒ **15.7 MB / 2 个 GPU 操作** | 2 次 = 31.4 MB | **755 MB** | **96** |
| **合计** | | | **1.16 GB** | **144** |

额外代价（容易被忽略）：
* **常驻显存**：24 层 × 2 × 7.34 MB = **352 MB** 的 k/v 副本要活到反向（它们是 score GEMM 的输入）。
* **7 路原子冲突**：同一源地址被 7 个相邻输出原子累加，`index_scatter_add8`（内核 `cuda.py:1277`）的真实代价会**高于**字节模型（L2 原子吞吐），这一项我的 5 ms 估计偏乐观。

### 3.2 能不能在 kernel 索引层面免掉（不改 autograd 契约）

四条路，按可行性排：

1. **反向换"分组求和"内核**（低风险，**不改契约**）：`repeat_interleave` 沿 head 轴是**连续 7 个**，所以反向等价于"沿 head 轴对 7 个步长相同的元素求和"——一个无原子、无 `zeros` 的 kernel。访存 7.34 读 + 1.05 写 = **8.4 MB/call**（省掉 7.3 MB 的原子写和 memset）。**省 ≈ 350 MB/步**。
2. **前向把 `rope(k)` 与展开融合**（中风险，不改契约）：`k_pre = rope(k)` 之后立刻展开，两步可以合成"读 [B,KV,S,DH] → 旋转 → 写 [B,NH,S,DH]"一个 kernel。**省一次 gather + 7.34 MB/call**。
3. **按 kv 组循环 cuBLAS**（不改契约，但更慢）：`strideB=0` 可以让 7 个 q 头共用同一份 k，但必须按 `B*KV=8` 组分别调用 → 前向 8 次 cublas 换掉 1 次；反向 `dk/dv` 需要 7 次 `beta=1` 累加（`strideC=0` 有写竞态，cuBLAS 不允许）→ 反向变 56 次调用。**得不偿失**。
4. **真正 0 拷贝：给 attention 内核加 `h_kv` 参数**（终局）：flash 类内核都这么做（`k_head = h / (NH/KV)`）。现有 `attn_nop_*`（`cuda.py:5569-5648`）**没有**这个参数，`_sdpa_lowmem`（`__init__.py:2704-2706`）自己也要先 `permute` 再调；物化路径的单次 `cublasSgemmStridedBatched` 也表达不了（`strideB` 必须是常数）。**要吃满这项必须改内核，或改 autograd 契约给"逻辑 stride-0 视图"（`GpuBuf` 目前是打包连续的，没有 stride 概念）。**

> 结论：**当下最划算的是 1+2（省 ≈ 0.75 GB + 48 个操作），第 4 条要等 flash 内核先变快**（现在 2.1× 慢于物化）。

---

## 4. Q3 —— `permute` / `swap_last2` / `transpose` 是视图还是真拷贝

| 算子 | 判定 | 证据 |
|---|---|---|
| `Tensor.permute` (CUDA) | **真拷贝** | `__init__.py:1338-1341` → `cuda.py:3496-3529`：`out = self.empty(out_shape)` + `permute8`/`permute8_v4` 内核 |
| `Tensor.swap_last2` (CUDA) | **真拷贝** | `__init__.py:1075-1080` → `cuda.py:3750-3766`：分配新缓冲 + `swap_last2` 内核 |
| `Tensor.transpose()` 无参 2D | **惰性视图** | `__init__.py:991-1005`：只建节点、翻 `_lazy_shape`，`_gpu` 复用原缓冲。只有走非 2D-matmul（`_materialize_lazy_T` `943`、`_cuda_logical_storage` `1447`、`to_numpy` `1576`）才 `transpose2d` 落地成拷贝（`cuda.py:3291`） |
| `reshape` / `view` | **视图** | `cuda.py:3733-3748`（`view_of=buf`，只改 shape 元数据） |
| `contiguous()` | **no-op** | `__init__.py:1353-1355`：CUDA 直接 `return self` |
| `__getitem__`（切片） | **真拷贝** | `__init__.py:1454-1457` → `index_gather` |

### 4.1 `_layer` 里的调用次数与字节

| 调用点 | 次数/层 | 每次张量 | 每次访存 | ×24 层（单向） |
|---|---|---|---|---|
| `q.permute(0,2,1,3)` `qwen.py:205` | 1 | 7.34 MB | 14.68 MB | 352 MB |
| `k.permute(...)` `qwen.py:206` | 1 | 1.05 MB | 2.10 MB | 50 MB |
| `v.permute(...)` `qwen.py:207` | 1 | 1.05 MB | 2.10 MB | 50 MB |
| `o.permute(0,2,1,3)` `qwen.py:222` | 1 | 7.34 MB | 14.68 MB | 352 MB |
| **前向小计** | **4** | | **33.56 MB** | **805 MB / 96 kernel / 9 ms** |
| **反向小计**（`_permute_cuda.bw` `1349`，4 次） | **4** | | **33.56 MB** | **805 MB / 96 kernel / 9 ms** |
| `swap_last2(p)` `__init__.py:892` | 1 | 58.72 MB | **117.44 MB** | **2.82 GB / 24 kernel / 30 ms** |
| `swap_last2(v)` `__init__.py:890` | 1 | 1.05 MB | 2.10 MB | 50 MB / 24 kernel / 1 ms |
| `.T`（7 个权重）`qwen.py:198-200,223,226-227,234` | 7 | — | **0**（惰性 + `gemm_nt`/`gemm_tn` 直接吃原始缓冲，`__init__.py:895-914`） | **0** |

> 所以"`transpose` 在 `_layer` 里被调用 7 次、每次 0 字节"——这条是**干净的**（旧的"物化 W^T 545 MB"确实修掉了）。
> 真正在流血的是 `permute`（1.6 GB/步往返）和 `swap_last2(p)`（2.8 GB/步）。

---

## 5. Q4 —— RoPE：`_rope_apply` 在反向要几个内核

### 5.1 现状拆解（`qwen.py:180-185`）

```python
rot = T.cat([-x[:, :, h:], x[:, :, h:]], dim=-1)   # 184
return x * cos_t + rot * sin_t                       # 185
```

| 阶段 | 具体操作 | kernel | 访存 |
|---|---|---|---|
| `x[:, :, h:]` | `index_gather`（`1457`） | 1 | 3.67 + 3.67 |
| `-…` | `_scale_cuda`（`767`，`self * -1.0`） | 1 | 3.67 + 3.67 |
| `x[:, :, :h]` | `index_gather` | 1 | 3.67 + 3.67 |
| `T.cat` | `_cat_cuda`（`1937`）：1 次 `empty(7.34MB)` + 2×`cat_place`（`cuda.py:3624`） | **2** | 2×(3.67+3.67) |
| `x * cos_t` | **`broadcast_to(cos, [56,512,64])`（`794`）+ `mul`（`796`）** | **2** | 7.47 + 22.02 |
| `rot * sin_t` | 同上 | **2** | 7.47 + 22.02 |
| `+` | `_add_cuda` → `ewise_add` | 1 | 22.02 |
| **前向合计** | | **10** | **117.7 MB** |

**反向（q 与 rot 都需梯度）：**

| 阶段 | kernel | 访存 |
|---|---|---|
| `add` 的 bw | 0（`_acc` 两个不同父节点，各存指针） | 0 |
| `mul(x,cos)` 的 bw：`broadcast_to(cos)` + `mul` + `unbroadcast`(sum_axis) | 3 | 36.96 |
| `mul(rot,sin)` 的 bw：同上 | 3 | 36.96 |
| `cat` 的 bw：2×`cat_extract`（`cuda.py:3652`） | 2 | 22.02 |
| `neg` 的 bw：`scale`（只作用于半宽） | 1 | 7.34 |
| 2×`__getitem__` 的 bw：`index_scatter_add`（zeros+原子） | 4（+2 memset） | 29.4 |
| `_acc` 回 q_pre 的 1 次 add | 1 | 22.02 |
| **反向合计** | **14（+2 memset）** | **≈155 MB** |

**×24 层 ×2 次调用（q、k）：前向 480 kernel / 5.65 GB，反向 672 kernel / 7.44 GB。**

### 5.2 换 half-split 融合内核能省多少

目标形态：**前向 1 个 clone + 1 个 kernel（29.4 MB/次），反向 1 个 kernel（22.02 MB/次）**。

```
省 = (235.4 − 58.8) + (310 − 44.0) = 176.6 + 266 = 442 MB/层  ⇒ 10.6 GB/步
     kernels: (20−6) + (28−2) = 40/层 ⇒ 960/步
ms  = 10.6 GB / 95 GB/s + 960×0.010 ≈ 112 + 9.6 ≈ 122 ms（其中反向 ≈ 74 ms）
```

### 5.3 ⚠️ 三个必须注意的坑（都实测过）

1. **现有 `T.fused_rope` 是交错式，与 Qwen 的 rotate_half 不等价。**
   同一份输入的实测：`max|interleaved − rotate_half| = 2.478`（输入量级 3.9）。
   核 `fused_rope_ip`（`cuda.py:1589-1603`）是 `out[2k]=x[2k]c−x[2k+1]s`，而 Qwen 需要 `out[i]=x[i]c−x[i+h]s`。
   **直接接线不会报错，只会静默把模型算错**（`examples/qwen25_sft.py:17` 的对照表也写了"引擎交错式 vs 标准 rotate_half"）。
2. **`fused_rope` 要求 `cos.n == x.n`**（`cuda.py:5089`），而 `_cos_sin` 给的是 `[1,S,DH]`。
   直接传会 `CudaError` → 被 `_fused_warn_once` 吞掉 → **静默回退到 `_rope_composed`**，那条路还多一个 `x @ P` 的 GEMM，**比现在更慢**。
   解法：① 每步把 cos/sin 展开成 `[BH,S,DH]` 缓存一次（14.7 MB/步，可忽略）；或 ② 内核按 `(row = idx/(S*DH))` 索引 `[S,DH]` 的小表（只读 131 KB，从 L2 命中）。
3. **half-split 的正确公式**（我已验证）：
   ```
   fwd:  out[i]   = x[i]*c[i]   - x[i+h]*s[i]
         out[i+h] = x[i+h]*c[i+h] + x[i]*s[i+h]      # 表是 [c,c] 重复，故 c[i+h]=c[i]
   bwd:  dx[i]   = g[i]*c[i]   + g[i+h]*s[i+h]
         dx[i+h] = g[i+h]*c[i+h] - g[i]*s[i]
   ```
   验证结果：`half-split vs rotate_half` 前向 max diff = **0.0**；反向 vs float64 数值微分最大绝对误差 **3.3e-09**（|g|max=2.9）。

---

## 6. Q5 —— `softmax_lastdim` 的前向/反向各有几趟，能否合并

### 6.1 现状（`cuda.py` 内核 `796-811` / `813-826`）

| | 趟数 | 逐趟做什么 | 行内访存（k=512） | 每层字节（[56,512,512]） |
|---|---|---|---|---|
| **前向** | **3** | ① 求 max ② `exp` + 写 y + 求和 ③ `y *= 1/s` | 读 x、读 x + 写 y、读 y + 写 y = **5k** | **293.6 MB** |
| **反向** | **2** | ① `dot = Σ g·y` ② `dx = y·(g−dot)` | 读 g,y、读 g,y + 写 dx = **5k** | **293.6 MB** |

合计 7.05 GB/步（前向）+ 7.05 GB/步（反向）= **14.1 GB/步**，是**全模型单项第一**。

### 6.2 能合并，而且有三层递进

1. **单趟 + 行内 smem/寄存器复用**（不需要算法改动）：
   k=512 一行 2 KB；一行一个 warp、8 warp/块 ⇒ 16 KB smem/块，V100 的 96 KB/SM 完全放得下。
   前向：读一次进 smem → 求 max/sum → 写一次 = **2k**（5k → 2k，**-60%**）。
   反向：读 g,y（2k）+ 写 dx（1k）= **3k**（5k → 3k，**-40%**）。
2. **causal 跳过**（现成代码：`attn_mask_softmax` 内核 `1638-1657`，`pos = r % k`，`for j<=pos`）：
   只处理下三角 ⇒ 再乘 ≈0.5。
3. **把 mask 并进内核**（同一内核 `1638-1657` 直接把 `j>pos` 写 0）⇒ 省掉 `scores + mask` 这个 **117.4 MB + 1 kernel**。

三层一起做（**不含** `scores` 的 `scale`，那条归第 3 条）：
* 前向：`(mask 117.4 + softmax 293.6) = 411 MB` → **≈59 MB**（半行、2 趟）⇒ 省 **352 MB/层 = 8.45 GB/步 ≈ 89 ms**；
* 反向：`softmax 293.6` → **≈147 MB**（半行、3 趟）⇒ 省 **147 MB/层 = 3.5 GB/步 ≈ 37 ms**；
* 合计 **≈126 ms**（前向 89 / 反向 37）。

### 6.3 现成件为什么没被用上

`T.attn_mask_softmax`（`__init__.py:3205-3239`）条件写死为 `scores.ndim == 2`，而 `_sdpa_materialized` 拿到的是 `[56,512,512]`。
但它的 `pos = r % k` 语义**正好**对应 `reshape(56*512, 512)` 的行主序 ⇒ **一次视图 reshape 就能用**（0 拷贝）。
唯一的额外成本是它内部 `rt.clone(scores._gpu)`（`3216`）会多一次 58.7 MB 拷贝；若允许"就地消费 scores 缓冲"（反向只用 `p` 和 `v`，`scores` 无第二消费者），这一次 clone 也能省掉。

> 附注：`_softmax_composed`（`3242-3253`）在**每次调用**都 `np.zeros((rows,k))` 再 `Tensor(mask)`，在 CUDA 上等于**每次 59 MB H2D**。它只作为 `attn_mask_softmax` 的回退路径（`3213`）存在，默认路径走不到，但谁把 `requires_grad` 关掉做推理就会踩到——建议顺手改成复用 `_CAUSAL_MASK_CACHE`。

---

## 7. Q6 —— 按收益排序的 5 条修复清单

> 排序依据 = §0.1 模型算出的 **ms/步（前向+反向合计）**。所有条目**都不需要改 autograd 契约**。
> 复算公式：`ms = MB/95 + kernels*0.010`；括号里给了"反向单独省多少"。

### 第 1 条：RoPE 换 half-split 融合内核 —— 预期 **122 ms**（反向 74 / 前向 49）

* **改哪里**：`qwen.py:211-212` 改成调用新的 `T.rope_half(x, cos, sin)`；`cuda.py` 的 `_FUSED_KERNELS` 里新增 `rope_half_fwd` / `rope_half_bwd`（沿用 `fused_rope_ip` 的写法，`cuda.py:1589/1610` 只改下标）。
* **依据**：前向 480 kernel / 5.65 GB + 反向 672 kernel / 7.44 GB → 前向 6 kernel / 58.8 MB/层、反向 2 kernel / 44.0 MB/层（§5.1 逐项）。
* **风险**：**中**。约定写错是**静默错**（不是崩溃），且 `cos.n` 不匹配会静默回退到更慢的 `_rope_composed`。
* **怎么验证**：
  1. CPU 公式对拍（脚本见 §8.1，`half-split == rotate_half` 必须是 0.0；反向 vs float64 数值微分 < 1e-8）；
  2. `TG_RUN_CUDA_TESTS=1 python -m pytest tests/test_backward_parity.py tests/test_qwen25_forward.py -q`（前者对 PyTorch float64 逐个参数比梯度，阈值 1e-6 相对）；
  3. A/B：`TG_NO_FUSED=1` 走老路径，同 seed 比 `dW` 的 `max|Δ|/max|W|` 与 20 步 loss/gnorm 曲线。

### 第 2 条：causal-softmax 单内核（mask + softmax 一起吃掉）—— 预期 **126 ms**（反向 37 / 前向 89）

* **改哪里**：`__init__.py:2671-2676`；把 `scores` 视图 reshape 成 `[BH*S, S]` 后交给 `T.attn_mask_softmax`（`3205`）/直接调 `rt.attn_mask_softmax_(scores_buf, causal=True)`（`cuda.py:5159`），去掉 `2675` 的 `scores + mask`。
* **依据**：§6.2 的三层递进（2 趟 fwd / 3 趟 bwd / 半行 / 去掉 117.4 MB 的 mask add）。
* **风险**：**中**。`attn_mask_softmax_` 是**原地**写（`5167 _split_forget_ptr`），必须确认 `scores` 无第二消费者（当前确实没有）；另外它的反向内核（`1661`）目前仍扫全行，要拿到"bwd −40%"需要加 `pos` 跳过。
* **怎么验证**：
  1. `TG_RUN_CUDA_TESTS=1 python -m pytest tests/test_sdpa_lowmem.py -q`（同一份输入比 fwd/dq/dk/dv，阈值 1e-5 相对——注意该文件注释里的教训：**必须同一份输入**）；
  2. `python -m pytest tests/test_backward_parity.py -q`；
  3. 用 `rt._kernel_count` 前后对比（应 -48/步），以及 `rt.pool_stats()` 看峰值。

### 第 3 条：SDPA 去拷贝 + 去 scale —— 预期 **64 ms**（反向 34 / 前向 30）

* **改哪里**：
  * `__init__.py:890` → `rt.sgemm_nt_strided_batched(g, v, BH, S, S, DH)`；
  * `__init__.py:892` → `rt.sgemm_tn_strided_batched(p, g, BH, S, DH, S)`；
  * `__init__.py:2627`（前向）与 `2635-2636`（反向）→ 把 `scale` 作为 alpha 传给 `_bmm_op`（`cuda.py:3845`，`3852-3853` 现在硬编码 `alpha=1.0`）。
  * 注意：`890/892` 是 `_matmul_cuda` 的**通用 3D 分支**（`877-894`），不要为了这一处改动把 `bmm` 的语义改掉——建议在 `_sdpa_materialized` 里显式写 VJP，而不是改 `_matmul_cuda`。
* **依据**：`swap_last2(p)` 117.44 MB + `swap_last2(v)` 2.10 MB + `dq/dk` scale 16.8 MB + 前向 `scores` scale 117.4 MB，×24。
* **风险**：**低**。`sgemm_nt/tn_strided_batched` 是既有 API，语义一一对应；alpha 是 `_bmm_op` 的既有参数。
* **怎么验证**：写一个 `p@v` 的单算子对拍（float64 参考，比 `dp/dv`），再跑 `test_backward_parity.py`；`TG_NO_FUSED` 不覆盖这条，建议加一个 `TG_SDPA_LEGACY=1` 开关做 A/B。

### 第 4 条：`silu_mul` 免 clone —— 预期 **20 ms**（前向）

* **改哪里**：`__init__.py:3053` 现在 `rt.silu_mul_fwd(rt.clone(a._gpu), b._gpu)`；在 `cuda.py:5038` 旁边加一个非原地版 `silu_mul_fwd_out(a, b, out)`（核 `silu_mul_fwd` 是 `a[i]=silu(a[i])*b[i]`，改成三指针即可）。
* **依据**：39.85 MB × 2（clone 的读+写）× 24 层 = 1.91 GB/步。
* **风险**：**低**（新内核 + 老路径保留 + `_fused_warn_once` 回退）。
* **怎么验证**：`python -m pytest tests/test_nn_ops.py tests/test_cpu.py -q`；GPU 上加一条逐位对拍（公式不变，应 bit-exact）。

### 第 5 条：GQA `repeat_interleave` 瘦身 —— 预期 **7 ms** + 352 MB 常驻

* **改哪里**：`__init__.py:1481` 的 `index_scatter_add` 换成"沿 head 轴分组求和"内核（无原子、无 `zeros`）；`qwen.py:214-215` 可进一步把 `rope(k)` 与展开融合成一个 kernel。
* **依据**：前向 403 MB / 48 kernel，反向 755 MB / 96 操作（§3.1）。改完反向降到 ≈8.4 MB/call（省 350 MB/步），前向少一次 gather。
* **风险**：**中**（新内核；但契约不变，仍是同一个 autograd 节点）。
* **怎么验证**：`python -m pytest tests/test_qwen_grad_ckpt.py tests/test_backward_parity.py -q`（GQA 展开/合并写错会被梯度对拍直接抓到）；再单独对拍 `repeat_interleave` 的 fwd/bwd（`tests/test_cpu.py` 里已有 CPU 版参照）。

### 没进前五、但值得排期的两条

| 项 | 预期 | 为什么不进前五 |
|---|---|---|
| `permute` 4 处视图化/融合（§4） | **19 ms**（前 9 / 反 9） | 需要给 `GpuBuf` 加 stride 视图语义 ⇒ **要改 autograd 契约**，风险高、收益中等。短期只能靠"把 permute 融进相邻内核"（例如 out_proj 前的那次）。 |
| `_acc` 改原地 `add_`（`1585`） | **0 ms** | 只省分配次数（8–10 次/层 × 24 = 200–240 次/步），不改带宽。低风险，可顺手做。 |

### 一个"不要做"的清单

* **不要开 `SDPA_LOWMEM`**：`docs/GAP_VS_PYTORCH.md:112` 实测 1.22 s → **2.60 s**（慢 2.1×）。它省显存（-5.4 GiB），但 flash 内核在 S=512 上比物化慢；`_sdpa_lowmem`（`__init__.py:2694`）自己还多 4 次 `permute`（`2704-2708`）。
* **不要直接接 `T.fused_rope`**（§5.3 坑 1）。

---

## 8. 附录：复现与口径

### 8.1 我在本机跑过的三个验证脚本（都在 `%TEMP%\audit_bwd\`，未写仓库）

| 脚本 | 作用 | 结果 |
|---|---|---|
| `trace_order.py` | 用 tiny 配置在 **CPU** 上构建 `_layer` 的完整计算图，按"闭包所在函数"归属每个节点，打印**反向实际执行顺序 + 输出形状** | 单层反向 **97 个节点**；`_acc` first=110 / accum=17（含 CPU 版 RMSNorm 的 7 个分解节点） |
| `rope_check.py` / `rope64.py` | RoPE 约定对拍 | `_rope_apply` vs numpy rotate_half = **0.0**；interleaved vs rotate_half = **2.478**；half-split vs rotate_half = **0.0**；half-split 反向 vs float64 数值微分 = **3.3e-09** |
| `final_numbers.py` | §0.1 模型的算术表 | 反向非 GEMM ≈296 ms；GEMM 380 ms ⇒ 676（实测 723，低 7%） |

CPU 追踪只用来**校验图结构与节点数**，所有 kernel/字节数都是按 CUDA 分支（`*_cuda`）的实现逐个算的；两者在 `_layer` 里结构一致，只有两处实现差异（`_attn_scores` 是融合节点、RMSNorm 是单节点），已在正文里说明。

### 8.2 建议的验证命令

```bat
:: 1) CPU 全量（默认 292 条，不需要 GPU）
python -m pytest tests -q

:: 2) GPU 专项（需在 V100 机器上）
set TG_RUN_CUDA_TESTS=1
python -m pytest tests/test_sdpa_lowmem.py tests/test_checkpoint_cuda.py -q

:: 3) 反向数值对拍（需 torch；逐个参数对 PyTorch float64，阈值 1e-6）
python -m pytest tests/test_backward_parity.py -q

:: 4) 融合核 A/B（现成的总开关）
set TG_NO_FUSED=1     &  python <bench>     :: 老路径
set TG_NO_FUSED=0     &  python <bench>     :: 新路径
```

分阶段计时沿用 `docs/GAP_VS_PYTORCH.md:141` 的方法（每段 `sync` 后计时、跑完释放），并额外打印：

```python
print("kernels/step =", rt._kernel_count)      # 纯内核发射数，用来验证"少 ~960 个"
print("pool =", rt.pool_stats())               # 峰值/命中率，验证分配次数
```

### 8.3 顺手复核的一个"已修项"（结论：确实修好了，但很脆）

用户说"checkpoint CUDA 生效"。我用同权重对拍 `GRAD_CKPT=0` vs `=1`（L=3，`CKPT_SEGMENTS=L`）：**39 个叶子全部 0.0 差异**。
机制值得记一笔：`checkpoint.bw`（`__init__.py:2570-2578`）调 `re.backward(g)`，而 `_backward_cuda`（`1038-1060`）对"**有 `_backward` 的节点**"一律调用、**不因 `_parents=()` 而停**，所以能递归进上一段的闭包。
⇒ **脆弱点**：谁要是把 `_backward_cuda` 改成"叶子就停"（看起来更符合 PyTorch 语义），checkpoint 会**静默丢梯度**。建议在 `checkpoint` 的 docstring 里把这条不变量写死，并保留 `test_checkpoint.py`。
另外注意：`checkpoint` 的 `bw` 里有嵌套 `_backward_cuda`（`1007`）⇒ 每段结束一次 `rt.sync()`（`1061`），`CKPT_SEGMENTS=1`（24 段）时**每步 24 次全流水线排空**，虽然只有 ~0.5 ms，但会打掉 host/GPU 重叠。

### 8.4 本报告的局限（必须说明）

1. **没有 GPU**：所有 ms 都是 §0.1 的解析估算，不是实测。等效带宽 95 GB/s 是拟合值（不是 V100 的流式带宽上限），模型对前向偏高约 17%（对反向偏低约 7%）。
2. **没测分配器**：`_pool`（`cuda.py:2713-2996`）的命中率、`cuMemAlloc` 抖动没实测；`silu_mul` 那次 +4.3% 里有相当一部分可能是"少几个 40 MB 临时缓冲 → 池命中率上升"，这部分收益**无法用字节模型解释**，也没有被本文计入任何一条修复的预估（属于额外红利）。
3. **`index_scatter_add` 的原子代价**：字节模型低估了 7 路 `atomicAdd` 的真实开销，第 5 条的 7 ms 偏乐观。
4. **未核对 `attn_nop_*` flash 内核的慢因**：只引用了官方实测 2.1×，没有深入（不在本次范围）。

---

*报告生成：2026-11-02 · 只读审计，未修改 `_tgai_orv` 内任何文件。*
