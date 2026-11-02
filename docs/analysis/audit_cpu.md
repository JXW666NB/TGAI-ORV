# CPU 侧开销静态审计：发射 1330 个 kernel 的那一段

- **审计对象**：`D:\DSH-Workspace\TGAI-对话档案\_tgai_orv`（只读，未改任何仓库文件）
- **方式**：纯静态读码 + 与本仓库自带的实测文档/工具对账（`tools/profile_kernel.py`、`tools/profile_step.py`、`docs/PRODUCTION_READINESS.md`、`docs/GAP_VS_PYTORCH.md`）
- **口径**：Qwen2.5-0.5B 形状（L=24 / D=896 / NH=14 / KV=2 / DH=64 / FF=4864 / V=151936 / TIE=1 / QKV_BIAS=1），B=4 / S=512，fp32 + AdamW，`GRAD_CKPT=0`，`SDPA_LOWMEM=0`，`FUSED_OPS=1`，`FUSED_ROPE=1`，即 `profile_kernel.py` 默认那条路径（**不含**梯度裁剪）
- **标注约定**：`[实测]` = 仓库里已有的测量值；`[估算]` = 我按代码结构推的，误差能到 ±50%，给的是数量级
- **本轮没跑 GPU**：本文所有 µs 级数字都是估算，唯一锚点是仓库自己测的 **10.3 µs/次发射（V100）**

---

## 0. 三条必须先说清楚的事（其中第一条会改变优化方向）

### 0.1 「GPU 只忙 111.6 ms ⇒ 85% 是 CPU」这一步推理不成立

先做一次自洽性检验，只用题面给的两个数：

```
744 ms - 111.6 ms = 632.4 ms 分摊到 1330 次发射 = 475 µs / 次发射
```

而这个仓库自己量过单次发射的宿主时间（`docs/PRODUCTION_READINESS.md:704-708`，`tools/profile_step.py` 的 host 侧计时）：

| 指标 | 4090 | V100 |
|---|---|---|
| kernel 启动 | 348 次/步 | 348 次/步 |
| host 侧总时间 | 2.88 ms/步 | **3.58 ms/步** |
| **每次启动平均** | **8.3 µs** | **10.3 µs** `[实测]` |

**475 µs 是 10.3 µs 的 46 倍**。纯 Python + ctypes 的发射不可能贵 46 倍（下面 §1 拆开算，也就 ~10 µs/次）。所以 632 ms **不可能**是发射开销。

### 0.2 原因：`profile_kernel.py` 只挂在 `launch()` 上，cuBLAS 完全在视野之外

`tools/profile_kernel.py:45-58` 干的事是替换 `CudaRuntime.launch`，前后各记一个 CUDA event，用 `cuEventElapsedTime` 求和。而本引擎的**矩阵乘根本不过 `launch()`**：

| 路径 | 代码 | 是否走 `launch()` | 是否被 profiler 计时 |
|---|---|---|---|
| `sgemm / gemm_nt / gemm_tn` | `cuda.py:4740 / 4760 / 4779` → `_sgemm` `cuda.py:4639` → `cublasSgemm_v2` | ❌ | ❌ |
| `bmm`（≥4 batch） | `cuda.py:3906` → `sgemm_strided_batched` `cuda.py:3939` → `cublasSgemmStridedBatched` | ❌ | ❌ |
| `_attn_scores` | `__init__.py:2625` → `sgemm_nt_strided_batched` | ❌ | ❌ |
| 逐元素 / 归约 / 拷贝 | `launch("ewise_add_v4", ...)` 等 | ✅ | ✅ |

也就是说 **`kernels = 1330` 只统计了「自研内核」的发射次数，`gpu_sum = 111.6 ms` 只统计了自研内核的 GPU 时间**，而这一步真正吃 FLOP 的 648 次 cuBLAS GEMM（前向 9×24、反向 18×24）——**既不在 1330 里，也不在 111.6 ms 里**。

### 0.3 把 GEMM 那部分补回来：500–600 ms 量级

按 0.5B 参数、2048 token 估（`2·N·tokens` 前向，反向 ≈ 2× 前向）：

```
前向  = 2 × 4.84e8 × 2048 ≈ 1.98 TFLOP
反向  ≈ 2 × 前向          ≈ 3.97 TFLOP
合计                      ≈ 5.95 TFLOP
```

V100 fp32（SGEMM，无 tensor core）峰值 15.7 TFLOPS，cuBLAS 实际 10–12 TFLOPS：
⇒ **GEMM 时间 ≈ 500–600 ms**。

于是 744 ms 的分解更像这样 `[估算]`：

| 项 | ms | 依据 |
|---|---|---|
| cuBLAS GEMM（前向+反向） | **500–600** | FLOPs / ~11 TFLOPS |
| 自研 kernel 的 GPU 时间 | **111.6** | `[实测]` profiler |
| CPU 发射 + autograd 建图 + 分配器记账 | **~40–90** | 1330×10.3 µs = 13.7 + 其余 |
| 合计 | ≈ 700–800 | 与 744 自洽 |

**三个独立旁证**（都不依赖我的 FLOP 估算）：

1. **同进程同配置，PyTorch 也要 0.610 s/步**（`docs/GAP_VS_PYTORCH.md` §10.1）。PyTorch 的 C++ 派发 ~5 µs/算子、几乎零 Python 开销，它慢只能是因为 GPU 真在忙。我们 744 ms = torch 的 1.22×，这个倍率正是「非 GEMM kernel 更多」的量级，不是「CPU 发射慢 40 倍」的量级。
2. **开 CUDA Graph 实测只省 4%（V100）/ 12.9%（4090）/ 0.5%（A10）**（`docs/PRODUCTION_READINESS.md:1251-1253`）。Graph 把全部发射压成 1 次，如果发射真是 632 ms，Graph 应该省 600 ms 而不是 4%。
3. **A10 那次对照实验的结论**（同文件 1251 行）："假设 1：Python 侧启动开销（348 次/步 × 8 µs ≈ 2.8 ms）→ 开 CUDA Graph 实测 64.4 → 64.1 ms（只省 0.5%）⇒ **假设不成立**"。

### 0.4 所以本报告能提供的"CPU 削减"上限是多少

- 发射本身：**1330 × 10.3 µs ≈ 13.7 ms**（占 wall 1.8%）
- autograd 建图 / `_acc` / 分配器记账 / cublas 宿主侧参数组装：**合计约 30–80 ms** `[估算]`
- ⇒ **CPU 侧可削减总量 ≈ 40–90 ms，乐观把能砍的都砍掉，wall 744 → 约 690–720 ms（−3%~7%）**

**想把 744 ms 打到 200 ms，只能动 GEMM（fp16/split-K/更大 batch），或者动"非 GEMM 但很慢"的那几个 kernel（`adamw_update` 18.8 + `sum_axis8` 14.5 + `softmax_lastdim_bwd` 8.8 = 42 ms，结构性问题，能压到 ~10 ms）。** 详见 §5.6。

### 0.5 60 秒验证（建议先做这个，再决定投人力方向）

三条任选，都只加几行、不改数值：

```python
# A) 给 cublas 也挂 event：复制 prof_kernel.py 的 prof_launch，改成包 CudaRuntime._sgemm
#    + sgemm_strided_batched + _bmm_op，重跑一次，看 gpu_sum 变成多少
# B) 或者最省事：跑 20 步，同时开一个窗口
#    nvidia-smi dmon -s u -d 1        # 看 sm__inst_executed / GPU util 是否长期 > 80%
# C) 或者直接量 CPU 侧：把 step() 里的 rt.sync() 去掉，只计"发完所有调用"的墙钟
#    t0=pc(); step_no_sync(); t1=pc()   # t1-t0 ≈ CPU 侧总开销（不含等 GPU）
```

判据：若 B 的 GPU util 长期 >80%，或 C 的 `t1-t0` 只有几十 ms，则 §0.3 的结论成立，本报告的 §5 应只做其中"顺手且低风险"的几条。

---

## 1. Q1：`launch()` 每次调用的固定开销，逐项拆开

### 1.1 先说一个前提：参数模板缓存**已经有了**，而且默认开

`cuda.py:1879-1893`（M8 注释）+ `cuda.py:3322-3346`（快路径）已经实现了"按 (kernel 名, 参数类型序列) 缓存 `holders/ptrs`，每次只改 `.value`"，开关 `TG_LAUNCH_FAST` 默认开（`_launch_fast_enabled()` 只在 `"0"/"false"/"no"/"off"` 时关闭）。

所以 **"launch 参数缓存"这条已经落地了，不用再提**。剩下的是快路径里残留的每次调用成本。慢路径（`cuda.py:3347-3361`）只在首次或异常回退时走。

### 1.2 快路径逐项拆解（`cuda.py:3298-3346`）

| # | 代码位置 | 做什么 | **有 Python 分配吗** | µs/次 `[估算]` |
|---|---|---|---|---|
| 1 | `3299` `self._activate()` → `2408-2410` | 2 次属性查找 + 1 次方法调用 + **1 次 `cuCtxSetCurrent` 驱动调用**（`self.ctx` 在 init 时必被设上 `cuda.py:2330`，所以这条**每次都真调驱动**） | 无 | **0.5–1.0**（驱动 0.2–0.5 + 调用 0.3–0.5） |
| 2 | `3300` `if self._capturing and name == ...` | 2 次属性 + 字符串比较 | 无 | 0.03 |
| 3 | `3302` `self._fn(name)` → `3278-3286` | `dict.get` 命中；**miss 时**才 `name.encode("ascii")`（1 次分配） | miss 时 1 次 | **0.06**（命中） |
| 4 | `3303` `if self._capturing:` | 属性查找 | 无 | 0.02 |
| 5 | `3322-3323` `if self._lfast is None` | 属性查找 | 无 | 0.02 |
| 6 | `3327` `key = (name, tuple(t[1] for t in args))` | **建 1 个外层元组 + 1 个类型元组 + 1 个生成器帧**，对 N 个参数逐个取 `t[1]` | **✅ 3 个对象/次** | **0.7–1.3**（N≈8–20） |
| 7 | `3328-3331` `self._ltmpl.get(key)` | 对含类型对象的元组求 hash + 查表 | 无 | 0.15 |
| 8 | `3333-3334` `for i, item in enumerate(args): holders[i].value = item[0]` | **`enumerate` 对象 1 个**；每个参数一次 `ctypes` 的 `value` setter（C 层做 int/float 转换） | **✅ 1 个对象/次** | **0.8–1.6**（0.08–0.15/参数 × 8–12 参数） |
| 9 | `3341-3343` `self.cu.cuLaunchKernel(fn, grid[0]...block[2], shared, self._stream, ptrs, None)` | 2 次属性查找 + `_FuncPtr.__call__`：**建 1 个实参元组、按 `argtypes` 逐项转换 11 个参数、调 C** | **✅ 1–3 个对象/次（ctypes 内部）** | **2.5–5.0** |

**每把一个指针变参数：`_ptr()`（`cuda.py:3425-3427`）**

| 行 | 做什么 | 分配 | µs |
|---|---|---|---|
| `3426` | `self._mark_submitted(buf.ptr)` → `3422-3423` **空函数** | 无 | 0.12（纯浪费的方法调用） |
| `3426` | `buf.ptr`（`__slots__` 属性）+ `int()` | int > 2^30 时 **✅ 可能新建 int 对象** | 0.05–0.10 |
| `3427` | `return (int(buf.ptr), ctypes.c_uint64)` | **✅ 1 个二元组 + 每次重新取一次 `c_uint64`（类属性，不分配）** | 0.15 |

⇒ **每次 `_ptr()` ≈ 0.35–0.45 µs**；每次 `launch` 平均 **3.4 个指针**（`adamw` 4 个、`permute8_v4` 2 个 + 18 个 int、`bcast_add4_v4` 3 个…）⇒ **1.2–1.6 µs/次发射**。

### 1.3 合计

| 来源 | µs/次发射 | 1330 次/步 |
|---|---|---|
| `_activate()`（含 1 次 `cuCtxSetCurrent`） | 0.5–1.0 | 0.7–1.3 ms |
| `_fn()` 查表 | 0.06 | 0.08 ms |
| key 元组构造 + dict get | 0.85–1.45 | 1.1–1.9 ms |
| `.value` 写入循环 | 0.8–1.6 | 1.1–2.1 ms |
| `_ptr()` × 3.4 | 1.2–1.6 | 1.6–2.1 ms |
| `cuLaunchKernel` 的 ctypes 调用 | 2.5–5.0 | 3.3–6.7 ms |
| 其余（属性查找/分支/return） | 0.5 | 0.7 ms |
| **合计** | **~6.4–11.3** | **8.5–15 ms** |

与仓库实测 **10.3 µs** 吻合（`profile_step.py` 的包装本身还要 ~0.6 µs）。取中值 **9.7 µs** 计入后面所有账。

### 1.4 慢路径对照（`cuda.py:3347-3352`，只在首次/异常回退）

```python
holders = []                                     # 1 list
ptrs = (ctypes.c_void_p * len(args))()           # 1 个 ctypes 数组对象（N 个槽）
for i, (val, typ) in enumerate(args):
    obj = val if isinstance(val, ctypes._SimpleCData) else typ(val)   # N 个 ctypes 对象
    holders.append(obj)                                                # list 扩容
    ptrs[i] = ctypes.cast(ctypes.byref(obj), ctypes.c_void_p)           # 每个 2 个临时对象
```

**每次 3 + 3N 个 Python 对象**（N=8 时约 27 个），实测约 25–40 µs。所以 M8 的缓存确实砍掉了 60–75% 的发射成本 —— 这也解释了为什么 `PRODUCTION_READINESS` 里 348 次发射只需 3.58 ms，而更早的文档估的是 8–15 µs/次。

---

## 2. Q2：每个 Python 算子平均发几次 `launch`

以 `qwen.py:195-243` 的 `_layer` 为准，逐句数。**注意：`launch` 次数 ≠ GPU kernel 数**，cuBLAS 的 GEMM 不算在 `launch` 里。

### 2.1 前向（`_layer`，每层）

| qwen.py 行 | 代码 | launch/次 | 说明 |
|---|---|---|---|
| `205` | `self.ln1[i](x).reshape(...)` | **1** | `rmsnorm_fwd`（`cuda.py:5152-5161`，1 kernel）；`.reshape` = **视图，0 次**（`cuda.py:3871-3886` 只换 shape 元数据） |
| `206-208` | `h @ Wq.T` / `Wk.T` / `Wv.T` | **0** | 各 1 次 `cublasSgemm`（`gemm_nt`，`__init__.py:895-914`，`.T` 是惰性视图不拷贝） |
| `210-212` | `q + bias` ×3 | **3** | `add_broadcast` → `bcast_add4_v4`（`cuda.py:3509-3519`） |
| `213-215` | `reshape → permute(0,2,1,3) → reshape` ×3 | **3** | reshape 0 次；**`permute` 是真拷贝 1 次**（`cuda.py:3634-3667`，`permute8_v4`） |
| `219-220` | `_rope_half_apply` ×2 | **2** | `rope_half_ip`（`cuda.py:5276-5293`）；`FUSED_ROPE=0` 时会变成 10 次/kernel 量级 |
| `222-223` | `repeat_interleave(NH//KV, dim=1)` ×2 | **2** | `index_gather8`（`cuda.py:3694-3714`），每次 1 kernel |
| `227-229` | `F.scaled_dot_product_attention(..., lowmem=False)` | **3** | 默认走物化路径（`__init__.py:2690-2709`）：`scale` 1（`ewise_scale_v4`）+ `+mask` 1（`bcast_add4_v4`）+ `softmax` 1（`softmax_lastdim`）；另外 2 次 cuBLAS（scores 的 NT-GEMM、`p@v`） |
| `230` | `o.reshape(...).permute(...).reshape(...)` | **1** | `permute8_v4` |
| `231` | `(o @ Wo.T).reshape(...)` + `x + ...` | **1** | GEMM 1 次 cuBLAS；残差 `add` 1 次 `ewise_add_v4` |
| `233` | `self.ln2[i](x).reshape(...)` | **1** | `rmsnorm_fwd` |
| `234-235` | `h2 @ W1.T` / `W2.T` | **0** | 2 次 cuBLAS |
| `238-239` | `T.silu_mul(g, u)` | **1** | `silu_mul_fwd`（`cuda.py:5176-5186`） |
| `242` | `(act @ W3.T).reshape(...)` + `x + ...` | **1** | GEMM 1 次 cuBLAS；`add` 1 次 |
| | **前向合计** | **19** | 另有 **9 次 cuBLAS** 不计入 launch |

### 2.2 反向（同一层，按反向顺序）

| 反向节点 | launch/次 | 说明 |
|---|---|---|
| 残差 add ×2 的 `bw` | 0（透传） + `_acc` 的 add 记在下面 | mask/bias 之类 `requires_grad=False` 的分支 0 kernel |
| `w3 / w1 / w2 / out_proj` GEMM 的 `bw` | **0** | 各 2 次 cuBLAS（`gemm_tn` 算 dW、`sgemm` 算 dA），共 8 |
| `silu_mul_bwd` | **1** | `cuda.py:5211-5221` |
| `rmsnorm_bwd` ln1/ln2 | **2** | `cuda.py:5163-5174`；内含 `zeros(w.shape)` → 1 次 `cuMemsetD32Async`（不是 launch） |
| `permute` 的反向 ×4 | **4** | q/k/v/o 各 1（`__init__.py:1347-1349`） |
| `rope_half_bwd` ×2 | **2** | `cuda.py:5295-5308` |
| `repeat_interleave` 的反向 ×2 | **2** | `index_scatter_add8` ×2（另有 2 次 memset） |
| `p@v` 反向 | **2** | `swap_last2(p)`、`swap_last2(v)` 各 1 kernel（`__init__.py:890-892` → `cuda.py:3888-3904`）+ 2 次 cuBLAS |
| `softmax_lastdim_bwd` | **1** | `cuda.py:4096` |
| `scores` 的 `scale` 反向 | **1** | `ewise_scale_v4` |
| `_attn_scores` 反向 | **2** | dq/dk 各一次 `scale`；2 次 cuBLAS |
| `+bias` 的反向 ×3 | **3** | `rt.unbroadcast(g,[896])` → `sum_axis8`（`cuda.py:3584-3597` + `3550-3578`） |
| `_acc` 二次命中触发的 `add` | **5** | 见 §4：`h`（3 个 GEMM 汇合，2 次）、`h2`（w1+w2，1 次）、层输入 `x`（ln1/ln2/残差 3 路，2 次） |
| **反向合计** | **24** | 另有 **18 次 cuBLAS** |

### 2.3 合计与对账

```
每层 launch = 19（前向）+ 24（反向）= 43
整步        = 43 × 24 + 290（adamw）+ 7（emb_fwd/emb_bwd/final rmsnorm×2/ce_fwd_rows/ce_fwd_mean/ce_bwd）+ 1（tie 词嵌入的 _acc add）
            = 1032 + 290 + 7 + 1
            = 1330         ← 与实测 1330 逐一对上
```

其中 5 个名字的**每步预测值**与 `docs/GAP_VS_PYTORCH.md:227-233` 的实测完全一致，这是这套台账可信的直接证据：

| kernel | 我的预测 | 实测 |
|---|---|---|
| `bcast_add4_v4` | 4×24 = **96** | **96** ✅ |
| `sum_axis8` | 3×24 = **72** | **72** ✅ |
| `ewise_scale_v4` | 3×24 = **72** | **72** ✅ |
| `softmax_lastdim_bwd` | 1×24 = **24** | **24** ✅ |
| `adamw_update` | 290（§3） | **290** ✅ |

### 2.4 直接回答题面的四个问号

| 问题 | 答案 |
|---|---|
| matmul → cublas 算 1 次？ | **是 1 次 cuBLAS 调用**，但它**不进 1330、也不进 111.6 ms**（§0.2）。整步 648 次（前 216 + 反 432），含 `_attn_scores` 的 NT-GEMM 与 `p@v` 的 strided-batched |
| `reshape` 几次？ | **0**。`cuda.py:3871-3886` 返回 `view_of=buf` 的新 `GpuBuf`（只改元数据、共用 ptr），既不分配也不发射。反向的 `rt.reshape` 同样是视图 |
| `permute` 几次？ | **各 1 次真拷贝**：前向 4 次（q/k/v/o）、反向 4 次，共 **8 次/层 = 192 次/步**（`permute8_v4`，`cuda.py:3652-3661` 走 float4） |
| `repeat_interleave` 几次？ | 前向 **1 次/调用**（`index_gather8`）→ 2 次/层；反向 **1 次/调用**（`index_scatter_add8`）**+ 1 次 `zeros` 的 memset**（`cuda.py:3726`）→ 2 次/层 + 2 memset。合计 4 launch/层 + 2 memset/层 |
| `_acc` 会不会触发 `add`？ | **会，但不是每次都触发**。首次命中只挂指针（0 kernel），二次命中才 `rt.add` → 1 个 `ewise_add_v4` + 1 次分配。实测 5 次/层 = 120 次/步（+1 次 tie 词嵌入）= 121 个 add kernel |

### 2.5 每层 launch 的 Top 3 来源

| 排名 | 来源 | 次数/层 | 次数/步 | 备注 |
|---|---|---|---|---|
| 1 | **`permute`**（前 4 + 反 4） | **8** | **192** | 全是真拷贝；q/o 各 7.34 MB、k/v 各 1.05 MB（单层往返 33.6 MB） |
| 2 | **`add` 系**（残差 2 + `_acc` 5） | **7** | **168** | 其中 120 次是"梯度汇合"，可以用 `beta=1` 的 GEMM 直接累加消掉 |
| 3 | **bias 加 + 其反向归约**（3 + 3） | **6** | **144** | 前向 `bcast_add4_v4`、反向 `sum_axis8`（后者 GPU 14.5 ms，见 §5.5） |

（并列第 4：`rmsnorm` 4/层、`rope` 4/层、`gather+scatter` 4/层）

---

## 3. Q3：`adamw_update` 为什么 290 次，能不能并成一次

### 3.1 290 = 每个参数张量一个 kernel，且数字能精确对上

`AdamW.step`（`__init__.py:3997-4010`）对 `self.params` 逐项调 `_cuda().adamw(...)`，`adamw()`（`cuda.py:5407-5430`）每次发 1 个 `adamw_update`。所以**次数 = 有梯度的参数张量个数**。Qwen2.5-0.5B 逐项数：

| 来源 | 每层 | ×24 |
|---|---|---|
| `attn.q/k/v/out_proj.weight` | 4 | 96 |
| `ffn.w1/w2/w3.weight` | 3 | 72 |
| `ln1/ln2.weight` | 2 | 48 |
| `attn.q/k/v_proj.bias`（`QKV_BIAS=1`） | 3 | 72 |
| | **12** | **288** |
| `token_embedding.weight`（tie，`params()` 已去重，`qwen.py:161-170`） | | +1 |
| `final_norm.weight` | | +1 |
| **合计** | | **290** ✅ |

与实测 290 严丝合缝 ⇒ 确认"一参数一 kernel"。

### 3.2 能不能并成 multi-tensor：能，而且**逐位一致**（不改数值语义）

`adamw_update` 是纯逐元素算子（`p,g,m,v` 同位下标，无跨元素归约），所以把 N 个张量拼成"指针数组 + 长度数组"，用一个 kernel 扫过去，**每个元素的算式和运算顺序完全不变 ⇒ 结果逐位相同**（不涉及浮点重排）。

推荐形态（对现有内核改动最小）：

```cuda
// 前提：host 侧把 290 个 (p,g,m,v,n) 打包成 4 个 device 端 int64 指针数组 + 1 个 int 长度数组
// grid = (max_blocks_per_tensor, n_tensors)；blockIdx.y 选张量，blockIdx.x 切块
extern "C" __global__ void adamw_multi(
    const unsigned long long* Ps, const unsigned long long* Gs,
    const unsigned long long* Ms, const unsigned long long* Vs,
    const int* Ns, int ntensors,
    float b1, float omb1, float b2, float omb2, float lr, float wd_lr,
    float eps, float bc1, float bc2, int do_wd)
{
  int t = blockIdx.y; if (t >= ntensors) return;
  float* p = (float*)Ps[t]; const float* g = (const float*)Gs[t];
  float* m = (float*)Ms[t]; float* v = (float*)Vs[t];
  int n = Ns[t];
  for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
      /* 与现有 adamw_update 一字不差的算式 */
  }
}
```

要点：

- **指针数组只在优化器构造时建一次**（`__init__.py:3969-3979` 那里 `m/v` 已经分配好，ptr 稳定；`p` 也是常驻 Parameter）；**只有 `p.grad` 每步是新缓冲区**，所以 grad 那一条数组需要每步刷一次（290 次 `int(ptr)` + 填数组 ≈ 30–50 µs，可忽略）。
- 标量（`b1/omb1/.../bc1/bc2/do_wd`）继续每步重算，和现在一样。
- 空梯度参数（`p.grad is None`）照旧跳过：把它们排除在数组之外即可。
- `do_wd` 现在按"全局 `self.wd` 是否为 0"取一个 int，`adamw_multi` 保持同一语义。

### 3.3 预计收益

| 项 | 现在 | 改后 | 省 |
|---|---|---|---|
| 发射次数 | 290 | **1**（或按 SM 切成 2–4 个） | **289 × 9.7 µs ≈ 2.8 ms CPU** |
| GPU 时间 | **18.8 ms** `[实测]` | 7 × 494 MB / 900 GB/s ≈ **3.8 ms**（带宽下限），实测预期 **6–8 ms** | **11–13 ms GPU** |
| `_split_forget_ptr` 调用 | 870 次（每次 3 个：`p/m/v`，`cuda.py:5409-5411`） | 0（批量版不需要） | ~0.3 ms |
| **合计** | | | **≈ 14–16 ms/步** |

### 3.4 风险

- **中**：要写新 kernel + 在 host 侧维护 device 端指针数组（`rt.h2d` 一次，或直接建一个 device 侧数组用 `cuMemcpyHtoD` 刷）。核心算子改动，必须做 **290 参数逐位对拍**（`tests/test_adamw_parity.py` 已有对拍骨架，可扩成多张量版）。
- 兼容 **CUDA Graph**：`_collect_adamw`（`cuda.py:6119-6154`）现在按"捕获时每次 `adamw_update` 的参数快照"来改写偏差校正节点；批量化后只有 1 个节点、参数少 13 个，**这条改写逻辑要一起改**（否则开 graph 时 bc1/bc2 改不到）。这是唯一一个"别处会连带"的点，务必一起验证。
- 收益上限受 §0.3 限制：这是**唯一一条 CPU+GPU 双收益、且量级（14 ms）明显超过其它项**的改动。

---

## 4. Q4：`_acc()` 与 `grads` 字典

### 4.1 代码

```python
# __init__.py:1581-1589
def _acc(grads, node, g):
    if getattr(node, "device", "cpu") == "cuda":
        rt = _cuda()
        old = grads.get(id(node))
        grads[id(node)] = g if old is None else rt.add(old, g)   # ← 二次命中才发 1 个 kernel
        return
```

`_backward_cuda`（`__init__.py:1038-1061`）用 `grads = {id(node): g}`，消费完 `grads.pop(id(node))`。

### 4.2 每步次数

| 项 | 每层 | 每步 | 说明 |
|---|---|---|---|
| `_acc()` 调用次数 | **~60** `[估算]` | **~1400** | 每个 autograd 边一次：GEMM 2、bias 2×3、`_acc` 汇合 5、reshape 视图 ~11、permute 4、rope 2、gather/scatter 4、softmax/scale 3、silu 2… |
| 其中**触发 `rt.add`** 的 | **5** | **121** | `h`（3 GEMM 汇合 → 2 次）、`h2`（w1+w2 → 1 次）、`x`（ln1+ln2+残差 → 2 次）、tie 词嵌入（emb_bwd + lm_head → 1 次） |
| `grads` 字典写/删 | ~60 / ~60 | ~1400 / ~1400 | `dict` 操作本身很便宜（~0.06 µs） |
| 纯 Python 成本 | | **0.8–1.2 ms/步** | 1400 × ~0.6–0.8 µs（含 `getattr`、`_cuda()`、`id()`、dict get/set） |
| `rt.add` 的附加成本 | | **1.2–1.5 ms CPU + 2–3 ms GPU** | 每次 1 次分配（~2 µs）+ 1 次释放（~1.5 µs，**还带一次 `cuMemGetInfo`，见 §5.2**）+ 1 个 `ewise_add_v4` |

### 4.3 明显的可省项（按性价比）

1. **GEMM 用 `beta=1` 直接累加，把 5 次/层的 `_acc` add 消成 0**（最干净）
   - 现状：`h` 被 q/k/v 三个 GEMM 各写一份 `[2048,896]` 梯度，靠 `_acc` 加两次。
   - 改法：`rt.sgemm/gemm_nt` 里 `beta` 现在是硬编码 `0.0`（`cuda.py:4675`）；给 `bmm`/`sgemm` 加一个可选 `acc_into=<GpuBuf>`，第二、三个 GEMM 直接 `beta=1` 写进同一个缓冲区。
   - 收益：−120 个 kernel（−1.2 ms CPU、−2~3 ms GPU、−120 次分配/释放）。
   - 风险：**中**。要保证"谁拥有这块梯度"的唯一性（现在靠 `_acc` 的 dict 语义），否则会重复累加/漏累加。建议只对 `h`、`h2` 这两个**形状相同、消费顺序确定**的点先做，收益 90 个 kernel。

2. **`_acc` 用原地加**：`rt.add_(old, g)`（`cuda.py:5310-5329` 已存在 `fused_add_ip`）可省一次分配 + 一次释放。
   - 前提：`old` 的引用计数为 1（`sys.getrefcount(old) == 2`）。现在是 `grads` 字典唯一持有者，通常成立；不成立就退回 `add`。
   - 收益：−121 次分配/释放 ≈ **0.4 ms CPU**（含省下的 `cuMemGetInfo`）。风险低，但要小心别改动 `p.grad` 正在引用的缓冲（叶子节点的 grad 会被 `opt.step` 读，必须排除）。

3. **`_acc` 里删掉 `getattr(node, "device", "cpu")`**：`Tensor` 一定有 `.device`，`getattr` 带默认值是纯浪费（1400 次 × 0.08 µs ≈ 0.1 ms）。一行改，风险 0。

4. **不要动 `grads` 的 `id()` 语义**：它能工作是因为 `_parents` 一直拉着节点不让 `id` 复用；改成 `WeakKeyDictionary` 只会更慢更危险。

---

## 5. 最省力的 5 条削减（不改数值语义）

排序原则：先"改一处、全局受益"，再"改一处、只受益一次"。所有 ms 都按 §0.4 的预算读：**CPU 侧总量只有 40–90 ms**。

### 5.1 【R1】AdamW 批量化：290 → 1–4 次发射

| 项 | 内容 |
|---|---|
| 改哪里 | `cuda.py:5407-5430`（新增 `adamw_multi`）+ `__init__.py:3997-4010`（`step()` 里改成一次性调用）+ 内核源码（`_M5_KERNELS` 里加） |
| 省多少 | **CPU −2.8 ms**（289 × 9.7 µs）+ `_split_forget_ptr` −0.3 ms；**GPU −11~13 ms**（18.8 → 6~8） |
| 风险 | 中（新内核 + host 指针数组；`_collect_adamw` 的 graph 改写要同步改） |
| 验证 | 290 参数逐个与旧路径逐位对拍；开 `TG_CUDA_GRAPH` 时 bc1/bc2 改写正确 |

### 5.2 【R2】`free()` 路径上那次 `cuMemGetInfo`：每次归一块都问一遍驱动

| 项 | 内容 |
|---|---|
| 证据 | `cuda.py:2852` `_pool_release` 的**快路径**尾部无条件调 `_pool_shed_ready()`；后者在 `2877-2889` 先 `_pool_frozen()`、再 **`free_device_bytes()` → `cuMemGetInfo_v2`**（`cuda.py:3010-3021`），然后 `elif free_b >= (2<<30): return` |
| 每步次数 | 每个 `GpuBuf` 死亡一次（`cuda.py:1969-1984` `__del__` → `rt.free`）。整步分配 ≈ 43 launch + 27 GEMM 的输出 + 视图 ≈ **1700–1900 块**，加 `zero_grad` 丢掉的 290 个梯度 ⇒ **约 2000 次/步** |
| 省多少 | 每次 1.5–2.5 µs（驱动调用为主）⇒ **3–5 ms CPU** |
| 改法 | 给 `_pool_shed_ready` 加节流：① 只在 `_pool_cached_bytes` 增长超过阈值（如 256 MiB）或每 N 次（如 64）才查一次；② 或直接复用 `_alloc` 里已查到的 `free_b` 缓存值（加一个 `self._free_b_cache = (值, 计数器)`）。**逻辑不变**，只是少问驱动 |
| 风险 | **极低**。唯一的语义是"在显存快满时提前丢缓存"，节流 64 次仍然远早于 OOM；OOM 兜底路径（`_alloc` 里的 `cuCtxSynchronize` + 重试）没动 |
| 注意 | 这是**唯一一条"每次算子都经过、但完全没人注意"**的固定成本，属于题面说的"改一处、全局受益" |

### 5.3 【R3】cuBLAS 的 `argtypes` 每次调用都重建（648 次/步）

| 项 | 内容 |
|---|---|
| 证据 | `cuda.py:3955-3965`（`sgemm_strided_batched`）、`cuda.py:3994-4004`（`_bmm_op`）、`cuda.py:4313+`/`4611+`（f16/bf16）**每次调用**都给 `cb.cublasSgemmStridedBatched.argtypes` 赋一个 19 元素的 list。ctypes 的 `argtypes` setter 会逐项校验并存成数组 |
| 次数 | 前向 2/层（scores NT、p@v）+ 反向 2/层（dp、dv）= **96 次/步**（`_bmm_op` 路径），若计 `_sgemm` 的 2 个 `c_float` 分配则 648 次 |
| 省多少 | 96 × 2–3 µs ≈ **0.2–0.3 ms**；再加 `_sgemm`（`cuda.py:4674-4676`）每次 2 个 `c_float` 分配 + `_capture_keep(alpha, beta)` 调用，648 × 1.5 µs ⇒ **再 1.0–1.3 ms**。合计 **1.2–1.6 ms CPU** |
| 改法 | 把 `argtypes/restype` 移到 `_init_cublas`（`cuda.py:2444-2499`）一次设好；`alpha/beta` 提成模块级常量 `_ONE = ctypes.c_float(1.0)` / `_ZERO = ctypes.c_float(0.0)`（只读，可安全复用；`_capture_keep` 仍需保留，但可以只在 `_capturing` 时才调） |
| 风险 | 极低。纯宿主侧改动，不碰数值 |

### 5.4 【R4】`launch()` 快路径的两处零收益开销

| 项 | 内容 |
|---|---|
| 证据 A | `cuda.py:3327` `key = (name, tuple(t[1] for t in args))`：**每次发射**建 2 个元组 + 1 个生成器帧（0.7–1.3 µs）。这个 key 只由"kernel 名 + 参数类型序列"决定，**每个调用点都是常量** |
| 证据 B | `cuda.py:3426` `self._mark_submitted(buf.ptr)` → `3422-3423` **空函数**，3.4 次/发射（0.12 µs × 3.4） |
| 改法 | A：给每个算子包装层一个模块级 `_KEY = (name, types_tuple)` 常量，或让 `launch` 接受可选的预建 `key`/直接接受 `tmpl`；最省事的做法是**在 `_ltmpl` 里用 `id(args)` 之外的第二级缓存**：`self._ltmpl2[(name, len(args), type(args[0][1])...)]` —— 但更干净还是让包装层传 key。B：把 `_mark_submitted` 改成**直接删掉那行**（函数体是 `return`，且 `cuda.py` 里所有 `_mark_submitted` 调用点都可以替换成 `pass`；保留函数名以防外部引用） |
| 省多少 | (0.7–1.3 + 0.4) × 1330 ≈ **1.5–2.3 ms CPU** |
| 风险 | 低。A 只要保证"同一个 kernel 名的参数类型序列"不变（现在也是按这个 key 缓存的，语义完全相同）；B 删的是空函数调用 |
| 补充证据 | `_mark_submitted` 是**死代码**，`_submit_stamp` 也从不自增（只在 `cuda.py:2277` 初始化为 0、在 `2779` 被写进 `_pool_live`、在 `3231` 赋给 `_done_stamp`）⇒ 那个 stamp 字段同样是死的。删掉不会有任何行为变化 |
| 附带 | `_fn(name)` 的 dict 命中已经很快（0.06 µs），不用动；`_activate()` 的 `cuCtxSetCurrent` 是**唯一一次真驱动调用**，如果确认进程里只有一个 context，可以按"上次设置过的 ctx"缓存（`if self._ctx_cur is not ctx: ...`），再省 0.3–0.6 µs × 1330 ≈ **0.4–0.8 ms**。风险：多 context/多线程场景会失效 —— 本运行时是单 context 单线程，但要在注释里写清前提 |

### 5.5 【R5】bias 的"前向加 + 反向归约"：144 次发射 + 14.5 ms GPU

| 项 | 内容 |
|---|---|
| 现状 | 前向 `q + bias` 3 次/层 → `bcast_add4_v4`（96 次/步，GPU 4.9 ms `[实测]`）；反向 `unbroadcast` → `sum_axis8`（72 次/步，**GPU 14.5 ms** `[实测]`，占自研 kernel 总 GPU 时间的 13%） |
| 为什么 `sum_axis8` 这么慢 | `cuda.py:3574` `grid, block = self._ewise_grid(n_out)`，`n_out = 896` ⇒ **只有 4 个 block / 896 个线程**，每个线程串行扫 2048 行（7.3 MB），V100 有 5120 个 CUDA core，**并行度只有 17%**。14.5/72 = 0.2 ms/次，而按带宽只需 ~0.08 ms |
| 改法（两步，可分开做） | ① **反向换两段归约**：`sum_axis8_rows`（grid = 2048×896/256，每线程读 1 个元素、`atomicAdd` 到 `out[col]`，或先写 partial 再第二段）→ GPU 预计 **14.5 → 2–3 ms**；② **前向 bias 融进 GEMM 的 epilogue**（`beta` 通道 + 一个 `bias` 指针，或更简单：把 `q+bias` 融进紧随其后的 `permute`，`permute8_v4` 读源时加一个 `bias[col]`）→ 省 72 次发射（**−0.7 ms CPU**）+ 0.45 GB 流量 |
| 省多少 | **GPU −12~13 ms**、CPU −0.7 ms（只做①也有 −11 ms GPU、−0 ms CPU） |
| 风险 | ①**低**（只改归约顺序；注意 `atomicAdd` 的求和顺序会变 ⇒ 与 numpy 的"逐行左折叠"不再逐位一致，**但 bias 梯度本来就是多路累加，现有实现也是分块，不保证逐位**；若要严格逐位，用"partial 数组 + 固定顺序第二段"）。②**中**（动 GEMM/permute 内核）。建议先只做①，它是纯 GPU 收益、CPU 无关、风险最低的一条 |

### 5.6 【不在这 5 条里，但必须写】真正的杠杆在哪

按 §0.3，剩下 ~500–600 ms 是 cuBLAS fp32 GEMM。CPU 侧再怎么优化，天花板是 §0.4 的 40–90 ms。要动大头只有三条：

| 方向 | 依据 | 预期 | 代价 |
|---|---|---|---|
| **fp16/bf16 GEMM + split-K fp32 累加** | 仓库已有骨架：`TG_FP16=1`（`cuda.py:4658-4666`）、`sgemm_split16`（`cuda.py:4507`）、`_sgemm_f16`（`cuda.py:4313`）。`docs/PRODUCTION_READINESS.md:729-742` 已定位："短 K 无损但无收益；长 K（dW，k=4992）速度 1.20× 但梯度误差 1.97e-2" | GEMM 500→150–250 ms（V100 tensor core），**wall 744 → 350–450 ms** | 高（要解决长 K 精度，判据已写好：梯度相对误差 ≤1e-5、600 步 loss 偏 ≤2%） |
| **CUDA Graph**（把 1330 + 648 次调用压成 1 次） | 已有 `graph_begin/graph_end`（`cuda.py:6035+`），但捕获会因池未命中失败（`GAP_VS_PYTORCH.md:243-251`）；**且仓库实测 V100 只省 4%** | **−4%（≈30 ms）**，不是文档里写的"→150–200 ms"（那个预测假设 GPU 只忙 111.6 ms，见 §0） | 中（要先 `reserve_capture_arena` 预热池；`_collect_adamw` 的节点改写已在） |
| **减少非 GEMM 的搬运算**（`permute` 192 次/步、`swap_last2` 48 次、`index_*` 96 次） | `docs/analysis/audit_backward.md` §2-§4 已逐项算过字节 | 合计 ~5–8 GB/步 ⇒ **20–40 ms** | 中高（要改 attention 的布局/内核参数） |

`GAP_VS_PYTORCH.md:251` 的"wall 有望 744 → 150–200 ms"这个预测**建议撤回**：它把"自研 kernel 的 GPU 时间"当成了"全部 GPU 时间"，漏了 cuBLAS。按仓库自己的 Graph 实测（V100 −4%），现实预期是 **744 → ~700 ms**。

---

## 6. 附录：每步 kernel 台账（预测 vs 实测）

`B=4/S=512/L=24`，`profile_kernel.py` 那条路径（无梯度裁剪）。

| kernel | 每层 | 每步（预测） | 实测次数 | 实测 GPU ms | 出处 |
|---|---|---|---|---|---|
| `adamw_update` | — | 290 | **290** | **18.8** | §3 |
| `permute8_v4` | 8 | **192** | — | — | 前 4 + 反 4 |
| `ewise_add_v4` | 7 | **168** + 1 | — | — | 残差 2 + `_acc` 5（+tie） |
| `bcast_add4_v4` | 4 | **96** | **96** | 4.9 | 3 bias + 1 mask |
| `rmsnorm_fwd` + `rmsnorm_bwd` | 2+2 | 96 + 2 | — | — | 层内 2+2，末尾 final 各 1 |
| `rope_half_ip` + `rope_half_bwd` | 2+2 | 96 | — | — | `FUSED_ROPE=1` |
| `sum_axis8` | 3 | **72** | **72** | **14.5** | bias 反向 `unbroadcast` |
| `ewise_scale_v4` | 3 | **72** | **72** | 4.8 | scores 1 + dq/dk 2 |
| `index_gather8` | 2 | 48 | — | — | GQA k/v 展开 |
| `index_scatter_add8` | 2 | 48 | — | — | 反向（+2 memset/层） |
| `swap_last2` | 2 | 48 | — | — | `p@v` 反向 |
| `silu_mul_fwd` + `silu_mul_bwd` | 1+1 | 48 | — | — | `FUSED_OPS=1` |
| `softmax_lastdim` | 1 | 24 | — | — | 物化 SDPA |
| `softmax_lastdim_bwd` | 1 | 24 | **24** | **8.8** | 同上 |
| `ce_fwd_rows` + `ce_fwd_mean` + `ce_bwd` | — | 3 | — | — | 无 `ignore_index` 的CE |
| `emb_gather_fwd` + `emb_gather_bwd` | — | 2 | — | — | 词嵌入 |
| 非 launch 的 GPU 操作 | — | — | — | — | 648 cuBLAS GEMM + ~50 memset + 2 D2D |
| **合计** | **43** | **1330** | **1330** | **111.6（仅自研）** | ✅ |

---

## 7. 未取证边界（诚实清单）

1. **我没跑 GPU**。所有 µs 都是按代码结构 + ctypes 常识估的；唯一硬锚点是仓库自测的 **10.3 µs/次（V100）**。若实测是 6 µs 或 15 µs，§1/§5 的所有 CPU 数字要按比例缩放。
2. **§0.3 的 GEMM 时间是 FLOP 推算，不是实测**。它依赖"这张卡是 V100、cuBLAS 跑到 10–12 TFLOPS"两个假设。**§0.5 的三条验证里任意一条做完，就能把它变成实测**。在此之前，"85% 是 CPU"和"~70% 是 GEMM"两种说法都只是推断，但后者有 4 个独立旁证（§0.3），前者只有一个自洽性矛盾（475 µs/launch）。
3. **290 个参数是算术吻合**（24×12+2=290，与实测 290 一致），我没有逐个名字去 checkpoint 里点。若某层缺 bias（`QKV_BIAS=0`）总数会变 218，届时实测不会是 290 —— 现在实测正是 290，反证了 bias 存在。
4. **`_acc` 的 5 次/层是从 43 这个总数倒推 + 手工枚举消费关系得到的**（`h`/`h2`/`x`）。如果某个 `_acc` 因 `requires_grad=False` 被跳过（例如 mask），实际会是 4 次 × 24，总数就会差 24 —— 但总数 1330 与其它 4 个名字的实测同时对上，说明 5 是对的。
5. **`_pool_release` 的次数（~2000/步）是估算**，来自"分配数 ≈ launch 数 + GEMM 输出数 + 视图数"。要精确，可加一个计数器打印 `rt._pool_releases`（现在没有这个字段，`pool_stats()` 里有 hits/misses/trims 但没有 free 计数）——**这也顺手说明观测性缺口：R2 的收益在现有 `pool_stats` 里看不见**。
6. `test_adamw_parity.py` / `test_backward_parity.py` 我只看了文件名与入口，没有逐行确认它们能否直接复用来做 §5.1/§5.5 的对拍。

---

## 8. 一句话结论

**1330 次发射、每次 ~9.7 µs，合计 12.9 ms（占 wall 1.7%）；把这 1330 次全删掉也只省 1.7%。**
真正能拿的 CPU 侧收益是"发射之外"的三处：**每次 free 的 `cuMemGetInfo`（3–5 ms）、cuBLAS 宿主参数组装（1.2–1.6 ms）、`launch` 快路径残留（1.5–2.3 ms）**，加上两个**同时降 GPU** 的结构性改造：**批量 AdamW（−14 ms）、bias 反向归约（−12 ms）**。五条全做 ≈ **−35 ms（wall 744 → ~710 ms）**，全部低/中风险、不改数值语义。

**若目标是 744 → 300 ms，唯一的路是 fp16/split-K GEMM（§5.6），不是 CPU 侧。**
