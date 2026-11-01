# TGAI GO 训练引擎显存分析：为什么"比理论值多一倍"

> 只读分析，未修改仓库任何文件，未连接任何远程机器。
> 仓库：`D:\DSH-Workspace\TGAI-对话档案\_tgai_orv`
> 目标形状：Qwen2.5-0.5B（494,032,768 参数，fp32 权重 + AdamW），V100-32GB。
> 行号均以本仓库当前工作树为准（`__init__.py` 共 4133 行，`cuda.py` 共 6063 行，`qwen.py` 共 663 行）。

---

## 0. 结论先说（TL;DR）

**"多一倍"不是玄学，是两个确定的结构性重复，各占一半左右：**

| 组成 | B=4/S=512 实测/推算 | 是谁 |
| --- | --- | --- |
| 权重 1.84 + 叶子梯度 1.84 + AdamW m/v 3.68 | **7.36 GiB** | 理论值里唯一算对的部分 |
| 前向图保留的激活 | **11.26 GiB** | 理论估的 3.6 GiB **少了 3 倍**（每层 480 MB，不是 150 MB） |
| **反向时同时存活的"中间梯度"** | **11.23 GiB** | ★ 理论值里**完全没算**这一项 —— 这就是"多出来的那一倍" |
| 词表维度张量 `[N,V]`（4 份或 1 份） | 4.64 / 1.16 GiB | 第二档原因 |
| **合计** | **29.85 GiB（无词表张量）/ 34.5 GiB（4 份）** | 实测 29.92 GiB→OOM（与 29.85 相差 0.24%） |

> 静态台账在 B=4/S=512 上给出 **29.85 GiB**，实测 OOM 前的活张量峰值 **29.92 GiB** —— 差 0.24%。
> 同一套台账在 B=1/S=256 上给出 9.66 GiB，实测 8.84 GiB（+9%，因为部分节点是零拷贝视图）。

**三句话版本：**

1. **`backward()` 从不逐节点释放梯度**（`__init__.py:1026`、`1044` 只有写、没有 `del`/`pop`），
   整个反向过程中**每一层的每个中间张量的梯度都一直躺在显存里**，直到 `backward()` 返回。
   于是"激活"这份内存在反向阶段被**完完整整复制了第二份**。
2. **前向图本身就比教科书厚 3 倍**：RoPE 在 `qwen.py:179-183` 走组合实现（一个 RoPE = 6 个物化节点），
   `permute` / `__getitem__` / `repeat_interleave` / `cat` 在 CUDA 后端**全部是真拷贝**（PyTorch 全是视图），
   SDPA 物化 3 份 `[BH,S,S]`（每层 176 MB，占激活的 37%）。
3. **`checkpoint()` 在 CUDA 上是空操作**（`__init__.py:2526-2528` 提前 `return fn(*inputs)`），
   所以 `GRAD_CKPT=True` 实测"无差异"**不是形状太小，是这条路径根本没生效**；
   `docs/AMP_PLAN.md:190-192` 与 `_msgAY.txt:7-11` 的归因需要更正。

---

## 1. 方法与可复现性声明（先说清边界）

- 本机是 Windows，**没有可用的 CUDA 设备**（仓库自带的 `tgai_go.train.cuda` 需要 nvidia 驱动 + sm_70+），
  因此**本文没有跑任何 GPU 实测**。所有"实测"数字均引自仓库内的既有记录：
  `docs/AMP_PLAN.md:160-193`、`CHANGELOG.md:7-8`、`_msgAY.txt:4-5`、`tests/test_chunked_ce.py:4-5`。
- 本文的定量部分是**代码推导的静态台账**：按 `qwen.py::Qwen25._layer`（185-223 行）逐算子枚举
  图节点，按 `cuda.py` 里各 kernel 的分配行为判定"新缓冲 or 视图"，再乘 24 层。
  台账脚本是一次性算术（只在 `%TEMP%` 里跑，未落仓库）。
- **交叉验证**：同一套台账同时命中两个不同形状的实测值（见 §0 表），
  因此我认为它对"哪一项占多少"的判断是可信的；对**具体某一个数字**仍建议按 §7 的仪器化复测。
- 判定"是否物化"的依据（`cuda.py`）：
  - `reshape` = **视图**（3733-3748，`view_of=buf`，不分配）；
  - `permute` = **新缓冲**（3496-3497 注释"输出是新的连续缓冲"，3505 之后 `out_shape` + `empty`）；
  - `broadcast_to` = **新缓冲**（3484 `out = self.empty(shape)`）；
  - `index_gather`（`__getitem__` / `repeat_interleave` 共用）= **新缓冲**（3556-3567）；
  - `cat_place` + `empty` = **新缓冲**（3624-3641）；
  - `add` / `mul` / `scale` = **新缓冲**（3305、4976、4035）。
  ⇒ 对比 PyTorch：`x.permute(...)`、`x[:, :, h:]`、`x.transpose(...)` 返回的都是**共享 storage 的视图**
  （`torch/_tensor_docs.py:6776` "Returns a view of this tensor…"），**一个字节都不占**。

---

## 2. Q1：我们的 autograd 在前向时保存了什么？

### 2.1 机制：闭包硬引用，不是 `saved_tensors`

引擎没有 `ctx.save_for_backward` 这一层。每个算子在手写 `bw(g, grads)` 闭包时，
**直接在 Python 闭包里捕获它需要的张量**（强引用），并把闭包挂到输出节点上：

```python
# __init__.py:738-758  add 的反向
def bw(g, grads):
    if self.requires_grad:  _acc(grads, self,  src)
    if other.requires_grad: _acc(grads, other, src)
out._backward = bw          # 闭包捕获 self / other（父节点）→ 硬引用
```

同时 `Tensor.__init__` 还会把父节点放进 `_parents`（`__init__.py:582`，`__slots__` 见 570-571）：

```python
570: __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward", "device", "_gpu",
571:              "_lazy_T", "_lazy_shape")
582: self._parents = _parents
583: self._backward = _backward
```

**关键差别**：闭包捕获的是"张量对象"而不是"保存的副本"，所以
**"保存"本身不额外花显存 —— 它把上游激活的寿命强行延长到反向结束**。
这和 PyTorch 的语义一致（PyTorch 的 `SavedVariable` 也是共享 storage），
**真正的问题不在"保存了什么"，而在"保存多久"（见 Q2）**。

### 2.2 逐算子：保存了什么、多大

以 B=4/S=512（N=B·S=2048 token，BH=B·NH=56 头）为例，单层单份大小：

| 算子 | 代码 | 前向保存（供反向） | 单层额外占用 | 备注 |
| --- | --- | --- | --- | --- |
| `matmul` / `Linear` | `__init__.py:922-933`（NN）；`894-907`（`W.T` 走 `gemm_nt`） | 两个输入 `a,b`（闭包 881-885 / 900-906） | **0** | a、b 本来就是图节点/参数，已在账上 |
| `rms_norm` | `3214-3236` | 只存 `x`、`w` 的引用，反向**重算** rsqrt（`rt.rmsnorm_bwd(g,x,w,…)` 3229） | **0** | 这点是对的，和 PyTorch `native_rms_norm` 同思路 |
| `silu` | `2681-2694` | 输入 `x`（`y` 是 `clone` 后原地改写，见 2684-2685） | **0** | 需要 x 才能算 `d = σ(1+x(1-σ))` |
| `silu_mul`（SwiGLU） | `2979-3000` | `a`、`b` 两个输入（2993 `silu_mul_bwd(g,a,b)`） | **0** | 融合得不错，1 个 kernel |
| `softmax` | `2664-2678` | **输出 `y`（概率矩阵 P）**（2676 `softmax_lastdim_bwd(g, y)`） | `[BH,S,S]` = **58.7 MB** | P 同时是下一步输入，勉强算必经 |
| **SDPA（默认，物化）** | `2599-2612` | **scores、masked scores、P 三份 `[BH,S,S]` 同时活** | **176 MB** | 见下方逐行 |
| SDPA（`lowmem`，未启用） | `2629-2661` | 只存 `o`、`m`、`l`（2642 `attn_nop_fwd_stats` 返回三元组） | 7.34 MB + 0.11 MB | **默认 `lowmem=False`（2421、2436），这条好路径没走** |
| `cross_entropy`（CUDA，无 ignore） | `2714-2750` | **整块 `logits [N,V]`**（闭包 2735 `gpu = logits._gpu`）；`ce_bwd` 在"自有大张量"时**原地**写回 | 1.16 GiB（全局一份） | `owned` 判定 2738-2744 |
| `cross_entropy`（含 `-100`） | `2776-2826` | `logits` + `lc_buf`(gather 副本) + `dlc` + `d` | **3×1.16 GiB** | 2809 / 2819 / 2820 三份词表大小 |
| `checkpoint` | `2509-2559` | 设计上"不保存"，但 CUDA 上早退（2526-2528） | — | **等于没接** |
| RoPE（`_rope_apply`） | `qwen.py:179-183` | 6 个中间节点：`x[:,:,h:]` / `-` / `cat` / `x*cos` / `rot*sin` / `+` | **45.0 MB**（q 39.8 + k 5.2） | 融合版 `fused_rope`（3068）**没被 qwen.py 调用** |
| `permute` | `qwen.py:203-205, 215`；`__init__.py:1322-1335` | 全部真拷贝 | **16.8 MB**（q 7.34 + k/v 1.05×2 + o 7.34） | PyTorch 是 0 |
| `repeat_interleave`（GQA） | `qwen.py:212-213`；`__init__.py:1455-1467` | 真拷贝 | **14.7 MB** | PyTorch 可用 `expand` 零成本 |
| `q+bias` / `k+bias` / `v+bias` | `qwen.py:199-202` | 每个都是独立节点 | 9.4 MB | PyTorch 的 `addmm` 只出 1 个张量 |
| causal mask | `2607-2610` | 每层每步重建 `np.triu(S,S)` 再 H2D，被 add 的闭包持有 | 1.05 MB（×24 层 = 25 MB）+ 每步 24 次 H2D | 应缓存 |
| `reshape` | `1255-1264` + `cuda.py:3733-3748` | **视图，0 字节** | 0 | 但视图 `_base≠None` → 让 CE 判 `owned=False`（见 Q5-#4） |

**逐层合计：480.2 MB**（明细见附录 A），×24 层 = **11.26 GiB**。
其中注意力三个 `[BH,S,S]` 占 **4.13 GiB（36.7%）**，RoPE 组合实现占 **1.08 GiB**。

### 2.3 SDPA 物化路径逐行（这是单层最大的一块）

```python
2599: def _sdpa_materialized(q, k, v, causal, scale):
2604:         scores = _attn_scores(q, k, scale)        # ① [BH,S,S] 58.7 MB（新缓冲）
2607:     if causal:
2608:         S = scores.shape[-1]
2609:         mask = np.triu(np.full((S, S), -1e30, dtype=np.float32), k=1)   # 每层每步重建
2610:         scores = scores + Tensor(mask)            # ② 又一份 [BH,S,S]（+ H2D 1.05 MB）
2611:     p = _Functional.softmax(scores, axis=-1)      # ③ 再一份 [BH,S,S]
2612:     return p @ v
```

① + ② + ③ 三份同时活（②③ 都是前者的子节点，反向都要用），
**每层 176 MB → 24 层 4.13 GiB**。而引擎里已经有 `_sdpa_lowmem`（2629-2661，不物化 P，
反向只读 `m`/`l`，每层只要 7.45 MB），但：

```python
2421:                                      scale=None, lowmem=False):
2434:         if lowmem and _sdpa_lowmem_ok(q, k, v, causal, dh):
2436:         return _sdpa_materialized(q, k, v, causal, scale)
```

`lowmem` 默认 `False`，`qwen.py:214` 也没传 → **这条省 4 GiB 的路径从来没被走过**
（`_sdpa_lowmem_ok` 要求 `dh==64`，Qwen2.5-0.5B 正好是 64，条件完全满足，见 2615-2626）。

---

## 3. Q2：`backward()` 结束后，`grads` 与中间张量是否及时释放？

### 3.1 答案：**`backward()` 内部不释放，返回后才释放；过程中峰值就是活张量的两倍。**

证据（CUDA 路径，`_backward_cuda`，`__init__.py:1000-1045`）：

```python
1013: topo, seen, stack = [], set(), [(self, False)]
1014: while stack:                       # ← 拓扑排序，topo 里【强引用】所有节点
1018:             topo.append(node)
1026: grads = {id(self): g0}             # ← 唯一的梯度存储
1027: for node in reversed(topo):
1028:     g = grads.get(id(node))
1029:     if g is None:
1030:         continue
...
1036:     if node._backward is None:      # 叶子：把 g 转成 .grad（1038 / 1042）
...
1043:     if node._backward is not None and _GRAD_ENABLED[0]:
1044:         node._backward(g, grads)    # ← 只往 grads 里【写】，从不删
1045: rt.sync()
```

三点事实：

1. **`grads` 没有出栈/释放**：全文件搜不到 `del grads[...]`、`grads.pop(...)`、`grads.clear()`
   （`_acc` 只在 1565-1573 写；`_backward_cuda` 只在 1026/1028 读写）。
   ⇒ 一个节点的梯度被它的消费者用完之后**仍然留在 dict 里**，直到整个 `backward()` 返回。
2. **`topo` 也强引用所有节点**（1013-1025），所以前向激活在整个反向过程中**一个都不能死**。
3. 因此反向峰值 ≈ `激活总量 + 中间梯度总量`，而这两者**尺寸几乎相同**（每个节点一份同形状梯度）。

CPU 路径**更糟**（`backward`，660-690）：

```python
686:             if node.grad is None:
687:                 node.grad = np.zeros_like(node.data)     # 每个中间节点都挂一份 .grad
688:             node.grad = node.grad + g                    # 再分配一份
```

CPU 上不但每个中间节点保留 `.grad`，`+` 还会新分配。CUDA 路径至少只给叶子挂 `.grad`（1036-1042），
这点是对的 —— **但中间的 `grads` dict 没管**。

### 3.2 有没有引用环？有没有全局缓存导致释放不了？

- **引用环：没有。** 边是单向的（子 → 父：`_parents` + 闭包），父不引用子。
  `tensor.permute` 的惰性转置 `_lazy_T`（`990-992`）也是子引用父。所以 loss/输出句柄一掉，
  CPython 立即按引用计数回收整张图，不需要 `gc.collect()`。
  （`tests/test_checkpoint.py:103-122` 用 `gc.get_objects()` 数 Tensor 也是这个前提。）
- **但"及时"二字不成立**，有两个独立的"跨边界滞留"：

  **(a) 反向过程中：`grads` 滞留（即上面的 11.2 GiB）。**

  **(b) 跨训练步：上一步的整张图会活到下一步前向结束之后。**
  `qwen.py::sft` 的内层循环：

  ```python
  400:         idxs = plan[step * batch:(step + 1) * batch]
  401:         for k, si in enumerate(idxs):
  402:             xb, yb = _make_xy(seqs[si], seq_len)
  403:             if cfg.get("CHUNK_CE"):
  405:                 zc = model.forward(xb, hidden=True)     # ← 新前向先跑完
  406:                 loss = chunked_ce(...)                  # ← 旧 loss 到这里才被覆盖释放
  ...
  411:             (loss * (1.0 / batch)).backward()
  ```

  `loss` / `zc` / `logits` 都是循环外变量；`zc = model.forward(...)` 执行时，
  **上一步的 `loss` 还活着**，于是"旧图 11.26 GiB + 新图正在长到 11.26 GiB"重叠。
  即使 (a) 修好，(b) 仍然会给出 2× 激活。修法很便宜（见 Q5-#3）。

- **全局缓存**：`_CE_SEL_CACHE`（2753、2756-2773）与 `_ROPE_P_CACHE`（3007、3032-3044）**都在，但都是死代码**：
  `_ce_selection` 全仓库没有任何调用点（只有定义在 2756），`_rope_perm` 只被 `_rope_compose`
  （3053）调用而 qwen.py 不用它。**所以当前没有"全局缓存吃掉显存"的问题** —— 唯一的常驻是
  分配器池（见 Q3）与 `Qwen25._rope` 里缓存的 cos/sin（`qwen.py:168-176`，约 0.25 MB，可忽略）。

- **另一个坑（不是显存翻倍原因，但会咬人）**：`Tensor.detach()` 在 CUDA 上会
  `Tensor(self.to_numpy().copy())`（`695-699`）—— **D2H 到主机再造一个 CPU 张量**。
  PyTorch 的 `detach()` 是零成本视图。目前训练循环没调用它，但任何"为了省显存先 detach 一下"的
  直觉写法在这是**反向优化**。

---

## 4. Q3：分配器为什么不把空闲块还给驱动？

### 4.1 设计意图：分层缓存（仿 PyTorch caching allocator）

- 申请：`_alloc`（`cuda.py:2941-2994`）→ 先按 `_pool_bucket`（2125-2140，向上取整到对齐桶，
  浪费 < 1/16）查 `_pool_free`，命中就直接交接指针，**一次 CUDA API 都不调**（2976 注释）。
- 归还：`GpuBuf.__del__`（1914-1929）→ `rt.free(ptr, token)`（2636-2653）→ `_pool_release`（2758-2803）：
  **只把指针塞回 `_pool_free`，不调用 `cuMemFree`**。

```python
2775: if self._pool_cached_bytes + int(size) > self._pool_cap:
2776:     self._pool_trim_to(self._pool_cap)         # 唯一的"正常回收"入口
2783:     chain = self._pool_free.get(size)          # 否则：进空闲链
2788:     self._pool_cached_bytes += int(size)
```

### 4.2 真正会 `cuMemFree` 的**全部**条件（共 5 条）

| # | 触发点 | 代码 | 条件 | 会不会在训练中发生 |
| --- | --- | --- | --- | --- |
| 1 | `_pool_release` → `_pool_trim_to(cap)` | 2775-2776、2864-2903 | `cached + size > 12 GiB` | 会，但只压到 12 GiB |
| 2 | `_pool_shed_ready`（每次 release 后调用） | 2809-2862 | `live + cached > 12.2 GiB`，且每尺寸只留 2 块、要 `cuCtxSynchronize` | 会（**会打断流水**，2843-2846 整上下文同步） |
| 3 | `_alloc` 里 OOM 兜底 | 2980-2984 | `cuMemAlloc` 返回非 0 且池非空 | 只在已经要炸时 |
| 4 | `empty_cache()`（显式） | 2655-2681 | 人工调用 | **库内没有任何调用点**（全仓库只有 `tools/profile_step.py:214`） |
| 5 | `_pool_shrink_idle` / `_pool_reclaim_idle` | 2908-2928、2930-2939 | 空闲时压到 256 MiB | **从未被调用（死代码）** |

常量（`cuda.py:2144-2152`）：

```python
2144: _POOL_CAP_BYTES   = 12 * (1024 ** 3)      # 空闲块上限 12 GiB
2148: _POOL_IDLE_KEEP   = 256 * 1024 * 1024     # 空闲只留 256 MiB
2149: _POOL_IDLE_TRIGGER = 2 * 1024 * 1024 * 1024
2152: _POOL_OCC_LIMIT   = (16 * 1024 + 256) * 1024 * 1024
```

### 4.3 所以 4.22 GiB 常驻是"按设计"的结果，不是 bug

- 上限是 **12 GiB** 而不是 PyTorch 那种"能用就留着"；实测 4.22 GiB 远低于 12 GiB，**永远不会被裁剪**。
- `_pool_reclaim_idle` 的 docstring（2931）说"空闲堆了超过 2 GiB 才回收"，但
  **代码比较的是 `_POOL_IDLE_KEEP`（256 MiB）且这个函数根本没人调用**（2932）；
  `_POOL_IDLE_TRIGGER`（2149）**定义了却没有任何引用** —— 典型的"设计写了、线没接"。
- 后果：`_pool_cached_bytes` 4.22 GiB 长期占着，使 `nvidia-smi` 看到的占用比"活张量"高 4 GiB，
  也让下一次大分配更容易 OOM（`_raise_alloc_failed` 的报错文案 3005-3006 已经把这件事印出来了）。

**接口现状**：`empty_cache()` 存在且实现正确（2655-2681：同步 → 逐桶 `cuMemFree` → 清零计数），
`pool_stats()`（2683-2704）也把 `live_bytes` / `cached_bytes` / `peak_bytes` / **`peak_live_bytes`** 都吐出来了
—— 观测接口是齐的，**只是没人调用 `empty_cache`**。

---

## 5. Q4：和 PyTorch 对比，逐条说明差异

> PyTorch 侧引用的是本机安装的 2.13.0+cpu 里**随包发布的源码/头文件**（C++ 引擎只有头文件，
> 无 .cpp；涉及引擎内部行为的结论我标注为"语义/头文件证据"，不冒充逐行源码）。

| # | 维度 | PyTorch | 我们 | 差异后果 |
| --- | --- | --- | --- | --- |
| 1 | **保存什么** | `FunctionCtx.save_for_backward(...)`，每个算子只存数学上必需的（`MmBackward` 存两个输入、`SoftmaxBackward` 存输出、`NativeLayerNormBackward` 只存 `mean`/`rstd` 这种 `[N,1]` 小量） | 闭包直接捕获父张量（`__init__.py:582-583`），**语义等价** | **不是差异点**。两边都靠"共享 storage + 延长寿命" |
| 2 | **保存多久** | 引擎按 `dependencies_` 引用计数调度：`graph_task.h:31-32` 的 `not_ready_`(Node→InputBuffer，**待消费的梯度**) + `dependencies_`(Node→计数)；某节点的梯度被**最后一个**消费者取走后立刻从 `not_ready_` 移除 → 内存归还。`release_variables()`（`node.h:561`）在 apply 后释放 saved tensors | `grads` dict 只增不减（`1026/1044`），`topo` 强引用所有节点（`1013-1025`） | ★**这是"多一倍"的主因**：我们把"激活"在反向阶段又复制了一份梯度 |
| 3 | **图的边** | `Node` 只持 `edge_list next_edges_`（`node.h:682`）= (指向下一 Node 的指针 + 输入序号)，**边不带张量数据** | `_parents` + 闭包都持有**张量对象** | 单看不多花字节（共享），但让"提前释放"失去可能 |
| 4 | **视图零成本** | `permute/transpose/T/slice/expand` 返回共享 storage 的视图（`_tensor_docs.py:6776`） | `permute`(3496)、`__getitem__`(3556)、`repeat_interleave`(3556)、`cat`(3624)、`broadcast_to`(3484) **全部真拷贝** | B=4/S=512 上每层多 31.5 MB（permute 16.8 + repeat 14.7），×24 = **0.74 GiB**；反向再一份 → 1.5 GiB |
| 5 | **注意力** | `F.scaled_dot_product_attention` 默认会挑 Flash / Memory-Efficient 后端（`nn/functional.py:6353-6433` 列出后端），**不物化 `[S,S]` 概率矩阵**；反向只保留 `out` + `logsumexp` | 默认走 `_sdpa_materialized`（`2421/2436`），三份 `[BH,S,S]`；好路径 `_sdpa_lowmem`（2629）条件完全满足却**没启用** | **4.13 GiB**（前向）+ 反向同尺寸梯度 → 约 **7 GiB 的差距** |
| 6 | **梯度检查点** | `torch.utils.checkpoint`：`ctx.save_for_backward(*tensor_inputs)` + `with torch.no_grad(): run_function(...)`（`checkpoint.py:271-275`），**CUDA 上正常工作**；非重入版还支持 `early_stop`（`1558+`） | `checkpoint()` 在 `2526-2528` 用 `x.device == "cpu"` 过滤，**CUDA 直接 `return fn(*inputs)`**；`res = Tensor(np.array(out.data))`（2545）在 CUDA 上 `out.data is None`，本来也造不出来 | 我们声称有、实际没有。`GRAD_CKPT=True` 在 GPU 上**永远是 no-op** |
| 7 | **`no_grad` 语义** | 输出 `requires_grad=False`，**根本不建 `grad_fn`** → 推理/评估零图开销 | 只把 `_GRAD_ENABLED[0]=False`（`497-507`），`requires_grad` 照旧、`_parents`/`_backward` 照建（`582-583`、`689/1043` 只是不"调用"） | `_probe_loss`（`qwen.py:346-352`）每次探针都会**白建一整张图和 11 GiB 激活**（用完即弃，但峰值会撞车） |
| 8 | **`empty_cache`** | `torch.cuda.empty_cache()`（`memory.py:216-228`）把缓存段还给驱动，不影响活张量；另有 `PYTORCH_CUDA_ALLOC_CONF` 的 `garbage_collection_threshold` 在主分配前按阈值主动回收 | `empty_cache()` 有（2655），但**没有调用点**，也没有"占用高时自动回收"的策略；`_pool_reclaim_idle` 是死代码 | 4.22 GiB 常驻 |
| 9 | **`expandable_segments`** | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 用 `cuMemMap`/VMM 把段做成可增长的一整块虚拟地址（`c10/core/AllocatorConfig.h:296、320-353`；`CachingDeviceAllocator.h:127` 的 `SEGMENT_MAP // a call to cuMemMap (used with expandable_segments)`），**消掉"碎片导致的 OOM"** | 自研池按**精确 bucket** 复用（2125-2140），没有段/VMM 概念；碎片不是我们的主因（6% 以内），但"池按尺寸分桶 + 12 GiB 上限"让大块无法合并 | 影响较小，属于"以后要不要上 VMM"的议题，**不是当前的 2×** |
| 10 | **为什么 PyTorch 不会多一倍** | ① 梯度按引用计数逐节点释放（#2）；② 视图不占内存（#4）；③ 注意力不物化 `[S,S]`（#5）；④ LayerNorm/RMSNorm 只存 `[N,1]` 统计量；⑤ 融合算子少（`addmm`、`silu`）；⑥ `checkpoint` 真能用 | 六条**全部不成立或部分不成立** | ①贡献 ~11 GiB，③贡献 ~7 GiB，②④⑤贡献 ~2–3 GiB |

**一句话**：PyTorch 不是"保存得更少"，而是**"用完就还"**（#2）**+"能不占就不占"**（#4#5）。
我们的实现是"先全留着、最后一起还"，于是最大值必然 ≈ 2 × 激活。

---

## 6. Q5：修复建议（按预期收益排序）

> 全部以 **B=4/S=512（N=2048）** 为基准，收益按 §1 的台账估算；
> 验证统一用 `rt.pool_stats()["peak_live_bytes"]`（`cuda.py:2697`）与 `peak_bytes`（2696）。
> 复测命令模板（不改仓库，用环境变量开关做 A/B）：
> `python examples/qwen25_sft.py --device cuda --batch 4 --maxlen 512 ...` 并在 step 结束后打印
> `T._cuda().pool_stats()`；**同一个形状跑两次（开/关）才算数**。

### #1 让 `grads` 逐节点释放（引用计数）—— 预期 **−9.4 GiB**

- **改哪里**：`python/tgai_go/train/__init__.py`
  - `_backward_cuda`（1000-1045）：建 `topo` 时顺便统计每个节点的"待消费父边数"
    `pending[id(p)] += 1`；在 `1044` 调完 `node._backward(g, grads)` 之后，
    对 `node._parents` 逐个 `pending[id(p)] -= 1`，减到 0 且 `p._backward is not None`
    （即非叶子）就 `grads.pop(id(p), None)`；**叶子不要 pop**（它的 `g` 已经交给 `.grad`，1038/1042）。
  - CPU 的 `backward`（660-690）同改，并顺手把 `node.grad = np.zeros_like(...)`（687）
    改成只在叶子建 `.grad`（和 CUDA 路径对齐）。
- **为什么是 9.4 而不是 11.2**：叶子梯度 1.84 GiB 本来就要常驻（`.grad`），中间梯度 11.23 − 1.84 ≈ 9.39。
- **收益**：反向峰值 `29.85 → 20.46 GiB`（无词表张量口径）；与 #2 叠加后 `≈ 8.5 GiB`（含 1 份词表约 9.7 GiB）。
- **验证**：`peak_live_bytes` 应从 ≈29.9 GiB 掉到 ≈20.5 GiB；
  再断言"任一步 `len(grads)` 不超过图深度量级"（可临时加一行 `print(len(grads))`，
  改完删掉；或直接在 `tools/` 下加一个只读探针脚本，不动库）。
- **风险**：低。唯一要小心的是"同一节点被两个消费者用"（fork/join）时**最后一次**才能删 ——
  用 `pending` 计数天然正确，不要用"用完即删"的朴素写法。

### #2 把 `checkpoint()` 的 CUDA 分支补上，并逐层包 —— 预期 **−11.1 GiB**

- **改哪里**：`__init__.py::checkpoint`（2509-2559）
  1. 删掉 `2526` 的 `and getattr(x, "device", "cpu") == "cpu"`，改成两边都支持；
  2. `2545` 的 `res = Tensor(np.array(out.data, copy=True))` 在 CUDA 上必须换成
     `res = Tensor._from_gpu(rt.clone(out._gpu), requires_grad=...)`；同时**显式丢掉 `out` 的图**
     （让 `_parents`/`_backward` 引用随 `out` 一起消失），否则"重算"等于没省；
  3. `2548-2556` 的 `bw` 现在依赖"重跑 `fn` 会把梯度累加到 `inputs.grad`"（`re.backward(g)`），
     CUDA 上应改成把 `g` 传进去、并复用主 `backward` 的 `grads` 机制，避免嵌套 `backward` 再开一个 dict。
- **配合**：`qwen.py:245-248` 已经有 `if self.use_ckpt: x = T.checkpoint(self._layer, x, i)`，
  `cfg["GRAD_CKPT"]=True` 即可；`_layer` 已经是纯函数式（`qwen.py:185-223`），**无需改模型**。
- **收益**：激活 11.26 → 每层边界 `x`（7.34 MB × 24 = 0.17 GiB），**−11.08 GiB**；代价约 +1 次前向（≈ +30% 时间）。
- **验证**：`peak_live_bytes` + **梯度逐位比对**（沿用 `tests/test_qwen_grad_ckpt.py` 的思路，
  但必须加 CUDA 用例 —— 现有用例被 `_cpu` fixture 固定在 CPU，见 `tests/test_checkpoint.py:18-21`，
  这正是这个 bug 漏网的原因）。
- **风险**：中。#2 与 #1 叠加后峰值 ≈ 8.5 GiB（含 1 份词表 ≈ 9.7 GiB）；此时 **#1 反而变成"次要项"**，
  如果只做一条，先做 #2（收益更大且不涉及反向引擎的引用计数逻辑）。

### #3 跨步释放旧图（在下一步前向之前）—— 预期 **−11.3 GiB**（形状相关）

- **改哪里**：`python/tgai_go/train/qwen.py::sft`（386-418）
  - `403-411` 的内层循环里，在**下一次** `model.forward(...)` 之前把上一步的图清干净：
    ```python
    (loss * (1.0 / batch)).backward()
    train_loss = float(loss.to_numpy())
    loss = None; zc = None; logits = None     # ← 关键：让旧图在下一步前向之前死
    ```
  - 更彻底：给 `Tensor` 加一个 `free_graph()`（把 `_parents=()`、`_backward=None` 沿链清掉），
    在 `opt.step()` 后调用。这也顺带解决"探针/评估与训练图重叠"（`qwen.py:382-385`、`422-425`）。
- **收益**：消除"旧图 + 新图"重叠，B=4/S=512 单图 11.26 GiB → **−11.26 GiB**；
  与 #1 叠加时这两项**不是同一块内存**（#1 管反向期，#3 管跨步），所以可以同时拿。
- **验证**：在 forward 结束处打点（把 `pool_stats()` 的 `live_bytes` 按阶段打印），
  确认"新前向结束时 live ≈ 单图"，而不是 ≈ 双图。
- **风险**：低。注意别把 `loss` 提前清掉导致后面 `train_loss` 取不到值（先取 float 再清）。

### #4 词表张量真正分块（并修 `ignore_index` 路径的 3 份副本）—— 预期 **−3.5 GiB**

- **事实核对（重要）**：`CHUNK_CE` 的收益**目前没有代码支撑**，理由有三：
  1. `chunked_ce`（`qwen.py:615-642`）里 `lg = T.checkpoint(lambda hh: hh @ W.T, hc)`（638）
     依赖 `checkpoint`，而它在 CUDA 上是 no-op（#2）；
  2. `chunk=int(cfg.get("CHUNK_TOKENS") or 8192)`（407）→ 默认 **8192 > N=2048，只有 1 块**；
  3. `sft` 里每步 forward 的形状决定一切，**分块只影响 CE 段**，不可能单独贡献 12.2 GiB。
  ⇒ `CHANGELOG.md:7-8` / `docs/AMP_PLAN.md:180-189` 的"−12.2 GiB 归因于分块 CE"**需要复测才能成立**
  （详见 §7）。
- **改哪里**：
  - `qwen.py:407` 的默认 `CHUNK_TOKENS` 从 8192 降到 **512~1024**；
  - `__init__.py::_cross_entropy_cuda_ignore`（2776-2826）三份 `[nvalid,V]` 收敛：
    `lc_buf`（2809 gather 副本）→ 直接在 `logits` 上做 `ce_fwd` 的 gather 版；
    `dlc`（2819）→ 用 `out=` 原地写（`ce_bwd` 已支持 `out=` 参数，见 2740）；
    `d`（2820 `emb_bwd` 又分配一整份 `[n,V]`）→ 让它直接散射进 `logits.grad` 的目标缓冲。
  - `_cross_entropy_cuda`（2738-2744）的 `owned` 判定被 `reshape` 视图破坏
    （视图 `_base≠None` → `owned=False` → 多分配一份 `dl`）：改成"只要不是共享上游激活的视图"就允许原地，
    或让 `chunked_ce` 直接把**二维 `lg`** 交给 CE（它现在已经是这样，所以分块路径反而更快 —— 这也是
    "分块实测变快/变小"的一部分真实原因）。
- **收益**：4 份 → 1 份，**−3.48 GiB**。
- **验证**：`peak_live_bytes` 差值应 ≈ 3.5 GiB；另外用 `tests/test_chunked_ce.py` 的
  数值等价断言（现有用例只在 CPU，**必须补 CUDA 用例**）。

### #5 把每层冗余物化节点砍掉（融合算子接线 + 视图语义）—— 预期 **−4.8 GiB**

按"前向省 1 份 → 反向也省 1 份"计（2×）：

| 改动 | 文件:行 | 单层省 | ×24×2 |
| --- | --- | --- | --- |
| `qwen.py:209-210` 的 `_rope_apply` 换成 `T.fused_rope(q, cos_t, sin_t)`（`__init__.py:3068`，已实现、已有回退） | `qwen.py:178-183` | 36.6 MB | **1.72 GiB** |
| `qwen.py:221` 的 `F.silu(g) * u` 换成 `T.silu_mul(g, u)`（`__init__.py:2979`，已实现） | `qwen.py:221` | 39.8 MB | **1.87 GiB** |
| `qwen.py:199-202` 三个 `+bias` 合成一次 `addmm`（`nn_ops.addmm` 已有，506-512 行） | `qwen.py:196-202` | 9.4 MB | 0.44 GiB |
| `qwen.py:212-213` GQA 用 `expand` 代替 `repeat_interleave`（需要给 CUDA 加零拷贝 expand；**短期可先只做 k**，v 必须实体） | `__init__.py:1455` | 7.3–14.7 MB | 0.34–0.69 GiB |
| causal mask 缓存（`_sdpa_materialized:2607-2610` 每层每步重建 + H2D） | `__init__.py:2607` | 1.05 MB | 0.05 GiB |
| **小计** | | **≈ 94 MB** | **≈ 4.8 GiB** |

**另外单列一条**：`SDPA 改走 lowmem`（`qwen.py:214` 传 `lowmem=True`，或把 `__init__.py:2421` 的默认改 True）
—— 每层省 169.6 MB（前向三份 → 一份 `o` + `m/l`）加反向约 95–117 MB，合计 **≈ 275 MB/层 → ≈ 6.6 GiB**。
它比 #5 其它项加起来还大，但属于"启用已有能力"而非"改算法"：**如果只做一件事且不想动 autograd，
就先把 `lowmem=True` 打开**（前提是 `_sdpa_lowmem_ok` 的 `dh==64` 契约满足，0.5B 正好满足；
建议先拿 `tests/` 的数值对拍跑一遍再上线）。

### #6（附）分配器：把 `empty_cache` 接上 —— 预期 **+4.2 GiB 可用**，但**不降 `peak_live_bytes`**

- **改哪里**：`cuda.py`
  - 在 `qwen.py::sft` 的 `opt.step()` 之后调用 `T._cuda().empty_cache()`（或 `_pool_trim_to(1<<30)`），
    只在 **B 大 / 接近 OOM** 时开（每次 `cuCtxSynchronize` 会掉吞吐，2655-2681 已同步一次）；
  - 或把已经写好但没人调用的 `_pool_reclaim_idle`（2930-2939）接到 `d2h` 之后
    （它本来就设计成"默认流空闲才回收"，不打断流水），并把 `_POOL_IDLE_KEEP`（2148）从 256 MiB 调到 1–2 GiB；
  - `_POOL_CAP_BYTES`（2144）12 GiB → **2 GiB**（`_pool_shed_ready` 的 12.2 GiB 阈值 2819 同步下调）。
- **收益**：`pool_stats()["peak_bytes"]`（live+cached）与 `nvidia-smi` 占用 **−4.2 GiB**；
  **`peak_live_bytes` 不变** —— 这一点必须说清楚，否则会误判"没效果"。
- **验证**：`peak_bytes` 与 `cached_bytes`；同时确认 s/step 没有明显变差（同步次数增加）。

### 收益叠加（B=4/S=512，估算）

| 阶段 | 反向峰值（不含词表张量） | 含 1 份 `[N,V]`（1.16 GiB） |
| --- | --- | --- |
| 现在（台账推算） | 29.85 GiB | 31.01 GiB（实测 29.92 时已 OOM） |
| 只做 #1 | ≈ 20.5 GiB | ≈ 21.6 GiB |
| 只做 #2 | ≈ 18.8 GiB | ≈ 19.9 GiB |
| **#1 + #2** | **≈ 8.5 GiB** | **≈ 9.7 GiB** |
| #1 + #2 + #4 | ≈ 8.5 GiB | ≈ 9.7 GiB（词表从 4 份降到 1 份，省 3.5） |
| 再加 #5（含 SDPA lowmem） | **≈ 3.5–4.5 GiB** | ≈ 4.5–5.5 GiB |

（#4 的收益体现在"词表 4 份 → 1 份"这一列；若原本就只走 1 份，则 #4 只值 0–1 GiB。）
到此 32 GB 卡可以把 batch 再翻几倍，或直接开 bf16 试更大模型。

---

## 7. 待复测的疑点（必须承认的不确定性）

`docs/AMP_PLAN.md:180-189` 与 `CHANGELOG.md:7-8` 声称"分块 CE 把 29.92 → 17.72 GiB（−12.2 GiB）"。
按本文的静态台账，**这个归因与代码不符**：

1. `chunked_ce` 依赖的 `T.checkpoint` 在 CUDA 上是 no-op（`2526-2528`）；
2. `CHUNK_TOKENS` 默认 8192 > N=2048，实际**只有 1 块**（`qwen.py:407`、`632-641`）；
3. 分块只作用于 CE 段，最大可解释的差值是 3 份 `[N,V]` = **3.48 GiB**，不是 12.2。

而 **17.72 恰好等于"前向结束时的 live"**：`权重 1.84 + Adam 3.68 + 激活 11.26 + 1 份 logits 1.16 = 17.94 GiB`
（台账 §0）。**反向峰值按台账应为 29.85 GiB**，如果那次真的"跑通"，说明它测到的可能不是同一条路径。

**最可能的解释（待证）**：两次测量的 **shape 不同** —— `sft(batch=4)` 的 `_make_xy`（`qwen.py:308-329`）
永远产出 `[1, S]`，即"batch=4"其实是 **4 次 B=1 的微批**（单图激活 ≈ 3.2 GiB），
而不是一次 `[4,S]` 的大图（单图 11.26 GiB）。若如此，12.2 GiB 里的大部分是 **batch 语义差异**，不是分块 CE。

**证伪/证实的最短实验（10 分钟）**：
```
# 两种形状，各自打印 pool_stats()["peak_live_bytes"]
A) ids.shape == (4, 512)  # 一次 forward            -> 期望 ≈29.9 GiB
B) for _ in range(4): ids.shape == (1, 512) ...    # 4 次微批 -> 期望 ≈17.9 GiB
```
且两次都**不要**开 CHUNK_CE。若 B 也落在 17~18 GiB，则"分块 CE 省 12.2 GiB"可以正式撤下，
改写成"分块 CE 省 ≤3.5 GiB（且需先修 `checkpoint` 才拿得到）"。

---

## 附录 A：单层前向图节点明细（B=4/S=512，N=2048）

| 节点 | 大小 (MB) | 说明 |
| --- | --- | --- |
| `ln1` 输出 | 7.34 | RMSNorm，反向重算（省得对） |
| `q = h@Wq.T` | 7.34 | |
| `k` / `v = h@Wk/Wv.T` | 1.05 ×2 | KV 头只有 2 |
| `q+bias` / `k+bias` / `v+bias` | 7.34 + 1.05 + 1.05 | 三个独立节点，可合成 `addmm` |
| `q.permute` / `k.permute` / `v.permute` | 7.34 + 1.05 + 1.05 | **真拷贝**（PyTorch 是视图） |
| RoPE（q，6 节点） | 39.80 | slice 3.67 + neg 3.67 + cat 7.34 + mul 7.34 + mul 7.34 + add 7.34 |
| RoPE（k，6 节点） | 5.20 | 同上，KV 头 |
| `k.repeat_interleave` / `v.repeat_interleave` | 7.34 ×2 | GQA 复制 |
| `scores = q@k^T` | 58.72 | `_attn_scores`（2569） |
| causal mask 常量 | 1.05 | 每层每步重建 + H2D |
| `scores + mask` | 58.72 | |
| `p = softmax(...)` | 58.72 | 反向用 y |
| `o = p@v` | 7.34 | |
| `o.permute` | 7.34 | 真拷贝 |
| `out_proj` | 7.34 | |
| `x = x + attn` | 7.34 | `fused_add_` 因 `requires_grad` 退化（3129-3131） |
| `ln2` 输出 | 7.34 | |
| `g = h2@w1` | 39.85 | FF=4864 |
| `u = h2@w2` | 39.85 | |
| `silu(g)` | 39.85 | |
| `act = silu(g)*u` | 39.85 | |
| `w3 out` | 7.34 | |
| `x = x + ffn` | 7.34 | |
| **合计** | **480.25** | ×24 层 = **11.26 GiB** |

`reshape`（3 处）为视图，计 0。

## 附录 B：关键行号索引

| 主题 | 位置 |
| --- | --- |
| `Tensor.__slots__` / 构造 | `__init__.py:570-599` |
| CPU `backward`（topo / grads / 每节点 `.grad`） | `__init__.py:660-690`（668-680 / 681 / 686-688） |
| CUDA `_backward_cuda`（topo / grads / 叶子 `.grad` / 无释放） | `__init__.py:1000-1045`（1013-1025 / 1026 / 1036-1042 / 1044） |
| `_acc`（只写不删） | `__init__.py:1565-1573` |
| `no_grad`（只关"调用"） | `__init__.py:494-507`；门控 `689`、`1043` |
| `detach()` 走 D2H | `__init__.py:695-699` |
| `checkpoint()` / CUDA 早退 | `__init__.py:2509-2559`（**2526-2528**、2545-2546、2548-2558） |
| SDPA 物化 / lowmem | `__init__.py:2599-2612`、`2615-2626`、`2629-2661`、`2421/2434-2436` |
| softmax / silu 保存 | `__init__.py:2664-2678`、`2681-2694` |
| CE（CUDA）/ CE(ignore) | `__init__.py:2714-2750`（2738-2744）、`2776-2826`（2809/2819/2820） |
| 死代码缓存 | `__init__.py:2753-2773`、`3007/3032-3044` |
| `silu_mul` / `fused_rope` / `fused_add_` | `__init__.py:2979-3000`、`3068-3116`、`3119-3137` |
| `RMSNorm._forward_cuda` | `__init__.py:3214-3236` |
| `AdamW`（m/v） | `__init__.py:3830-3888`（3844-3845 分配） |
| 模型逐层前向 | `qwen.py:185-223`（rope 209-210、GQA 212-213、SDPA 214、SwiGLU 221） |
| `chunked_ce` | `qwen.py:615-642` |
| `sft` 循环（跨步滞留） | `qwen.py:386-418`（400-411） |
| 池：桶 / 上限 / release / empty_cache / 死代码 | `cuda.py:2125-2140`、`2144-2152`、`2758-2803`、`2655-2681`、`2908-2939` |
| 池：OOM 兜底 / 报错文案 | `cuda.py:2980-2984`、`2996-3023` |
| `pool_stats`（含 `peak_live_bytes`） | `cuda.py:2683-2704` |
| 视图 vs 拷贝 | `cuda.py:3733-3748`（view）、`3496`、`3461-3494`、`3556`、`3624` |
| PyTorch：调度引用计数 | `torch/include/torch/csrc/autograd/graph_task.h:31-32` |
| PyTorch：`release_variables` / 边不带数据 | `.../autograd/node.h:561`、`node.h:682` |
| PyTorch：checkpoint 实现 | `torch/utils/checkpoint.py:271-275` |
| PyTorch：empty_cache | `torch/cuda/memory.py:216-228` |
| PyTorch：expandable_segments | `torch/include/c10/core/AllocatorConfig.h:296,320-353`；`CachingDeviceAllocator.h:127` |
| PyTorch：视图语义 / SDPA 后端 | `torch/_tensor_docs.py:6776`；`torch/nn/functional.py:6353-6433` |
