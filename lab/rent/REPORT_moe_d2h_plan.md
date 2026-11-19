# MoE 路由的 12 次阻塞式 D2H：取证、方案与结论

> 任务：为「消除 MoE 路由的 12 次阻塞式 D2H」做探索 + 方案 + 风险（不落地、不改源码）。
> 结论先行：**no-go**。实测这 12 次 D2H 只值 **5.1 ms/步（seq2048 的 1.3%）**，
> 不是 50~56 ms；它们的主机阻塞时间是真的（29.8 ms/步），但**已经被 GPU 的排队工作完全遮住**。
> 全部 26 次 D2H 一起拿掉也只有 **5.8 ms**。见 §2、§6。

---

## 0. 口径与环境（先钉死，否则数字不可比）

| 项 | 值 |
| --- | --- |
| 远端 | Tesla V100-PCIE-32GB，驱动 580.88，Python 3.11.9 |
| ORV | `tgai_go.__version__` 报 **0.9.97**；但 `train/` 8 个关键文件 md5 与**本地 HEAD 1.0.6 完全一致**（`_moe_env.py` 实测），所以本文行号/行为都对应 HEAD 代码 |
| 配置 | seq=2048（§2.5 另有 512）、vocab=96000、20 步、ckpt 关、`--amp gemm`、batch=1、12 层、d=1024、d_ff=3328、E=4+1 共享、top_k=2、n_q=16、csa=4、`--opt int8gpu` |
| 基线复现 | `step_s_steady` = **0.38266 s**（探针跑）/ 0.38169、0.38067（消融对照跑）→ 与给定基线 0.382 s 一致 |
| 计时 | 步时用训练器自己的 `step_s_steady`（`secs[2:]` 均值）；kernel 用 CUDA event（`prof_event_kernels.py`，**不**用"调用前后各 sync"）；D2H 主机阻塞用 `perf_counter` 包住 `cuMemcpyDtoHAsync+cuStreamSynchronize`，只包真正的那一次调用 |
| 微基准 | CUDA event 一对包住"一整块"（一层/12 层），每迭代只 `rt.sync()` 一次；每个 block 先 3 次预热再 10 次取均值 |

---

## 1. 现状取证：这 12 次 D2H 读什么、干什么、为什么必须在主机

### 1.1 生产路径（bench 真正跑的那条）

`v3_pretrain.forward_loss` → `MOE.route_topk` → `MOE.moe_topk_orv` → `moe_dispatch_orv`：

| 位置 | 代码 | 作用 |
| --- | --- | --- |
| `v3_moe.py:110-111` | `idx_buf, _val = rt.moe_topk(sel._gpu, k)`；`gate_buf = rt.moe_gather_pairs(...)` | 设备端 top-k，产出 `idx int32[N,k]` |
| **`v3_moe.py:112`** | **`eids = np.asarray(idx_buf.to_numpy() ...)`** | **★ 第 1 类 D2H：每层 1 次，读回 `int32[2048,2]` = 16 384 B** |
| `v3_moe.py:161` | `e_flat = np.asarray(eids).reshape(N*k)` | 主机拿到 pair 级专家号 |
| `v3_moe.py:160` | `x_rows = index_select(x, 0, tok)`（`tok = repeat(arange(N),k)`） | 设备 gather；`tok` 是主机常量，每层 H2D 16 KiB |
| `v3_moe.py:197` | `order = np.argsort(eids, kind="stable")` | **主机排序**：按专家分组，稳定序 |
| `v3_moe.py:202` | `xs = index_select(x_rows, 0, order)` | 把 `order` H2D（16 KiB）再设备 gather |
| `v3_moe.py:208-218` | `counts = np.bincount(eids)`；`for e: if end > start: rows = xs[start:end]` | **主机 Python 循环 + 可变形状切片**：每个专家的 GEMM 行数 `count_e` 不同 |
| `v3_moe.py:204` | `g_sorted = gate[order]` | 1-D 数组索引（见 §1.3 的第二个 D2H） |
| `v3_moe.py:221-226` | `cat(parts)` → `scatter_add(y_sorted*g_sorted, order, N)` | 散回原位；反向是 `gather_rows` |

### 1.2 为什么必须在主机侧（三条原因都成立，不是"某一处偷懒"）

1. **Python/NumPy 控制流**：`argsort` / `bincount` / 逐专家 `for` + `continue`（`v3_moe.py:197,208-218`）都在主机。设备端做不了"按数据的循环"。
2. **可变形状分配 + 内核标量参数**：`moe_gather_rows` 的输出是 `empty((m, d))`、grid=`m`，`m` 由 `count_e` 决定（`cuda.py:8449-8456`）；`moe_scatter_rows` 同样（`cuda.py:8458-8465`）。`begin`/`m` 是 `ctypes.c_int` 实参。库内 `MoE._combine_cuda` 也一样：`off` 读回后 `begin=int(off[e])`、`count=off[e+1]-begin` 直接当标量发（`__init__.py:5007-5015`）。
3. **主机索引数组要回传**：`order`（int32，16 KiB）在 `index_select` 和 `scatter_add` 里各 H2D 一次；`tok`（16 KiB）同理。每层至少 4 次 16 KiB 的 H2D（这部分只值 ~0.1 ms，列在这里是为了说明"分派确实依赖主机索引"）。

### 1.3 顺带查出来的第二个 MoE D2H（比路由那个更大，但同样被遮住）

`gate[order]` 是 **1-D** 数组索引，其反向走了主机兜底：

* `__init__.py:1948`：`flat_ok = (shape == (n_src, row_size))`，对 1-D 形状 `(4096,)` 恒为 **False**（要 `(4096,1)` 才成立）→ 不进 GPU 的 `scatter_add_rows`；
* 落到 `__init__.py:1954`：`g_np = np.asarray(g.to_numpy(), ...)` → **第 2 类 D2H：`float32[4096]` = 16 384 B，每层 1 次**，随后 `np.zeros(4096)` + `np.add.at` + `rt.h2d(full)`。

（这是 `__init__.py` 里的通用 `_array_index_cuda`，不是 MoE 专用；但 v3 的 MoE 分派每层都踩一次。）

### 1.4 库内 `MoE` 类（题目指的 L4660）是另一条路径

`MoE._combine_cuda`（`__init__.py:4932`）每次前向读回的是 **`off_buf`：`int32[E+1]` = 20 B**（`__init__.py:4977-4979`），用于：
`if count <= 0: continue` 跳空专家、`begin/count` 当标量实参、以及 `moe_gather_rows` 的可变形状。**bench 用的 v3 模型不走这条**（走 §1.1），两者都各 1 次/层，但读回内容不同（16 KiB vs 20 B），本文的实测数字全部属于 §1.1。

---

## 2. 实测

### 2.1 每步 26 次 D2H 的逐项拆分（seq2048，20 步均值）

| # | 调用点（探针抓的栈） | 次数/步 | 每次字节 | 每次主机阻塞 | 每步主机阻塞 |
| --- | --- | --- | --- | --- | --- |
| 1 | `v3_moe.py:route_topk:112` `idx_buf.to_numpy()`，`int32[2048,2]` | **12** | 16 384 | **0.47 ~ 7.50 ms**（均值 2.48） | **29.79 ms** |
| 2 | `__init__.py:bw:1954`（`gate[order]` 反向兜底），`float32[4096]` | **12** | 16 384 | **4.13 ~ 18.42 ms**（均值 7.56） | **90.73 ms** |
| 3 | `v3_pretrain.py:_scalar:125`（loss 标量） | 1 | 4 | 9.07 ~ 10.25 ms | 9.84 ms |
| 4 | `nn_ops.py:clip_grad_norm_:248`（gnorm） | 1 | 4 | 0.03 ms | 0.03 ms |
| 5 | `v3_ops.py:_rope_table`（仅头两步各 2 次，一次性） | ~0.1 | 524 288 | 0.16 ms | 0.02 ms |
| — | **合计** | **26.1** | — | — | **130.42 ms/步** |

**这张表纠正了此前报告的口径**：以前记的是"26 次、合计 108.6 ms、4.18 ms/次"，把三类完全不同量级的东西平均了。真实情况是 2.48 / 7.56 / 9.84 / 0.03 ms 四档。**MoE 路由那 12 次是 29.8 ms，不是 50~56 ms。**

### 2.2 主机阻塞时间 ≈ 该点的队列深度，与字节数无关

同一批 12 次"路由 D2H"（一层一次）的阻塞时间差 15 倍：

* CSA 层（9 层）：**0.47 ~ 1.36 ms**
* Global 层（第 3/7/11 层）：**6.01 ~ 7.50 ms**（全注意力往队列里塞的工作多得多）
* 反向里的 12 次（gate bwd）：**4.13 ~ 18.42 ms**（反向队列更深）
* clip 里那次：**0.03 ms**（此时队列已排空）

16 KiB 在 PCIe 3.0 x16 上约 1.3 µs ⇒ **阻塞时间 99.9% 是"等流排空"，不是拷贝**。

### 2.3 第 2 步 vs 第 18 步（1-based）

| 项 | 第 2 步 | 第 18 步 |
| --- | --- | --- |
| 路由 D2H 总阻塞 | 28.15 ms | 31.51 ms |
| 逐层（12 层，ms） | 0.54,1.12,1.13,**7.26**,0.49,0.67,1.16,**7.38**,1.18,0.58,0.64,**6.01** | 0.47,1.05,1.18,**7.35**,1.22,1.23,0.70,**7.37**,0.89,1.36,1.22,**7.50** |
| gate-bwd D2H 总阻塞 | 91.68 ms | 94.61 ms |
| loss 标量 | 9.07 ms | 10.25 ms |
| 每步 D2H 合计 | 128.93 ms | 136.41 ms |
| 每步墙钟（forward→forward） | 0.3879 s | 0.3815 s |

结论：**早/晚期对阻塞时间几乎没影响**（负载差 <3%），因为瓶颈是队列深度而不是数据量。

### 2.4 决定性实验：把 D2H 换成同形状的主机数组（不改源码，运行时打补丁）

`probe_d2h_ablation.py` 只替换 `RT.d2h`：路由那次返回**上一轮真实 dump 的 `eids`**（同一分布、同样的 kernel 形状），gate-bwd 那次返回 `ones(4096)`，其余照旧。这样**下游控制流、H2D、所有 kernel 形状完全不变，只去掉"阻塞拷贝 + 流同步"**。

| 消融 | 去掉什么 | steady（两次） | Δ vs 对照 |
| --- | --- | --- | --- |
| `none`（对照） | — | 0.38169 / 0.38067 | — |
| `route` | 12 次路由读回（29.8 ms 阻塞） | 0.37604 / 0.37614 | **−5.1 ms** |
| `gatebwd` | 12 次 gate-bwd 读回（90.7 ms 阻塞） | 0.38075 / 0.38142 | **−0.0 ms** |
| `both` | 上两者（共 120.5 ms 阻塞） | 0.37560 / 0.37544 | **−5.8 ms** |
| `loss` | 只去 loss 标量 | 0.38081 / 0.38173 | 0.0 ms |
| `scalar` | loss+gnorm | 0.36791 / 0.36613 | −14.2 ms ⚠️ **不可用作 D2H 结论**：gnorm 被伪造成 1.0 后 `clip_grad_norm_` 的 `if norm > max_norm` 不成立（`nn_ops.py:254`），**整段梯度缩放被跳过**，这是省了 GPU 工作，不是省了 D2H |

seq512 同法（`ablate_d2h_seq512.ps1`）：对照 0.2320 / 0.2326，`route` 0.2325 / 0.2325，`both` 0.2336 / 0.2320 ⇒ **也没有收益**。

### 2.5 GPU 到底忙不忙（当前 HEAD，CUDA event 逐 leaf）

`PROF_LEAVES=1` 跑 12 步（打点会抬高墙钟，但 kernel 时间不受影响）：

* kernel 叶子合计 **345.43 ms/步**，段内空档 **35.79 ms**，打点墙钟 0.452 s
* 干净步时 0.381 s ⇒ **GPU 忙 ≈ 345/381 = 90.6%，空转 ≈ 36 ms/步**
* 分段 inclusive：forward 141.2 / backward 248.9 / clip 33.6 / optimizer 22.6 ms；段外只剩 ~6 ms
* 主机侧：`cuMemAlloc` 14.8 次/步（1.0.5 修池后已从 1225 次降下来）、`cuCtxSynchronize` 2 次/步

⇒ **"108.6 ms 阻塞 = GPU 空转 130 ms" 这个结论属于 1.0.4 时代那次打点跑**（那次墙钟 497 ms，扣掉 ~7 ms 打点 ≈ 490 ms，正好对上 CHANGELOG 里 1.0.4 的 seq2048 0.489 s；1.0.5 修池后 0.398、1.0.6 接回 fp16 后 0.382）。在 0.382 s 的当前版本上 GPU 已接近饱和，主机阻塞被 GPU 工作遮住了 —— 这就是为什么拿掉 120 ms 阻塞只省 5.8 ms。

---

## 3. 真实路由分布（从真实 20 步里 dump）

**dump 方式**：`probe_moe_d2h.py` 包住 `MOE.route_topk`，它返回的 `eids` 本来就是主机 numpy（现有 D2H 的产物），直接 `np.bincount` 统计，不额外拷贝；第 2 / 第 18 步的 12 层 `eids` 存进 `moe_route_eids.npz`（`s2_l0..11` / `s18_l0..11`），微基准直接用这份分布。

### 3.1 每层每专家 token 数（seq2048，N·k/E = 1024 为期望值）

| | 第 2 步 | 第 18 步 |
| --- | --- | --- |
| 12 层 min（逐层取最小专家） | 914 ~ 1052 | **784** ~ 1012 |
| 12 层 median | 1018 ~ 1047 | 948 ~ 1031 |
| 12 层 max（逐层取最大专家） | 1032 ~ 1110 | 1045 ~ **1311** |
| 变异系数 CV = std/mean | **0.009 ~ 0.069** | **0.014 ~ 0.204** |

第 18 步最不平衡的一层：`counts=[889, 978, 902, 1311]`（CV 0.204），max/mean = **1.28**。
⇒ **`pad=1.25` 在这一层就装不下（1311 > 1280）**；`pad=1.5`（1536）才安全。

> 注意：这个 bench 的 MoE **没有** aux loss / bias（`balance="none"`，`v3_pretrain` 也没传 `capacity_factor`）。真实长跑里路由可以继续集中，CV 只会更大；本表只是 20 步内的观察值，**不能当成上界**。

---

## 4. 方案对比

先给微基准背景（§5 详解）：当前 MoE 分派 12 层 **29.3 ms/步**（前向），其中 GEMM 21.6 ms、其余 7.7 ms 是 gather/scatter/cat/索引。

| 方案 | 做法 | 预期收益（实测/推算） | 风险 | 工作量 | 确定性 | 显存 | CUDA Graph |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **A. 免排序 padded-dense** | 固定 `C=ceil(N·k/E·pad)`，`rt.moe_permute` 设备端分槽 → 固定形状 GEMM + 掩码 → unpermute | **净 ≈ −3 ms**：主机 −5.1（实测），kernel **+2.0**（pad1.25 实测） | ①pad1.0 丢 1525 pair（第18步）；②极端不均衡会丢更多（实测 one-hot：pad1.25 丢 1536）；③`moe_permute` 用 atomicAdd 分槽 ⇒ 行序每次不同 ⇒ dW 求和顺序变 ⇒ **逐位不可复现** | 中：`v3_moe.moe_topk_orv` 换实现 + 反向 + 掩码，~150-250 行；若换确定性分槽需再加一个 scan 内核 | ✗（除非改成分槽确定性） | +4~8 MB/层（pad1.25/1.5） | ✓（permute 本来就是为免同步写的） |
| **B. 设备端计数 + 固定上限 + strided-batched** | 偏移留设备，`_bmm_op` 直接吃设备偏移 | **不可行**：现成 3-D `T.bmm` 只有 **fp32**，实测 `[4,C,d]@[4,d,d_ff]` 是 **7.02 ms/层 vs fp16 1.80 ms/层（3.9×）**；12 层 = **+62 ms/步**。除非新写 fp16 strided-batched 封装，那收益就退回 A 的 −3 ms | 高（动 `cuda.py`） | 大（新 cublasGemmEx batched 封装 ~80 行 + 接入 + 对拍） | 同 A | 同 A | ✓ |
| **C. 流水/预取** | 提前一层发起 D2H | **0 ms**：路由值下一行就要用来 `argsort`，无可重叠对象；且阻塞本来已被 GPU 遮住（`both` 消融 −5.8 ms） | — | — | — | — | — |
| **D. 不动 D2H，改 GPU 忙时**（推荐） | 见 §6.2 | attention fp32 batched GEMM 20.3 ms；adamw 22.5；clip 缩放段 ≈14.5；moe_scatter_pairs 8.3 | 低（都是纯 kernel 优化，逐项 A/B 可验） | 小~中 | 不变 | 不变 | 无关 |

**为什么 A 的 −3 ms 也不值得**：−5.1 是"阻塞全去掉"的上界，+2.0 是 pad1.25 的确定成本；净 −3 ms 落在消融噪声（±1 ms）里，而代价是丢 token 的容量风险 + 新的非确定性 + 150~250 行热路径改写。

---

## 5. 微基准：现状路径 vs padded-dense（seq2048，真实规模 + 真实分布）

脚本 `bench_moe_dispatch.py`：N=2048、k=2、d=1024、d_ff=3328、E=4；x 是 `[N,D]`（token 主序，与生产一致）；权重按生产形状 `W1/W2=(d_ff,d)`、`W3=(d,d_ff)`；`T.amp_on(fp16)` 且权重登记进 f16 影子缓存（与 `--amp gemm` 一致）；内部 D2H 在计时期间**打桩**（保留全部 kernel 形状/发射），主机阻塞已在 §2 单独测。每层跑 12 层真实分布，event 计时。

> 口径说明：**现状路径是"逐专家 2-D fp16 GEMM"**（`moe_dispatch_orv` 里 `for e: V.situ_glu_orv(rows, W1[e], ...)`），不是 strided_batched —— 生产里没有专家维的 batched GEMM（融合分组路径 `cublasSgemmGroupedBatched` 因 `ewise_mul_v4` 越界默认关闭，`__init__.py:4562-4568`）。所以"现状 vs padded-dense"的差异 = gather/scatter 的行数 × pad + GEMM 行数 × pad，而不是"批量 vs 逐次调用"的差异。

### 5.1 一层/12 层的 GPU 时间（ms）

| 路径 | 前向 /12 层 | 前向 /层 | 前向+反向 /12 层 | 丢弃 pair |
| --- | --- | --- | --- | --- |
| A 现状（第2步分布） | **29.25** | 2.44 | 97.90 | 0 |
| A 现状（第18步分布） | **28.77** | 2.40 | 98.29 | 0 |
| B pad=1.00 (C=1024) | 26.13 / 26.18 | 2.18 | 87.15 / 87.19 | **661 / 1525** |
| B pad=1.25 (C=1280) | 31.21 / 31.30 | 2.60 | 96.49 / 96.52 | 0 / **31** |
| B pad=1.50 (C=1536) | 33.82 / 33.72 | 2.82 | 115.59 / 115.53 | 0 / 0 |

### 5.2 pad 因子曲线（前向，相对 A 的 Δms/步；A=29.0 取两步均值）

| pad | C | GEMM 行数 | 前向 ms | Δ vs A | 正确性 |
| --- | --- | --- | --- | --- | --- |
| 1.00 | 1024 | 4096 | 26.2 | **−2.8** | ✗ 丢 661~1525 pair，max\|Δ\| 0.131（\|out\|max 0.171） |
| 1.25 | 1280 | 5120 | 31.3 | **+2.3** | ✓ 单层无丢弃时 max\|Δ\| 9.1e-7；但 12 层合计已丢 31 pair（第18步分布），被丢的 pair 输出会缺一块 |
| 1.50 | 1536 | 6144 | 33.8 | **+4.8** | ✓ max\|Δ\| 2.9e-7 |

**pad 因子 ↔ 时间**：pad 每 +0.25，前向 +2.0~2.5 ms/步（GEMM 行数线性上涨）。
**pad 因子 ↔ 正确性**：pad1.0 必丢（CV 0.2 时 max/mean=1.28）；pad1.25 在 20 步内刚好够、但已丢 31 pair；pad1.5 才稳。

### 5.2b 极端分布下的丢弃（同一套 N=2048,k=2,E=4；期望 1024/专家）

| 分布（合成） | 每专家 token 数 | pad1.0 (1024) | pad1.25 (1280) | pad1.5 (1536) |
| --- | --- | --- | --- | --- |
| 真实第18步（最不平衡层） | 889 / 978 / 902 / 1311 | 287 | **31** | 0 |
| 两专家被吃满（one-hot） | 2048 / 2048 / 0 / 0 | 2048 | **1536** | 1024 |
| 70/20/10 倾斜 | 1433 / 1842 / 615 / 206 | 1227 | **715** | 306 |
| 一个专家全空 | 1366 / 1365 / 1365 / 0 | 1024 | **256** | 0 |

⇒ **固定容量对"路由集中"没有安全区**：两专家吃满时连 pad1.5 都要丢一半 token。当前 bench 的训练配置没有 aux loss / bias，路由集中是完全可能的，所以 A 方案的容量风险不是理论风险。

**其它边界**：`某专家 0 token` 是安全的（现状 `if end > start: continue`；padded 路径该专家 gw 全 0、贡献恰为 0）；`top_k=1` 时 pair 数减半、`C=N/E`，相对不均衡与 k=2 同构（风险不变）；`E=1` 时 `C=N·k`，无失衡问题；`seq512` 实测无收益（§2.4）。

### 5.3 GEMM-only：为什么方案 B（strided-batched）不可行

一层 4 专家 × 3 个 GEMM（84 GFLOP）：

| 形状 | fp16 2-D（现状，每专家一次） | fp32 `T.bmm`（现成 3-D 路径） | 倍数 |
| --- | --- | --- | --- |
| m=1024 | **1.80 ms**（46.6 TFLOPS） | 7.02 ms（12.0 TFLOPS） | 3.9× |
| m=1280 | 2.24 ms | 8.59 ms | 3.8× |
| m=1536 | 2.44 ms | 10.38 ms | 4.3× |

⇒ 用现成 `_bmm_op`/`strided_batched` 做 padded-dense，**每步多 62 ms**，直接把 seq2048 打回 0.44 s。

---

## 6. go / no-go

### 6.1 结论

| 方案 | 预期收益（ms/步，占 0.382 s） | 风险 | 工作量 | 判定 |
| --- | --- | --- | --- | --- |
| A padded-dense（pad1.25） | **≈ −3 ms（−0.8%）** | 中高（丢 token / 非确定性） | 150~250 行热路径 | **no-go** |
| B 设备端计数 + strided batched | 现成路径 **+62 ms**；自写 fp16 batched 后 ≈ A | 高（动 `cuda.py`） | 大 | **no-go** |
| C 预取/流水 | **0 ms** | — | — | **no-go** |
| D 改 GPU 忙时 | 单项 8~20 ms，可叠加 | 低 | 小~中 | **go** |

**判定依据（任务给的阈值是 30 ms）**：MoE 路由 12 次 D2H 的**全部**可回收时间是 5.1 ms，扣掉 padding 成本后 3 ms 左右 —— 比阈值低一个数量级，而且换来的是容量丢 token 与逐位不可复现。**该放手。**

### 6.2 精力应该放哪（都有当前 HEAD 的实测数字，且 GPU 已 90% 忙 ⇒ kernel 时间省 1 ms 就是步时省 1 ms）

按 `prof_kernels_now.json`（12 步、逐 leaf CUDA event）排序：

1. **attention 的 fp32 batched GEMM：20.3 ms/步**
   `cublasSgemmStridedBatched [1024,1024,2048,1]` 30 次 = 10.84 ms + `[64,2048,2048,16]` 12 次 = 9.50 ms。
   两条都走 `__init__.py:1234/1254` 的 `sgemm_nt/tn_strided_batched`，而这两条**没有 fp16 分支**；batch=16 那条就是 3 个 global 层注意力的反向（`[64,2048,2048,16]` = dq 形状），batch=1 那条（30 次/步）形状与 CSA 分支一致、**我没有逐层定位到调用点（未验证）**。V100 上 fp16 是 4×，这两条保守能拿 **−8~12 ms**。
   *这和 1.0.6 刚做的"3-D GEMM 接回 fp16"是同一类问题，只是发生在 batched/4-D 路径上。*
2. **`adamw_multi_m8v16` 22.50 ms/步（1 次调用）**：int8 优化器的融合核，带宽/占用率问题，值得单独看。
3. **clip 段 33.6 ms（inclusive）**，其中梯度缩放那一趟 ≈ **14.5 ms**（消融实测：把 gnorm 伪造成 ≤max_norm，`nn_ops.py:254` 的 `if norm > max_norm` 不成立，缩放整段被跳过，步时 −14.2 ms）。缩放本身走 `nn_ops.py:270-281`：大张量（>1M 元素）**逐张量** `scale_inplace`，321 个张量逐个发射。
4. **`moe_scatter_pairs` 8.29 ms/步（12 次）**：MoE 门控反向的散射，12 次 8.3 ms，可以并进路由 kernel 或换成按行 warp 归约。
5. `softmax_lastdim(_bwd)` 10.5 ms（6 次）、`sumsq_partial` 5.0、`mul_bcast8` 6.6。

顺带一个**免费清理**（不是性能项）：`__init__.py:1948` 的 `flat_ok` 对 1-D 索引恒 False，导致 `gate[order]` 反向走主机兜底。改法 3~6 行（允许 `len(tail)==0`，把 `scatter_add_rows` 的 `(n_src,1)` 结果 reshape 回 1-D；对重复下标要么只在下标唯一时走 GPU、要么接受 atomic 非确定性）。**实测步时收益 0 ms**（`gatebwd` 消融），但能省 90.7 ms/步主机时间、少 12 次 D2H + 12 次 H2D，代码更统一。**只有在你要动这条路径时才做**。

### 6.3 建议的落地顺序与验证标准

1. **先做 attention batched GEMM 的 fp16 化**（方案 D-1）
   * 验证：① 对拍同一批 q/k/v，fp16 vs fp32 输出 `max_rel ≤ 5e-4`（AMP 的既有口径）；② `step_s_steady` 中位数下降 ≥ 8 ms（3 次预热+3 次取中位，极差 <1%）；③ amp-off 时逐位不变（fp16 只在 amp 开时生效）。
2. **再看 clip 缩放段**（D-3）
   * 验证：gnorm 与逐位裁剪结果不变（同 gnorm 下 `scale` 逐位一致）；clip 段 inclusive 从 33.6 → ≤20 ms。
3. **adamw / moe_scatter_pairs**（D-2/D-4）
   * 验证：参数更新逐位一致（同输入两次 step 的 `p` 逐位相同）；kernel ms 下降。
4. 若还想碰 MoE 分派：**只做"把 `gate[order]` 反向切到设备端"这一条**（§6.2 免费清理），**不要**做容量 padding。

全局护栏：任何改动都必须过 `git status -- python/` 干净（本轮没改源码）；每项都要"3 次跑取中位 + 极差 <1%"；对拍要包含 amp-off 逐位、amp-on 1e-4 两档。

---

## 7. 新建文件清单与未验证项

### 7.1 新增文件（全部在 `lab/rent/`，`python/` 未改动，`git status -- python/` 为空）

| 绝对路径 | 作用 |
| --- | --- |
| `D:\DSH-Workspace\TGAI-对话档案\_tgai_orv\lab\rent\probe_moe_d2h.py` | 20 步真实跑 + 运行时只读打点：逐次 D2H（字节/dtype/形状/栈/主机阻塞）、逐层路由分布、step2/18 dump |
| `…\lab\rent\probe_d2h_ablation.py` | **决定性消融**：只替换 `RT.d2h`，去掉指定 D2H 后测步时（`--ablate none/route/gatebwd/both/loss/scalar`） |
| `…\lab\rent\ablate_d2h.ps1` / `ablate_d2h_seq512.ps1` | seq2048 / seq512 的消融矩阵（每档 2 次） |
| `…\lab\rent\bench_moe_dispatch.py` | 微基准：现状分派 vs padded-dense（pad 1.0/1.25/1.5）vs fp32 bmm；含空专家/极端不均衡丢弃统计 |
| `…\lab\rent\probe_moe_equiv.py` | padded-dense 与 numpy 参考的对拍（tiny 2.2e-11；真实规模 A vs B） |
| `…\lab\rent\probe_moe_grad_equiv.py` | 梯度对拍（dx / dW1 / dW2 / dW3，pad1.25，真实分布） |
| `…\lab\rent\_moe_env.py` | 远端版本/8 个 train 文件 md5/能力探测（只读） |
| `…\lab\rent\moe_d2h.json` | 26 次 D2H 逐次记录 + 逐层分布 + 每步表 |
| `…\lab\rent\moe_route_eids.npz` | 第 2/18 步 × 12 层的真实 `eids`（微基准的输入分布） |
| `…\lab\rent\bench_moe_dispatch.json` | 微基准原始结果 |
| `…\lab\rent\prof_kernels_now.json` | 当前 HEAD 的逐 leaf kernel 归因（345.4 ms/步） |
| `…\lab\rent\ablate_d2h.log` / `ablate_d2h_seq512.log` | 消融原始输出（含每次 `step_s_steady`） |

复现命令（PowerShell，勿嵌套引号）：

```
scp -i %USERPROFILE%\.ssh\v100_tgai lab\rent\probe_moe_d2h.py Administrator@192.168.0.103:D:/orv_bench/
ssh -i ... "D:\orv_pip_0962\Scripts\python.exe -X utf8 D:\orv_bench\probe_moe_d2h.py --data D:\orv_bench\data2048_v96000 --vocab 96000 --seq 2048 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --amp gemm --batch 1 --steps 20 --log-every 99 --no-final-save"
powershell -NoProfile -ExecutionPolicy Bypass -File lab\rent\ablate_d2h.ps1
ssh -i ... "D:\orv_pip_0962\Scripts\python.exe -X utf8 D:\orv_bench\bench_moe_dispatch.py"
```

### 7.2 未验证 / 边界

* **只测了单卡 V100 + 这一套配置**。A100/H100（bf16、更大带宽）上 D2H 相对占比可能不同；没有数据。
* **只测了 20 步**。长跑里路由会继续集中（无 aux loss），CV 可能远超 0.20；本报告的 pad 结论只对 20 步成立。
* **seq512 只做了消融**（无收益），没做 512 的微基准与分布统计。
* `moe_permute` 的确定性只在源码层判断（`atomicAdd(&counts[e],1)` 分槽），**没有实跑两次比对 dW 是否逐位相同**——若要落地 A 方案，这一步必须先做。
* strided-batched 的 fp16 封装只估了工作量（~80 行），**没有原型**，所以"自写后 ≈ A"是推断不是实测。
* 消融里"伪 eids"用的是第 2 步分布回放，与逐步真实分布有 <3% 的负载差异（§3 已量化），对 5 ms 量级的结论不构成影响，但严格说不是"零扰动"。
* `prof_kernels_now.json` 是逐 leaf 打点跑（墙钟 0.452 s，会被探针抬高 ~65 ms），**kernel 合计 345.4 ms 不受影响，分段 inclusive 时间会偏大**，引用时请注意。
