# MoE 反向优化：根因与修复方案（2026-11-11）

> 状态：**根因已定位，修复未实施**。本文记录实测数据与具体方案，避免下一轮重复排查。

## 1. 为什么要做（实测依据）

T=512、28 层 2.97B 的整步分解（`tools/split_512.py`，CUDA event）：

| 段 | 时间 | 占比 |
| --- | --- | --- |
| 前向链（attn+MoE） | 237.0 ms | 30% |
| **MoE 反向** | **332.4 ms** | **42%** ← 最大头 |
| 注意力反向 | 206.6 ms | 26% |
| 词表头 | 11.2 ms | 1% |
| 合计 | 787 ms | — |

注意力那 206.6 ms 已可用逐层 CUDA Graph 压到约 208→29 ms 量级；**剩下最大的一块是 MoE 反向**。

## 2. 根因：MoE 每层退化成 8 次小 GEMM

`MoE._combine_cuda` 本可以调 `_moe_grouped_cuda`（一次 `cublasSgemmGroupedBatched`），
但该路径**默认关闭**（`TGAI_MOE_GROUPED` 未设时为 0），所以走逐专家循环：

```
每层：8 个专家 × 3 个投影（gate/up/down）= 24 次 GEMM
每次 M≈128（top-2 of 8，seq=512 → 每专家约 128 行）
```

微基准（`tools/moe_gemm_bench.py`，单个 gate 投影，8 组 128 行）：

```
① 8 次独立 GEMM   6.11 ms      ← 约 V100 fp32 峰值的 9%
② 1 次分组 GEMM   rc=700 非法访存
```

## 3. 分组路径为什么崩：ABI 签名不匹配（不是行数问题）

`cuda.py::_grouped_ready()` 加载的是 **`cublasSgemmGroupedBatched`**（非 Ex 版），
但 `argtypes` 按 **Ex 版**签名声明：

```c
// 真·cublasSgemmGroupedBatched（CUDA 10/11）
cublasStatus_t cublasSgemmGroupedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,                    // ★ 标量
    const float *alpha,
    const float *const Aarray[], int lda,
    const float *const Barray[], int ldb,
    const float *beta,                      // ★ 在 B 之后、C 之前
    float *const Carray[], int ldc,
    int batchCount, const int groupSize[]);
```

而代码里声明的顺序是
`handle, int* transa, int* transb, int* m, int* n, int* k, float* alpha, ptr A, int* lda, ptr B, int* ldb, float* beta, ptr C, int* ldc, int batchCount, int* groupSize`。

于是 `m` 位置收到的是**指针**，驱动把指针低 32 位当 `int m` 读 → 天文数字的 m
→ 读越界 → `rc=700`。

这也解释了旧注释里"某专家行数 >64 时越界、n=8 正常"的误判：**崩溃与行数无关**，
只要分组路径被走到就必然崩（n=8 时 `_moe_grouped_cuda` 因 `<= 64` 提前返回，所以"没崩"）。

## 4. 修复方案（两条，推荐 A）

### 方案 A（推荐）：pad 到统一 M，用标量 m/n/k

非 Ex 版要求所有组共享 `m/n/k`。MoE 每个专家的行数不同（100~160），
但可以 **pad 到 max(count)**：

```
rows_padded = 每专家 padding 到 M_max（多出的行填 0，算完丢掉）
一次分组调用：m=M_max, n=d_ff, k=d_model, batchCount=8, groupSize 全 1
```

- 代价：约 `M_max/mean(count)` ≈ 1.1~1.2 倍的多余 FLOPs；
- 收益：24 次 GEMM 合并成 3 次（gate/up/down 各一次）；
- 正确性可验证：与逐专家路径逐位或 ≤1e-5 对照。

### 方案 B：改用 `cublasGemmGroupedBatchedEx`

Ex 版才支持逐组 `m/n/k`，签名是
`(handle, transa, transb, const int *groupedM, ..., int groupCount, const int groupSize[], computeType)`。
需要重新声明 argtypes 并把指针数组补齐。**不需要 pad**，但要确认运行时 libcublas 有这个符号。

## 5. 验收标准

1. 分组路径与逐专家路径数值对照 ≤1e-5（加进 pytest）；
2. MoE 反向从 332 ms 明显下降（目标 ≤200 ms），整步 709 ms 相应下降；
3. 全量 pytest 全绿、升版本、发布、回读 sha256。

## 6. 注意

- 改这一段必须**先备份** `cuda.py`（时间戳），因为它是所有 GEMM 的公共路径；
- 分组路径要保留 `TGAI_MOE_GROUPED` 开关，出问题能立刻退回逐专家；
- 关闭时的行为与现在**完全一致**，不能改变现有数值。

## 7. 2026-11-11 更新：签名已修正，但分组调用仍返回 rc=7

已实施：把 rgtypes 改成正确的标量版、_grouped_sgemm 加一致性检查、非 0 返回码**静默回退**。
	ools/grouped_fix_check.py 实测：

| 场景 | 修复前 | 修复后 |
| --- | --- | --- |
| 形状一致，走分组 | rc=700 崩溃 | **rc=7**（参数校验不过）⇒ 自动回退 |
| 形状不一致 | rc=700 崩溃 | 回退，**0.00e+00 逐位一致** |
| 库可靠性 | 一调就崩 | 永不崩、结果永远正确 |

**rc=7 的信息是「parameter number 2 had an illegal value」**，第 2 个参数即 	ransa 位置
（我们传 CUBLAS_OP_T=1）。据此推测该 libcublas 版本的非 Ex 版符号**参数布局与公开文档不一致**，
下一步应验证的两个假设：

1. 该版本把 m, n, k 放在 	ransa, transb **之前**；
2. lpha/beta 需要**设备**指针（cublasSetPointerMode(DEVICE)）而非主机指针。

另外已确认：运行时 libcublas **只有** cublasSgemmGroupedBatched，
没有 cublasGemmGroupedBatchedEx / cublasSgemmGroupedBatchedEx（	ools/grouped_ex_probe.py），
所以方案 B 在这台机器上不可用。

**在 rc=7 调通之前，MoE 反向仍是逐专家循环，不做任何性能宣称。**

## 8. 2026-11-11 更新（二）：放弃 cublasSgemmGroupedBatched，改走 strided-batched

	ools/grouped_rc7_probe.py 试了四组参数组合，结果：

| 用例 | 返回 |
| --- | --- |
| (T,N) 多组（当前用法） | rc=7，**parameter number 2** |
| (N,N) 多组 | rc=7，**parameter number 1**（即 handle 位置） |
| (T,N) 单组 | rc=7，parameter number 1 |
| (N,N) 单组 | rc=7，parameter number 1 |

**连 handle（第 1 个参数）都被判非法**，说明问题不是某个取值，而是这本 libcublas 里
这个符号的**参数布局与公开文档完全不同**（或者该符号并非 cublas 的公开 API）。

**结论：不再尝试这条路。** 继续逐个试参数是低效的，且无论试成什么样都依赖未文档化的行为。

### 替代方案：cublasSgemmStridedBatched（成熟 API，行为确定）

思路：
1. 每层把 8 个专家的权重**堆叠**成连续缓冲 [G, N, K]（每个专家一份，一次拷贝，可缓存复用）；
2. 把 gather 出来的行**pad 到统一的 M_max** 放进连续缓冲 [G, M_max, K]；
3. 一次 cublasSgemmStridedBatched 算完 8 个专家；
4. 从 [G, M_max, N] 里按各自 count 取回。

代价：每层需要额外拷贝（权重堆叠 ~134 MB/层，行 pad）；
收益：24 次小 GEMM → 3 次 strided batched 调用。

### 决策

- 先把**目标 2（Muon 层合并正交化）提前**做——它的收益明确（62.9 s/步 → 可用区间），
  且不需要跟未文档化的 cublas 行为搏斗；
- MoE 融合按上面的 strided-batched 方案重做，作为后续；
- 当前 cuda.py 的分组路径已改成**失败即安全回退**，不影响正确性，可以留在原地。

## 9. 2026-11-11 更新（三）：正确的原语测量 —— 目标 332→200 ms 通过融合无法达成

### 之前那个微基准是错的

早先测出「8 次小 GEMM = 6.11 ms」，但实测整条前向链只有 237 ms —— 自相矛盾。
原因是那次微基准**在计时循环里每次都新建输出 buffer**，把分配算进了 GEMM 时间。

### 正确测量（`tools/moe_prim_bench.py`，预分配 + 预热，seq=512、8 专家各 128 行）

| 单层 MoE 原语 | 时间 |
| --- | --- |
| gate 投影 8× `gemm_nt` | **0.99 ms** |
| down 投影 8× `gemm_nt` | **1.12 ms** |
| router | 0.03 ms |
| `moe_topk` | 0.02 ms |
| `moe_compact` | 0.23 ms |
| `moe_gather_rows` ×8 | 0.11 ms |
| ewise mul ×8 | 0.12 ms |

### 决定性换算

gate 投影：8 × 2 × 128 × 1024 × 4096 = **8.59 GFLOP**，耗时 0.99 ms
⇒ **8.7 TFLOP/s = V100 fp32 峰值（15.7）的 55%**。

**小 GEMM 跑到 55% 已经是高效区间。** MoE 反向的 332 ms 主要是真实计算。

融合能省什么：每次 GEMM 的启动开销约 5~10 µs，8 次约 **40~80 µs**，
相对于单层 2.1 ms 的 GEMM 时间是 **4~8%**。

### 结论

**「MoE 反向 332 ms → ≤200 ms」通过 GEMM 融合（无论 grouped 还是 strided-batched）无法达成。**
需要降低 40% 才能达标，而融合最多给 8%。

真要再快，只能从算法层面减少 MoE 的计算量（更低精度 GEMM、或减少专家数），
那已不属于「融合」这个任务的范畴。

`TGAI_MOE_GROUPED` 开关与「失败即安全回退」保留（正确性已有保障），
但**不再以性能为目标推进**。

## 10. 2026-11-11 更新（四）：修正第 9 节 —— 真正的空间在「小算子 + 同步」，不在 GEMM

第 9 节只测了**前向原语**就下结论，漏掉了真实模块的整体开销。补测（`tools/moe_layer_bench.py`，
真实 `T.nn.MoE` 模块，CUDA 路径）：

| 项 | 单层 | ×28 层 |
| --- | --- | --- |
| MoE 前向 | **6.42 ms** | 180 ms |
| MoE 前向 + 反向 | **13.12 ms** | 367 ms |
| 反向（差值） | **6.70 ms** | 188 ms |

而 GEMM 只占前向 **3.1 ms**（gate 0.99 + up ≈1.0 + down 1.12）
⇒ **前向有 3.3 ms（约 51%）是非 GEMM 开销**。

### 这些开销是什么（已测单项）

| 反向/前向小算子 | 8 次合计 |
| --- | --- |
| silu 反向 ×8 | 0.122 ms |
| ewise mul ×8（[128,4096]） | 0.115 ms |
| scale ×8 | 0.112 ms |

单个 `[128,4096]` 的 mul 若按带宽（约 6 MB / 900 GB/s）应约 7 µs，
实测约 15 µs/次 —— **约 2 倍差距，是启动开销主导**（launch-bound）。

此外每层还有一次 `rt.d2h(off_buf)` **强制主机同步**（为了跳过空专家）。

### 修正后的结论

- 第 9 节说「332 ms 主要是真实计算、融合只能省 4~8%」——**对 GEMM 部分成立，对整体不成立**；
- 真正可优化的是**小算子的数量**和**每层那次 D2H 同步**，不是 GEMM 融合；
- 因此目标「MoE 反向 332 → ≤200 ms」**未必不可达**，但路径要改成：
  1. 把每层 `d2h(off_buf)` 改成设备端条件（去掉 28 次同步）；
  2. 把 gather / silu / mul / scatter 这些小算子做**算子融合**（不是 GEMM 融合），
     把每层二十几次启动压到个位数。

**下一步按这条路做，并实测是否到 200 ms。**

## 11. 2026-11-11 更新（五）：再修正 —— 第 10 节那个 6.42 ms 也被污染了

第 10 节用 `moe_layer_bench.py` 测得单层前向 6.42 ms，据此算出「51% 是非 GEMM 开销」。
但那一次 repeat 少、且**首次调用的内核编译/分配被摊进了均值**。

重测（`tools/d2h_cost.py`，预热更充分）：

| 项 | 时间 |
| --- | --- |
| MoE 前向基线 | **4.18 ms/层** |
| 去掉 d2h 同步（缓存 off） | 4.33 ms/层（**差值 −0.14 ms，噪声内**） |

### 三个候选优化的实际收益（全部实测）

| 候选 | 实测收益 | 折算 28 层 |
| --- | --- | --- |
| GEMM 融合（grouped / strided-batched） | 每次省 5~10 µs 启动 | **约 4~8%** |
| 去掉每层 `d2h` 同步 | **≈ 0**（−0.14 ms/层，噪声） | **≈ 0** |
| 小算子融合（silu/mul/scale 各 ×8） | 约 0.3~0.5 ms/层 | **约 14~28 ms** |

### 最终判定（第三次修正，这次数据自洽）

单层 MoE：前向 **4.18 ms**、前向+反向 **约 11 ms**（⇒ 反向约 6.7 ms）。
其中 GEMM 占前向 3.1 ms ⇒ 非 GEMM 开销约 **1.1 ms（26%）**，不是 51%。

28 层合计约 **308 ms**，与 split_512 实测的 332 ms 吻合。

**要达标需从 332 降到 200 ms（−40%）；三项优化加起来最多约 30~40 ms（−12%）。**
⇒ **目标「MoE 反向 332 → ≤200 ms」在只做"融合/去同步"的前提下不可达。**

想真到 200 ms 只能减少计算量本身：
- 更低精度 GEMM（fp16/bf16 做 MoE 专家投影，误差需另测）；
- 或减少专家数/激活专家数（改模型，不属于本任务）。

### 测量方法上的教训（记下来避免再犯）

本轮三次测同一件事得到了 6.11 / 6.42 / 4.18 ms 三个数，前两个都偏大，原因都是
**首次调用开销被摊进均值**、或**计时循环里新建了 buffer**。
`tools/moe_prim_bench.py` 与 `tools/d2h_cost.py` 都是预分配 + 充分预热的正确写法，
以后性能测量一律按这个范式。

## 12. 2026-11-11 更新（六）：四条路径全部实测完，目标不可达（最终）

### 关键一致性核对

`tools/moe_amp_bench.py` 实测单层 MoE **fwd+bwd = 11.82 ms**
⇒ 28 层 **331 ms**，与 `split_512.py` 报的「MoE 反向 332 ms」**完全吻合**。

这说明那个 332 ms 其实包含**梯度检查点的前向重算**（MoE fwd 4.18 + 真反向 7.6），
不是纯反向。这一点之前没意识到。

### 四条路径全部实测（单层 → 28 层）

| 路径 | 实测 | 折算 | 距 −40% 目标 |
| --- | --- | --- | --- |
| GEMM 融合（grouped / strided-batched） | 每次省 5~10 µs | **−4~8%** | 差很远 |
| 去掉每层 `d2h` 同步 | −0.14 ms（噪声） | **≈ 0** | 无 |
| 小算子融合（silu/mul/scale） | 约 0.3~0.5 ms/层 | **−4~8%** | 差很远 |
| **AMP / fp16 GEMM** | 11.82 → **10.21 ms/层（1.16×）** | **−13.6%**（332→287 ms） | **仍差 87 ms** |

四条加起来乐观估计约 **−25%**（332 → 约 250 ms），**仍达不到 200 ms**。

### 为什么 fp16 只给 1.16× 而不是 2×

理论 2× 需要走 tensor core。这里的专家 GEMM 是 `M=128` 的瘦长矩阵，
tensor core 的 tile 利用率低，且 fp16 输入 + fp32 累加走的是 `cublasSgemm` 的
非 tensor-core 路径，所以只有约 1.16×。

### AMP 的数值代价（实测）

AMP vs fp32 输出相对差 **4.31e-04**。这是 fp16 的正常量级，但**比 ORV 其它路径的
判据（1e-5）大一个量级以上**，属于真实的精度权衡，不能默默打开。

### 最终结论

**「MoE 反向 332 ms → ≤200 ms」在这个模型结构 + V100 上不可达。**
需要 −40%，而四条路径合计最多约 −25%。

要突破只能改模型或换硬件：
- 减少激活专家数（top-2 → top-1，改模型语义）；
- 更小的 d_ff（改模型）；
- 换 A100/H100 并使用真正的 tensor-core fp16/bf16 路径（**本机无硬件，未测**）。

这三条都超出「融合优化」的任务范围。



