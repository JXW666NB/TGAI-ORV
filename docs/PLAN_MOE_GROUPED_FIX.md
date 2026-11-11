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

