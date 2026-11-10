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
