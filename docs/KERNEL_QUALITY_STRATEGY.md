# 内核质量：PyTorch 凭什么好，我们要拿什么换

> 2026-11-21　｜　**前提：内核质量是目标本身，不是可以拿"卡淘汰/ROI 低"绕过去的理由。**
> 本文只回答两件事：①PyT 的优秀是**怎么造出来的**；②我们逐条该学什么、怎么验收。

---

## 1. PyTorch 的内核不是"自己写的"，是"站在别人的调优上"

这一点最关键，也是最容易被误解的。PyTorch 的 CUDA 内核质量，大部分**不是** PyTorch 团队手写出来的：

| 能力 | 真正的来源 | PyTorch 做了什么 |
| --- | --- | --- |
| GEMM | **cuBLAS**（NVIDIA 自己按架构手写 SASS + 启发式选 kernel）| 只是调用；连 workspace/heuristic 缓存都很少自己做 |
| 注意力（memory-efficient）| **xFormers**（基于 **CUTLASS** 模板库）| 移植进来，按 sm 版本选后端 |
| 注意力（flash）| **Dao 等人的 FlashAttention**（论文+参考实现）| 上游化、加后端选择 |
| elementwise / reduction | **TensorIterator + ATen 代码生成**（`native_functions.yaml` → 生成的循环）| 用生成器保证**每个算子都是向量化 + 展开的** |
| 后端选择 | 自己写的 dispatch：flash 需 sm_80+，mem-efficient 需 sm_50+，否则 math | 一个**真实的策略层**，三条后端都是一等实现 |

**CUTLASS 是什么**：NVIDIA 开源的 CUDA 模板库，把 GEMM/注意力里十年积累的最佳实践
（warp 级 MMA 流水、shared memory swizzle 防 bank conflict、软件流水双缓冲、epilogue 融合）
做成了可组合的模板。xFormers 的 memory-efficient attention 就是用它拼出来的。

**代码生成的威力**：PyT 新增一个 elementwise 算子基本不用手写 kernel —— 写 YAML 声明，
生成器产出"向量化 `float4` 读写 + 循环展开 + 类型特化"的循环。**结果是所有 elementwise
算子都稳定在同一水平线上**，不会出现"这个算子快那个算子慢"。

⇒ 所以"PyT 内核优秀"的准确表述是：**它把 cuBLAS/CUTLASS/FlashAttention 的调优成果吸收进来，
再用代码生成保证平凡算子不拖后腿，最后用一个真实的 dispatch 层按硬件选最优后端。**

## 2. 我们与它的差距，逐条对得上

| 维度 | PyTorch | 我们 | 差距性质 |
| --- | --- | --- | --- |
| GEMM | cuBLAS | **也调 cuBLAS**（`cublasSgemmStridedBatched` + `op=T`）| ✅ 无差距（实测物化路径 3%）|
| elementwise | 代码生成，必然向量化 | 手写，**逐算子质量不均** | ⚠️ 结构性风险 |
| 注意力 | CUTLASS 流水 + 按架构选后端 | 自己写 tiling，Pascal 退化成滑窗重算 | ❌ 实打实的差距 |
| dispatch | 按 sm 版本选三条后端 | 有分档但**退化路径没有独立调优** | ❌ 差距 |
| autotune | cuBLAS 启发式 + 后端选择 | 有 `attn_tile` 等少数旋钮，**无按形状自动定档** | ❌ 差距 |
| 测量体系 | 每 PR 跑 kernel benchmark，对齐硬件上限 | **没有 kernel 级基准**，只有端到端步时 | ❌ **最要命的一条** |

最后一条是根因：**我们在"内核质量"上没有测量，就必然无法改进**。
端到端步时会被数据加载、Python 开销、分配器噪声掩盖，改一个内核 ±5% 根本看不出来。

## 3. 该拿什么换（按"机制"而不是"补丁"）

### 3.1 先建测量（本轮就做）

新增 `bench/kernels/bench_kernels.py`：**逐内核**测我们 vs torch vs 硬件上限。

* 测什么：注意力 fwd/bwd、softmax、rmsnorm、silu、GEMM（多种形状）、MoE gather/scatter。
* 算指标：达到 **峰值 FLOPS 的百分比** 或 **带宽 GB/s 的百分比**（按 cc 查表），
  而不是"多少毫秒" —— 只有对齐硬件上限，才知道还差多少。
* 判据：任何内核改动必须让对应行的百分比上升，且端到端不掉。

### 3.2 注意力：学 CUTLASS 的"小 smem 也能高效"，而不是"退化成滑窗"

具体到 P100 那条退化路径，值得做的（按收益排序）：

1. **两段式流水（双缓冲）**：把 K/V 的加载与 MMA 计算重叠；Pascal 没有 `cp.async`，
   但可以用 `__ldg` + 手动双缓冲 + 寄存器预取实现 2~3 级流水。
2. **只重算 softmax 统计量，不重算分数**：FlashAttention-2 的做法 —— 反向用保存的
   `logsumexp` 重算 `P = exp(S - lse)`，避免存/重算整块分数；我们现在是"滑窗 + 重算"，
   重算比例更高。
3. **寄存器分块替代大 smem**：把累加器完全放寄存器，smem 只放 K/V 分块 ⇒ 48 KB 也能上 BC=32。
4. **按架构给独立内核**：sm_60 一套、sm_70+ 一套，各自有自己的最佳 BC/DCHUNK，
   并**各自有实测表**（现在只有 sm_70+ 有）。

### 3.3 elementwise：从"手写"改成"生成"

把常用 elementwise（add/mul/silu/rmsnorm 等）统一到一个模板：`float4` 向量化读写、
`#pragma unroll`、`__launch_bounds__`、grid-stride loop。目标：**带宽达到 P100 峰值的 85%+**
（现在没有数据，这正是要测的）。

### 3.4 dispatch：把"退化路径"变成"第二实现"

PyT 的 mem-efficient 在 sm_50+ 都高效，是因为它本来就是为小 smem 设计的**一等实现**；
我们的滑窗版是"装不下就退化"。改法：**按架构准备两条都调优过的实现**，
用 `cc` 选择，而不是"主实现装不下 → 退化"。

### 3.5 autotune：按形状定档并缓存

像 cuBLAS 启发式那样：对 (S, Dh, dtype, cc) 组合，从离线表选 BC/DCHUNK/tile；
首次运行时可小范围实测并写入缓存文件。我们有 `attn_tile` 的基础，扩成完整表即可。

## 4. 验收标准（写死，避免自我感觉良好）

| 内核 | 指标 | 当前 | 目标（P100 fp32） | 目标（V100 fp32） |
| --- | --- | --- | --- | --- |
| 注意力 fwd | 峰值 FLOPS 占比 | 未测 | ≥ 45% | ≥ 50% |
| 注意力 bwd | 峰值 FLOPS 占比 | 未测 | ≥ 35% | ≥ 40% |
| softmax | 带宽占比 | 未测 | ≥ 85% | ≥ 85% |
| rmsnorm | 带宽占比 | 未测 | ≥ 85% | ≥ 85% |
| GEMM（[4096,1152]×[1152,1152]）| 峰值 FLOPS 占比 | 未测 | ≥ 70%（cuBLAS 上限）| ≥ 70% |
| MoE gather/scatter | 带宽占比 | 未测 | ≥ 80% | ≥ 80% |

> 这些数字不是拍脑袋：P100 上 cuBLAS SGEMM 通常能到 85~90% 峰值，
> 带宽类算子（softmax/rmsnorm/elementwise）成熟的实现在 80~90% 是常态；
> 注意力因为要搬运 + 重算，fwd 45%、bwd 35% 是"合格"线而不是"优秀"线。

## 5. 与"P100 值不值得"的关系

上一版文档我写"P100 不值得优化"，那句话在这里作废 —— **内核质量是要补的通用能力**，
P100 只是恰好把它暴露出来。正确的表述是：

* **优先级**：先建测量 → 再补 elementwise（通用收益）→ 再改注意力（收益最大、最难）。
* **P100 可以作为"低成本实验台"**：它的 48 KB 约束逼着我们写出真正紧凑的内核，
  这种内核在 V100/A100 上同样能用（smem 更宽裕只会更好）。
* **不追求在 P100 上"赢 PyT"** —— 追求**让每个内核达到硬件上限的合格线**，
  赢不赢是副产品。
