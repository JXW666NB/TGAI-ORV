# 现代 MoE 架构对照与 ORV 的差距

> 调研日期：2026-11-11。事实来源以各模型技术报告为准（见文末）。
> 目的：明确 ORV 的 MoE 实现要补哪些能力，才能训"真实世界的 MoE"而不只是 V2.7。

## 1. 主流 MoE 的配置对照

| 模型 | 总参 | 激活/ token | 路由专家 | top-k | 共享专家 | 平衡方案 |
| --- | --- | --- | --- | --- | --- | --- |
| Switch Transformer (2021) | — | — | — | 1 | 无 | aux loss |
| GShard (2020) | — | — | — | 2 | 无 | aux loss |
| Mixtral 8x7B (2024) | 46.7B | 12.9B | 8 | 2 | 无 | aux loss |
| DeepSeek-V2 (2024) | 236B | 21B | 160 | 6 | 有 | aux loss |
| **DeepSeek-V3 (2024)** | **671B** | **37B** | **256** | **8** | **1** | **aux-loss-free（bias）** |
| Llama 4 Scout (2025) | — | — | 16 | 1 | 1 | — |
| Llama 4 Maverick (2025) | — | — | 128 | 1 | 1 | — |
| Qwen3-MoE (2025) | — | — | — | 2 | — | aux loss |

**要点**：
- top-k 的主流区间是 **1~8**，专家数从 8 一路到 **256**；「top-1」只在 Llama 4 出现且**同时带共享专家**。
- **共享专家是现代 MoE 的标配**（DeepSeek-V3、Llama 4 都有），作用是吸收通用知识，让路由专家专注特化。
- 平衡方案已经从 aux loss 转向 **aux-loss-free**（V3）：给每个专家一个 bias `b_i`，**只用于 top-k 选择，不进梯度**；门控权重仍用原始分数。

## 2. ORV 当前的差距（逐项）

| 能力 | ORV 现状 | 重要性 |
| --- | --- | --- |
| **共享专家** | ✅ **0.9.37 已支持**（`n_shared` / `shared_d_ff`） | 已完成 |
| **aux-loss-free 平衡** | ✅ **0.9.38 已支持**（`balance="aux_free"`） | 已完成 |
| **sigmoid 门控 + top-k 内重归一化** | ⚠️ 有 `sigmoid` / `softmax` / `softmax_topk` 三种，需核对与 V3 语义是否一致 | 中 |
| **Z-loss（router 数值稳定）** | ✅ **0.9.39 已支持**（`z_loss_coef`） | 已完成 |
| **容量因子 + 残差旁路** | ✅ **0.9.40 已支持**（`capacity_factor`）；**限制：CPU/CUDA 丢弃集合不同** | 已完成 |
| **细粒度专家（mN 小专家、mK 路由）** | ⚠️ 支持任意 `n_experts` / `top_k`，但没有"每专家更小"的配套约定 | 中 |
| **多专家下的分组 GEMM** | ❌ 每专家一次 GEMM；分组路径 `rc=7` 未通 | **高**：见下 || **Expert Choice 路由** | ❌ 无 | 低：自回归解码用不了 |

## 3. 一个重要的确认：多专家下"分组 GEMM"才是主要矛盾

调研材料里对教学版实现的注解，正好印证了我们此前的实测方向：

> 教学版用 Python `for e in range(num_experts)` 逐个调用专家，
> 但**生产实现（Megatron / vLLM）全部用 grouped GEMM 或融合内核**，
> 跨专家批量计算；否则 N 大时 GPU 大部分时间在发射 kernel。**这才是 MoE 真正难的一层。**

对照实测账：

| 配置 | 每专家行数（seq 512） | 每层 GEMM 次数 |
| --- | --- | --- |
| V2.7：top-2 / 8 专家 | 128 | 24 |
| top-8 / 256 专家 | 16 | 768 |
| top-32 / 256 专家 | 64 | 768 |

- V2.7 这一格，24 次 GEMM 的发射开销只占 4~8% ⇒ **分组收益小**（已实测）；
- 真实配置下 **768 次/层**，且每专家只有 16~64 行 ⇒ 发射开销与低效 GEMM 成为主要矛盾。

**结论：分组 GEMM 对 V2.7 不值得（已测），但对"真实 MoE"是必须的。**
此前因为 V2.7 的测量而搁置这项工作是判断失误，现在按真实配置重新定位。

## 3.1 分组 GEMM 的实测结论（2026-11-11，`tools/strided_moe_probe.py`）

`cublasSgemmStridedBatched` **符号存在且数值正确**（相对差 0.00e+00 ~ 1.36e-06），
但**含打包开销后在三种配置下全都更慢**：

| 配置 | 逐专家 | 打包权重 | strided 本身 | 含打包总耗时 | 相对 |
| --- | --- | --- | --- | --- | --- |
| V2.7（8 专家 × 128 行） | 0.988 ms | 0.401 ms | 0.886 ms | 1.287 ms | **慢 1.30×** |
| 64 专家 × 16 行 | 3.167 ms | 3.118 ms | 3.090 ms | 6.208 ms | **慢 1.96×** |
| 256 专家 × 8 行 | 8.215 ms | 12.346 ms | **5.914 ms** | 18.260 ms | **慢 2.22×** |

### 关键判读

1. **strided 本身在 256 专家下确实更快**（5.914 vs 8.215，**1.39×**）——
   说明"减少启动次数"这个方向本身是对的；
2. **但打包权重的开销把它完全吃掉**，且专家越多亏得越多；
3. 根因是**带宽**：G=256、M=8 时，每个专家算 8×1024×4096×2 = 67 MFLOP，
   却要读 4096×1024×4 = 16.8 MB 权重 ⇒ **4 FLOP/byte**。
   V100 的 15.7 TFLOP/s 与 900 GB/s 之比约 17 FLOP/byte，
   所以这个形状**内存受限**，任何需要重排权重的方案注定亏。

### 唯一可行的做法（未实施）

**把专家权重从一开始就存成连续的 `[G, N, K]`**，而不是 G 个独立 `Linear`。
这样 strided batched 直接用，**打包开销为零**。

按上表"strided 本身"一列估算，收益上限是：

- V2.7 配置：0.886 / 0.988 ⇒ **1.12×**
- 64 专家：3.090 / 3.167 ⇒ **1.02×**
- 256 专家：5.914 / 8.215 ⇒ **1.39×**

代价是**专家参数布局重构**（`state_dict` 兼容、权重加载需一次性重排）。
收益 ≤1.39× 且只有偏多专家的配置才明显 ⇒ **暂不实施**，除非明确要主攻 256 专家级模型。


## 4. 建议的补齐顺序

1. **共享专家**（`n_shared` 参数，始终激活、不参与路由）—— 纯前向加法，改动小、价值高；
2. **aux-loss-free bias**（`b_i` 只用于 top-k、不进梯度；门控用原始分数）—— 语义清晰，可单测；
3. **分组 GEMM**（多专家合并；这也是 CUDA 库该有的"硬功夫"）—— 需要先解决 `cublasSgemmGroupedBatched` 签名问题，或改 `SgemmStridedBatched`；
4. **Z-loss** + **容量因子/残差旁路**（数值稳定与工程完备性）；
5. **细粒度专家**的配套约定（每专家 `d_ff` 缩小、top-k 等比放大）。

## 5. 参考

- DeepSeek-V3 Technical Report, arXiv:2412.19437（256 路由 + 1 共享，top-8，aux-loss-free）
- DeepSeekMoE, arXiv:2401.06066（细粒度 + 共享专家）
- Auxiliary-Loss-Free Load Balancing, arXiv:2408.15664
- Mixtral of Experts, arXiv:2401.04088（top-2，46.7B 总参 / 12.9B 激活）
- Switch Transformer, arXiv:2101.03961（top-1 + aux loss + 容量因子）
- ST-MoE, arXiv:2202.08906（Z-loss）
- GShard, arXiv:2006.16668（top-2 + all-to-all）
- Qwen3 Technical Report, arXiv:2505.09388
- 系统性综述：[MoE 教程速查表](https://raw.githubusercontent.com/wanshuiyin/auto-claude-code-research-in-sleep/43a5daf6f8a2db7b87b48cb51d45c6ddb3c9a6d3/docs/tutorials/moe_tutorial_en.md)（对照表与工程坑位）
