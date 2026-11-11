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
| **aux-loss-free 平衡** | ❌ 无（有 `qb_bias` 做分位平衡，语义不同） | **高**：V3 的核心创新 |
| **sigmoid 门控 + top-k 内重归一化** | ⚠️ 有 `sigmoid` / `softmax` / `softmax_topk` 三种，需核对与 V3 语义是否一致 | 中 |
| **Z-loss（router 数值稳定）** | ❌ 无 | 中：fp16 下 router logit 易溢出 |
| **容量因子 + 残差旁路（token 丢弃）** | ❌ 无（当前是"多少 token 就多少行"，不设上限） | 中：多卡/静态缓冲才必须；单卡可不用 |
| **细粒度专家（mN 小专家、mK 路由）** | ⚠️ 支持任意 `n_experts` / `top_k`，但没有"每专家更小"的配套约定 | 中 |
| **多专家下的分组 GEMM** | ❌ 每专家一次 GEMM；分组路径 `rc=7` 未通 | **高**：见下 |
| **Expert Choice 路由** | ❌ 无 | 低：自回归解码用不了 |

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
