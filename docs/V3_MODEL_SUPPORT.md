# ORV 对真实 V3 的支持度（2026-11-13 核对，权威清单）

> 本文只回答一个问题：**ORV 现在能不能训练真实的 TGAI V3 1.5B？**
> 依据是 `D:\TGAI-V3-1.5B\03_src\model_v3.py` 的**默认值与真实预设置**
> （`add_savev3_preset.py`），不是推测。

## 1. 真实模型的配置（源码实测）

| 参数 | 默认值（`model_v3.py`）| 真实预设置（`add_savev3_preset.py:52-60`）|
| --- | --- | --- |
| `n_experts` | 8 | **16** |
| `n_activated`（top_k）| **2** | **2** |
| `n_shared_experts` | 1（无门控、直接相加）| 1 |
| `attn_res_enabled` | **True** | **True** |
| `attn_res_block` | 4 | **2**（16 层 = 8 个 block，对齐论文 N≈8）|
| `use_csa` | 存在（属性，训练脚本有 `CSA插桩` 日志）| 同 |

## 2. ORV 的支持度

| 组件 | ORV 状态 | 证据 |
| --- | --- | --- |
| 全局注意力（GQA + 部分 RoPE + 掩码）| ✅ 已完成 | 单层对齐 <1e-4；块化版本前向 1.2e-07、反向 1e-9 |
| 块化注意力（不物化 S×S）| ✅ 已完成 | `v3_flash_op.flash_attn`，512/512 块 |
| MoE 路由 top_k=1 | ✅ 已完成 | `route_top1`，前向 5.07e-07 |
| **MoE 路由 top_k=2** | ❌ **未实现** | CUDA 无 `topk`；需按分数取前二并归一 |
| 共享专家 | ✅ 今天补齐 | `build()` 增加 `S1/S2/S3`，`--no-shared` 可关 |
| SiTU-GLU | ✅ 融合内核 | 单算子峰值 756 → 228 MiB |
| RMSNorm | ✅ 融合内核 | 每层留存 12 → 4 MiB |
| **CSA（压缩稀疏注意力）** | ❌ **未实现** | 压缩池化 + 因果深度卷积都还没有 |
| **BlockAttnRes（跨 block 残差）** | ❌ **未实现** | 真实模型 16 层 = 8 个 block，每 block 一条残差通路 |

## 3. 结论（必须随任何"ORV 能训 V3"的说法一起给出）

**ORV 目前能训的是 V3 的骨架，不是完整的 V3。** 具体地：

- 已能跑通：嵌入 → (全局注意力 + MoE top1 + 共享专家) ×N → 最终 norm → 头 → 交叉熵，
  真实训练步（AdamW、权重真更新）、梯度检查点、块化注意力，全部实测通过。
- 缺失：**top_k=2、CSA、BlockAttnRes** 三项。其中 BlockAttnRes 与 CSA 会改变
  显存与计算的结构（前者增加跨层激活，后者改变注意力复杂度）。

### 对本仓库那张对比表的限定

`docs/V3_FAIR_CKPT_COMPARISON.md` 的表是 **ORV(top_k=1) vs PyT(top_k=1)**：
两侧自洽、同口径、同为真实训练步，但**都不是真实 V3**（真实为 top_k=2 + BlockAttnRes + CSA）。
该表回答的是"**两个引擎在相同模型上的差异**"，不回答"ORV 能否训真实 V3"。

## 4. 补齐顺序建议（按显存收益/风险排序）

1. **top_k=2**（改动小、必须做）：在 numpy 侧对 `s_all` 取前二 + 归一化 gate，
   分派时把两个专家的输出按各自 gate 加权求和。验收：与 numpy 参考 <1e-5。
2. **BlockAttnRes**：按 `attn_res_block` 分组，每组维护一条跨层残差；
   注意它会**增加**常驻激活，需要重新测检查点下的峰值。
3. **CSA**：压缩池化（`latent` 维度）+ 因果深度卷积，工作量最大，最后做。
