# ORV vs PyTorch 审查（2026-11-10）

> 原则：**每个数字都标实测来源**；没测过的写"未验证"，不外推。
> 目标：让 ORV（PyPI `tgai-orv`）能替代 PyTorch 做**单卡自定义模型训练**；TGAI V2.7 只是验收样例。

## 0. 一句话结论

- **能力面**：训练闭环（前向/反向/优化器/调度/裁剪/累积/DataLoader/AMP/checkpoint/权重互转）齐，
  0.9.14 补齐 Dropout / 池化 / BatchNorm / `Module.train()|eval()`。
- **质量**：多数路径**逐位一致（0.00e+00）**，其余 ≤1e-7~1e-5。
- **显存**：3B MoE 全参 + **真 AdamW** 在 32 GB 单卡可跑（≈25 GB）；PyTorch 同配置需 ≈48 GB，
  实测 31.34 GB 处 OOM。**这是 ORV 的核心卖点**。
- **速度**：PyTorch 纯 fwd/bwd 仍快约 1.6–2×（T=512 口径）；ORV 通过融合优化器 + embedding 设备化 +
  逐层图反向把差距大幅缩小（T=128 整步 1843 → **721 ms**，逐位一致）。**T=512 的图反向尚未打通**。

## 1. 训练模式支持

| 模式 | 状态 | 依据 |
| --- | --- | --- |
| 预训练 | ✅ | `tests/test_api_pretrain.py::test_pretrain_tiny_end_to_end_runs`；3B MoE 单卡整步实测 |
| SFT | ✅ 数据+训练 | `sft_data`（JSONL→打包）、`tests/test_qwen_sft_smoke.py` |
| DPO | ✅ 可训 | `DPOTrainer`（0.9.16）：GPU 20 步 loss 0.688→**0.213**、margin 0.0995→**14.398**；数学件 `align.dpo_pair_loss` 与 torch 对照通过 |
| GRPO | ✅ 可训 | `GRPOTrainer`（0.9.16）：GPU 一步 loss 有限、裁剪范数 0.367；`grpo_loss` 原先在 CUDA 上出 nan（常量未对齐设备 + 用 `.data` 统计），0.9.16 已修 |
| RLHF / PPO | ❌ 未实现 | 无 value model、无 PPO trainer（DPO/GRPO 属偏好优化，不是完整 RLHF） |

## 2. 质量（与 torch 同类实现对照）

| 项 | 相对差 | 来源 |
| --- | --- | --- |
| gather / scatter（前+反向） | **0.00e+00** | `tests/test_gather*` |
| CUDA Graph 逐层反向（loss / 梯度合计） | **0.00e+00** | `tools/graph_whole_step.py` |
| 融合 fp16-AdamW vs 逐参数 | **逐位** | `tests/test_fused_adamw*` |
| embedding 设备化 vs numpy 路径 | **loss 0.00e+00** | `tools/full_step_embed.py` |
| LayerNorm（CUDA 前/反向） | 4.38e-08 / 9.29e-07 | `tools/layernorm_gpu_check.py` |
| conv2d（CUDA，fwd/dx/dw/db，6 组配置） | **2.39e-07** | `tools/conv_cuda_parity.py` |
| conv1d（CUDA，3 组配置） | 1.26e-07 | `tools/conv_cuda_extra.py` |
| MaxPool2d（CUDA vs CPU / vs torch / 反向） | **0.00e+00** | `tools/layers_cuda_check.py` |
| AvgPool2d（CUDA vs torch / 反向） | 4.76e-08 / 9.69e-08 | 同上 |
| BatchNorm2d（CUDA vs CPU，含 running 统计） | **0.00e+00** | 同上 |
| torch 权重互转（logits / 往返） | ≤1e-5 / **逐位** | `tests/test_interop.py` |
| LR 调度 | 4.55e-15 | `tests/test_sched*` |
| DataLoader 批次内容与顺序 | 逐元素一致 | `tests/test_loader*` |

## 3. 速度与显存（V100-32GB，单卡）

### 3.1 加速做了什么（都有实测）

| 优化 | 实测 |
| --- | --- |
| 融合 fp16-AdamW（原 2550 个张量逐个更新，≈0.98 s/步） | 0.375 → **0.047 s（8×）**，与逐参数**逐位相同** |
| embedding 设备化（省掉 128 MB/步主机往返） | T=128 −160 ms；**T=512：1.359 → 1.198 s（−162 ms）**，loss 0.00e+00 |
| 逐层 CUDA Graph 反向 | T=128 整步 1843 → **721 ms（2.56×）**，loss/梯度 0.00e+00 |
| 逐层图（28 层 attention 单测） | 1016.3 → **140.5 ms（7.24×）**，梯度 0.00e+00 |
| 融合投影路线（一次全宽 GEMM + 分头切片） | 等价但**无提速**（7.81 ms → 7.81 ms）⇒ **已放弃** |

### 3.2 与 PyTorch 的差距（未解决部分，如实）

| 项 | 数值 |
| --- | --- |
| PyTorch fwd / bwd（T=512，同结构） | 0.438–0.586 s / 0.471–0.536 s |
| ORV fwd / bwd（T=512，加速前表） | 0.53 s / 1.16 s（逐层） |
| 根因 | ① 逐层重算（梯度检查点在 **Python 层**，非 C 端）；② 分头小 kernel（q/k/v 各 16 次） |
| T=512 图反向 | 28 张图**捕获成功**（每图 459 MiB），单独驱动时 28 层**逐位一致 0.00e+00**；但**接进整步**（捕获后再跑含 MoE 的前向链）触发 `cuMemcpyDtoHAsync rc=700`，**未根因定位** |

### 3.3 显存（核心卖点）

| 方案 | 真 AdamW 需求 | 32 GB 可跑？ |
| --- | --- | --- |
| PyTorch 3B fp32 | 权重 12.1 + 梯度 12.1 + 两个 fp32 状态 24.2 ≈ **48 GB** | ❌ 实测 31.34 GB 处 OOM，只能退 Adafactor |
| **ORV（fp16 动量）** | ≈ **25 GB** | ✅ 可跑（实测） |

## 4. 与 PyTorch 的能力缺口（逐项）

| 缺口 | 状态 |
| --- | --- |
| Dropout / MaxPool / AvgPool / BatchNorm（1d/2d） | ✅ **0.9.14 已补**，torch 对照通过 |
| `Module.train()` / `eval()` | ✅ **0.9.14 已补**（递归到子模块） |
| `Tensor.max/min/var/std/norm` | ✅ **0.9.15 已补**（max/min 用 one-hot mask，梯度只落到 argmax）|
| CUDA 上的 `abs/relu/clamp/log` | ✅ **0.9.16 沉到 C 端**（新增 7 个内核），`_cpu_only` 调用点 **4 → 0** |
| `Tensor.to/cuda/cpu`、`minimum` | ✅ **0.9.16 已补**（`maximum` 的 CUDA 分支也修了）|
| 高层 DPO / GRPO 训练器 | ✅ **0.9.16 已补**（GPU smoke 通过）|
| CUDA `conv` `groups>1` | ⛔ 未实现（明确抛 `NotImplementedError` 并给修法，不静默算错） |
| PPO / RLHF | ⛔ 未实现 |
| 多卡（DDP/FSDP/流水线） | 不在本阶段范围 |
| `torch.compile` 类图优化 | 未做（CUDA Graph 覆盖逐层反向，非通用编译） |

## 5. 复跑入口

### 5.1 已评估但**不启用**的优化（有实测，避免重复投入）

| 项 | 实测 | 结论 |
| --- | --- | --- |
| 融合投影（一次全宽 GEMM + 分头切片） | 等价（5.55e-07 / 梯度 9.14e-07），发射 339→302（−10.9%），但块时间 **7.81 → 7.81 ms** | **放弃**：无墙钟收益 |
| 低显存 SDPA（`functional.sdpa_lowmem`，不物化 P） | 与物化版**逐位一致**（前向与 dq/dk/dv 均 0.00e+00）；时间 **3.91 → 4.74 ms（慢 1.21×）**；净显存无差异 | **不默认启用**：慢；显存收益未测出 |

```powershell
# 质量对照
python tools\conv_cuda_parity.py          # conv2d CUDA vs torch
python tools\conv_cuda_extra.py           # conv1d / groups 拒绝 / 模块类
python tools\layers_cuda_check.py         # 池化 / BatchNorm / Dropout（CUDA）
python tools\graph_whole_step.py          # 图反向 loss/梯度 0.00e+00 + 整步提速
# 速度
python tools\full_step_512.py             # T=512 老口径（1.198 s）
python tools\graph_whole_step_512.py      # T=512 图反向（当前 rc=700，未通）
```

## 6. 下一步（按价值排序）

1. 修 T=512 图重放 `rc=700`（打通后 T=512 也能吃到图加速）
2. 补 `Tensor.max/min/var`（PyT 面 + 让池化/BN 不必绕路）
3. CUDA `groups>1` conv
4. 高层 DPO/GRPO trainer（loss 已有，只差训练循环 + 对照测试）
5. 目标：T=512 整步 ≤ 0.911 s（PyT Adafactor 步），显存仍 ≤32 GB
