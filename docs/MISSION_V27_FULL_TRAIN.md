# 任务书：在 V100(32GB) 上用 ORV **全量训练 V2.7（2.97B MoE）**

> 用户设定（2026-11-08）：这是 ORV 第一次在产线展示身手的机会。过关 = ORV 真正具备训练能力。

## 验收星级（用户原话整理）

| 星 | 判据 |
| --- | --- |
| **★1** | 能做 **LoRA 微调**，效果/速度与 PyTorch 齐平 |
| **★2** | 能做**全参训练**：预训练、SFT、DPO |
| **★3** | **不用主机内存**，把**显存压到 32 GB 以内**完成上述训练 |
| 终点 | 在 V100 上用 ORV 完成**强化训练（DPO / GRPO）** |

## 显存账（2.97B，fp32 主权重方案）

| 组成 | 字节/参数 | 合计 |
| --- | --- | --- |
| fp32 master 权重 | 4 | 11.9 GB |
| fp16 梯度 | 2 | 5.9 GB |
| **int8 分块 AdamW m/v** | 2 | 5.9 GB |
| **小计（状态）** | 8 | **23.8 GB** |
| 激活（梯度检查点，B=1/S=512） | — | 2~4 GB |
| **总计** | — | **≈26~28 GB ⇒ 可进 32 GB** |

若改用 fp16 master（4→2 B）可再省 5.9 GB（**24~26 GB** 起步），代价是更新精度需误差反馈。

⇒ **★3 在数学上可行**，关键是 ORV 必须实现：int8 分块动量 + fp32/fp16 master + 梯度检查点 + MoE 稀疏更新。**不需要** CPU offload。

## 已验证的事实（都有实测证据，别再推翻重做）

| 项 | 结论 | 证据 |
| --- | --- | --- |
| V2.7 结构 | 2.97B MoE，L28 D1024 NH16 DH64 FF4096 V32779，**8 专家激活 2**，bf16，**TIE=True** | `.TG` 头 + `model_config` |
| **真源权重** | `D:\TGAI\models\V2.7\tgai_v27_v5_merged.pt`（**不是** `last_step.pt`）| 位模式/982 张量逐位 0 差异 |
| 路由语义 | `z=x@router.T + qb_bias` → `topk(2)` → **softmax 只在选中的 2 个上** | `tgai_engine.c:1120`、`tgai_cuda.cu:301-307` |
| RoPE | **interleaved**（相邻维度对共享频率，j=0 频率=1.0）| 预算表实测；ORV 已有 `_rope_composed` |
| 张量命名 | `attn.q/k/v/out_proj`、`moe.router`、`moe._qb_bias`、`experts.E.w1/w2/w3`→gate/up/down | 旁挂 982/982 |
| **ORV MoE 对齐** | 真实第 0 层权重：`max|Δ|=6.1e-05`、`rel=1.19e-07` **PASS** | `D:\tgai_orv\p2.log` |
| 参考前向 | CPU 跑通 8 token，logits 已存 `p1_logits_topk.npy` | `p1.log` |
| `.TG` V4 | 读写+名字表+流式；983/983 逐位；CI success | 提交 `0d693a0`/`6624862` |
| 直连 | `ssh Administrator@192.168.0.103 '单条命令'`（单引号包住）| 已配 key |

## 实施顺序

| 步 | 内容 | 判据 |
| --- | --- | --- |
| **P1 收尾** | 用 `_rope_composed` 拼 `V27Model`（28 层）→ 与 `p1_logits_topk.npy` 比 logits | rel ≤1e-4 |
| **P2a** | **int8 分块 AdamW**（block=64/128，per-block scale） | 与 fp32 AdamW 100 步参数偏差 ≤1e-3；显存实测 |
| **P2b** | **梯度检查点**（MoE 层按需重算） | 激活显存 ↓≥50%，step 时间 ↑≤40% |
| **P2c** | **★1 LoRA**（attn q/k/v/out + router）| 与 PyTorch LoRA 同配置 loss 曲线对齐 |
| **P2d** | **★2/★3 全参**：master(4B)+grad(2B)+int8 m/v(2B)，B=1/S=512 | `nvidia-smi` 峰值 <32 GB 且 **host RSS 不涨** |
| **P3** | SFT 冒烟：100 步 loss 下降、显存稳定 | 无 OOM/无 NaN |
| **P4** | **DPO**（先实现）→ **GRPO** | 偏好对训练 loss 下降；窄任务奖励上升 |

## 纪律

不瞎猜；实测优先；改前备份；动产线前授权；**不许放松判据**；每次提交前本地 `pytest` 全绿（这条我本轮违反过一次，已记录）。