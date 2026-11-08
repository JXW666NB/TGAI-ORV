# V2.7（3.0B MoE）接入 ORV 的实测事实与缺口

> 目标：**单张 V100-32GB 上把 V2.7 训练起来（3B 全参）**。
> 本文只写**实测**，推断处明确标注。数据来源：`D:\TGAI\models\V2.7\` 的 `.TG` 头 + `last_step.pt`。

## 1. 机器实测（V100 机器）

| 项 | 实测值 | 来源 |
| --- | --- | --- |
| 显存 | 32768 MiB，V100-PCIE-32GB（sm_70，**无 bf16 TC**）| nvidia-smi |
| 主机内存 | **31.9 GB**；两个推理实例占 ~22 GB ⇒ 空闲 **5.9 GB** | GlobalMemoryStatusEx + 进程 WorkingSet |
| PCIe | **3.0 x16**（≈11 GB/s）| `nvidia-smi --query-gpu=pcie.link.gen.current,width` |
| 盘 | 致态 Ti600 1TB NVMe：**1.48 GB/s 冷读 / 3.01 GB/s 缓存** | 本文实测 4 GB 顺序读 |
| fp16 GEMM | 4096³ **78.2 TFLOPS**（= 125 峰值的 62.6%，**证明用了 Tensor Core**）| 本仓库 `gemmbench.py` |
| fp32 GEMM | 4096³ **11.9 TFLOPS**（= 15.7 峰值的 75.8%）| 同上 |

## 2. V2.7 真实结构

`.TG` 头：`magic=49414754 v3 weight_dtype=6(BF16)` `L=28 D=1024 NH=16 FF=4096 V=32779 max_seq=8192`
`experts=8 activated=2` `arch=0` `num_kv_heads=0`；982 张量（869×bf16 + 113×fp32）。

`last_step.pt`（983 张量）+ `model_config`：
`n_experts=8 n_activated=2 n_shared_experts=0 moe_balance="both" rope_theta=10000.0 dropout=0.1`
（另有 `global_step=3804 loss=1.2566 best_ppl=3.3923`）

### 张量命名（实测，前几层样例）

```
token_embedding.weight                  (32779, 1024)
blocks.N.ln1.weight / ln2.weight        (1024,)          RMSNorm
blocks.N.attn.q_proj/k_proj/v_proj      (1024, 1024)     ← k/v 与 q 同宽 ⇒ 无 GQA，head_dim=64
blocks.N.attn.out_proj.weight           (1024, 1024)
blocks.N.attn.rope.cos / rope.sin       (1,1,8192,64)    ← 预算好的 RoPE 表
blocks.N.moe.router.weight              (8, 1024)        ← 8 专家打分
blocks.N.moe._qb_bias                   (8,)             ← 负载均衡旁路偏置
blocks.N.moe.experts.E.w1.weight        (4096, 1024)     ← SwiGLU gate
blocks.N.moe.experts.E.w2.weight        (4096, 1024)     ← SwiGLU up
blocks.N.moe.experts.E.w3.weight        (1024, 4096)     ← SwiGLU down
```
moe/experts 相关张量 **756 个**（≈ 28 层 × 8 专家 × 3 + router/qb）。

## 3. ORV 现有能力 vs 需要（缺口清单）

| 项 | ORV 现状 | V2.7 需要 | 缺口 |
| --- | --- | --- | --- |
| `.TG` bf16 读 | ✅ `read_tg`（dtype 6，`4dd058b`）| bf16 | 无 |
| `.TG` bf16 写 | ✅ `pack.TGWriter`（`DT_BF16=6`）| 存回产线 | 无 |
| 大文件读取 | ✅ mmap 流式（实测 5.64 GB 解析 **0.02 s**，RSS≈0）| — | 无 |
| 权重命名 | `Qwen25` 用 `blocks.N.attn.qkv.weight` 等 | 分离 q/k/v + `moe.experts.E.w1/w2/w3` | **需映射层** |
| MoE 路由 | `nn.MoE` **top_k=1**（argmax+sigmoid）| **top-2**，`moe_balance="both"` | **需实现 top-2** |
| 负载均衡 | 无 `_qb_bias` 概念 | 有 `_qb_bias`(8,) | 需定义语义（读/写/忽略） |
| GQA | `qwen.py` 支持 KV 头 | V2.7 **无**（k/v 同宽）| 无 |
| RoPE | 在线计算 | 文件里有预算表 | 可忽略文件里的表，按 θ=1e4 在线算 |

## 4. 全参训练的内存方程（32 GB 显存 + 31.9 GB 内存）

3.0B 的 fp32 状态 = 权重 12 + 梯度 12 + AdamW 两动量 24 = **48 GB**，单卡 32 GB 装不下。

| 方案 | 显存 | 内存 | 每步 PCIe | 结论 |
| --- | --- | --- | --- | --- |
| **A. fp32 master 驻内存 + int8 动量** | fp16 权重 6 + fp16 梯度 6 + int8 动量 6 + 激活 ~8 = **26 GB** | 12 GB | 18 GB ≈ **1.6 s/步** | ✅ 唯一可行 |
| B. 动量也放内存 | 20 GB | 36 GB | 2.2 s | ❌ 内存不够 |
| C. LoRA（非全参）| ~10 GB | 小 | 无 | ✅ 可与推理共存 |
| D. 状态放 SSD | — | — | 48 GB ≈ 32 s | ❌ 盘 1.5 GB/s 太慢 |

**前提**：训练时必须**停掉两个推理实例**（释放 27 GB 显存 + 22 GB 内存）。

## 5. 验收口径（必须实测，不接受"看起来对"）

1. **前向对齐**：用 `last_step.pt` 在 PyTorch 里写一个等价参考实现（MoE top-2 + SwiGLU + RMSNorm + RoPE），
   与 ORV 的 V2.7 前向比 logits：**相对误差 ≤1e-4**（fp32）。
2. **`.TG` ↔ `.pt` 往返**：`last_step.pt` → `.TG` → ORV 加载 → 前向，与直接从 `.pt` 前向一致。
3. **一步训练**：loss 有限、梯度范数合理、参数有更新；再跑 20 步看 loss 下降。
4. **全参**：优化器状态完整（权重+梯度+两动量），除 int8 量化外无近似。

## 6. 分阶段（对应 goal）

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| P0 | `.TG` bf16 读写 + mmap 流式 | ✅ 完成（`4dd058b`、`d93e0c9`）|
| P1 | 命名映射 + **MoE top-2** + 前向跑通 + 与 torch 参考对齐 | ⬜ 进行中 |
| P2 | int8 动量 + fp32 master 驻内存 + PCIe 重叠 + 梯度检查点 | ⬜ |
| P3 | V100 上 SFT（全参，~2 s/步）| ⬜ |
## 7. 名字旁挂（2026-11-08 落地）

`.TG` 权重表**只存 FNV-1a 哈希、不存名字** ⇒ 当检查点用时"无法枚举参数"。
方案：新增**纯旁挂**文件 `<model>.TG.names.json`（产线 C 引擎不读它 ⇒ 推理零风险）。

* 工具：`tools/make_tg_names.py --tg <TG> --pt <pt>|--keys-from <txt>`（按 FNV 哈希**精确匹配**，绝不按形状猜名）
* 写入：`export.save_tg()` 现在会顺手生成旁挂；失败也不影响 `.TG` 可用
* 读取：`read_tg.list_tensors()` 优先用旁挂；**没有旁挂时名字返回 None，绝不编造**
* 一致性：`.TG` 大小与旁挂记录不符时 `load_names()` **直接报错**（防止改名错位后静默训错）

**实测（真实 V2.7）**：源头 983 个名字，`.TG` 982 个表项，**匹配 982/982 = 100%**，回读 982/982 有名字。

### 顺带查出的架构事实

`last_step.pt` 里有 `lm_head.weight`，而 `.TG` 里**没有** ⇒ **产线 `.TG` 是词嵌入绑定（tied embedding）**。
因此 ORV 侧建模 V2.7 必须用 `TIE=True`，否则会多/少一份输出层权重、logits 对不上。
## 8. 路由语义（以 C 引擎源码为准，2026-11-08 查证）

来源：`D:\TGAI\01_源码\TGAI-GO-wt\cuda\tgai_cuda.cu` 第 151~164 行的字段注释
（引擎结构体 `d_router / d_qb_bias / d_expert_w1..w3`）：

| 项 | 引擎语义 |
| --- | --- |
| `_qb_bias` | **形状 `[n_experts]`(F32)，必须在 top-k 之前加到 router logits 上**。缺失时为 NULL（老模型 V2.5/V2.6 行为不变）|
| 不读它的后果 | 源码原话："旧实现完全没读这个张量 ⇒ **top-k 选错专家 ⇒ 输出乱码/退化**"（2026-10-13 修掉的乱码根因）|
| 专家权重布局 | `d_expert_w1/w2/w3` = `[L, E, d_ff, d_model]` / `[L,E,d_ff,d_model]` / `[L,E,d_model,d_ff]` ⇒ 与 `.pt` 的 `w1(ff,d) / w2(ff,d) / w3(d,ff)` 一致 |
| LM head | `d_emb_t` = `[d_model, vocab]`（由 `d_emb` 转置而来）⇒ **确证 tied embedding** |

⇒ ORV 侧建模 V2.7 必须：
1. `TIE=True`（无独立 lm_head）；
2. 路由：`z = x @ router.T; z += qb_bias; topk(z, k=2)`（**偏置只影响"选谁"，语义按上表**）；
3. 专家：`w1→gate`、`w2→up`、`w3→down`，SwiGLU。

## 9. ORV 已有能力复核（避免重复造）

`nn.MoE(d_model, d_ff, n_experts, top_k=2, gate="softmax")` —— **构造默认就是 top_k=2**，
且前向支持"按 token 分组、空专家不参与 GEMM"。⇒ P1 里"实现 top-2 路由"**不需要**从零写，
只需：① 把 `qb_bias` 加在 top-k 之前；② 权重名映射。
## 10. 路由语义（C 引擎源码权威注释，2026-11-08 二次确认）

`D:\TGAI\01_源码\TGAI-GO-wt\src\tgai_engine.c:1117-1125`：

```
/* ★ QB 路由偏置 (2026-10-09 加，TGAI 2.7 必需)
 * TGAI V2.7 用 quantile balancing（model_config.moe_balance='both'），
 * 每层有 blocks.{i}.moe._qb_bias [n_experts]。语义：router logits **先加本偏置**，
 * **再做 top-k 选择（softmax 之前）**。
 * 向后兼容：老模型（V1/V2/V2.5/V2.6/mini）没有这个权重 ⇒ 保持 NULL ⇒ 行为不变。 */
```

⇒ ORV 侧 V2.7 前向的 MoE 部分确定为：

```
z = x @ router.T
z = z + qb_bias          # 只在有该张量时（V2.7 有）
ids, w = topk(z, k=2)    # 大值优先，平局取小下标（与 nn.MoE 文档一致）
w = softmax(w)           # 源码明确 "softmax 之前" 才加偏置 ⇒ 门控是 softmax
out = Σ_i w_i * Expert_id_i(x)      # Expert: silu(x@gate.T)*(x@up.T) @ down.T
```

**仍需实测确认的一点**（下一轮用 logits 比对判定）：softmax 是在 **top-k 选中的 2 个** logits 上做，
还是在**全部 8 个**上做后再取前 2。两种都自洽，必须用数值比对定夺，不靠猜。