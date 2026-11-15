# V3 → ORV 移植：真实状态与剩余工作（2026-11-16 核对）

> 旧的 `docs/V3_ORV_FIRST_TRAIN_STEP.md` §3 是 **11-13** 写的，之后代码又推进了。
> 照那份清单去"补"会重复造已有的东西。以本文为准。

## 一、已经有（实测核对过源码，不是回忆）

| 能力 | 位置 | 状态 |
| --- | --- | --- |
| SiTU-GLU（融合内核） | `cuda.py:525 situ_glu_fwd/bwd`、`v3_ops.py:51/461/496` | ✅ |
| RoPE（half / fused，含 float4） | `cuda.py:2169/2218/2521`、`v3_ops.py:98/113` | ✅ |
| MoE top_k=2 + 偏置队列 | `v3_moe.py:81 route_topk`、`:127 moe_topk_orv` | ✅（注释称严格对齐 model_v3.py:1039-1057）|
| CSA（压缩 + 因果卷积） | `v3_ops.py:258 csa_compress`、`:313 csa_conv_causal` | ✅ |
| scatter_add（专家散回） | `cuda.py` 的 `scatter_add_rows`、`v3_scatter.py:36` | ✅ |
| **index_add（1-D / 2-D）** | `v3_scatter.py`（2026-11-16 新增，复用 scatter 内核）| ✅ |
| 共享专家 | `__init__.py:4487+` 的通用 `T.nn.MoE(n_shared=...)` | ⚠️ 通用模块有，V3 路径未接线 |
| topk（CPU） | `__init__.py:2561` | ⚠️ **CUDA 上抛 NotImplementedError** |
| einsum（CPU） | `__init__.py:2707` | ⚠️ **CUDA 上抛 NotImplementedError** |

## 二、真缺口

### 1. BlockAttnRes（唯一完全缺失）

`model_v3.py:1227-1282` 的语义（逐行照搬自论文伪代码）：

```
V = stack(blocks + [partial_block])        # [N+1, B, T, D]
K = norm(V)                                 # 逐块 RMSNorm，等价
w = proj.weight.reshape(-1)                 # (D,)   ← 伪查询，不是 Linear 输出
logits = einsum('d,nbtd->nbt', w, K)        # [N+1, B, T]
h = einsum('nbt,nbtd->btd', softmax(logits, 0), V)   # [B, T, D]
```

**不能照抄**：`einsum` 与 `stack` 在 CUDA 上都没有。用现有 GPU 算子等价改写：

| 原式 | 改写 | 用到的已有算子 |
| --- | --- | --- |
| `einsum('d,nbtd->nbt', w, K)` | 对每个 i：`(K_i.reshape(-1, D) @ w.reshape(D, 1)).reshape(B, T)` | `@`（2-D matmul）|
| `stack` + `softmax(0)` | 不需要 stack：对 list 逐元素做 `max` → `exp` → `sum` → 除 | `T.maximum`、`exp`、`+`、`/`（全逐元素）|
| `einsum('nbt,nbtd->btd', p, V)` | `Σ_i p_i.reshape(B,T,1) * V_i` | 广播乘 + `+` |
| `norm(V)` | 逐块 `rms_norm`（数学等价，且省 fp32 副本）| `nn_ops.rms_norm` |

这样全部落在已有 GPU 算子上，**不新造内核、不产生主机往返**。

### 2. 共享专家接线（条件性）

取决于 1.5B 配置里 `n_shared_experts` 是否 > 0（`v27.py:7` 记的是 0，V3 1.5B 需查配置）。

### 3. topk / einsum 的 CUDA 版（非阻塞）

目前都可绕：
- topk → 路由在主机侧用 numpy（`route_topk` 已经这么做，`scatter_add` 的 ids 本来就是 numpy）
- einsum → 用上面的 matmul/逐元素改写

## 三、完成判据（沿用目标）

V100 上跑起 V3（GlobalAttention + CSA + MoE top_k=2 + BlockAttnRes）完整训练步，给出：
① 与 PyTorch 同配置的前向数值误差；② 峰值显存对比；③ s/step 与 tok/s 对比。三条都要实测数字。
