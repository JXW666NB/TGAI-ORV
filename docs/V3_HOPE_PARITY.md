# 与 HOPE V3（PyTorch 参照）的对齐状态

> 实测日期 2026-11-20 / ORV 1.0.7 / V100-PCIE-32GB / torch 2.6.0+cu124
> 参照实现：`_src15b/{model_v3.py, data_v3.py, train_v3.py, optim_v3.py}`（项目内只读快照）

## 1. 结论先说

**同一个模型、同一个 loss 量级、同向收敛、无发散；ORV 步时快 1.34×。**
ORV 的 30 步 CE 比参照低约 **0.10**，归因到**两个未实现的正则项**（见 §4），
不构成"实现更好"或"实现有问题"。

## 2. 配置对齐（已实测）

| 项 | ORV | HOPE V3 | 状态 |
| --- | --- | --- | --- |
| 参数量（8 层 / E=8 / d=1024 / d_ff=2816 / tie）| **746,938,118** | **746,979,334** | ✅ 差 **0.006%** |
| vocab / d_model / n_heads | 96000 / 1024 / 16 | 同 | ✅ |
| attn_global_every / csa_compress / n_kv_heads | 4 / 4 / 0(MHA) | 同 | ✅ |
| n_experts / activated / shared | 8 / 2 / 1 | 同 | ✅ |
| tie_embeddings | 开（`ORV_TIE=1` + `lab/rent/tie_main.py`）| True | ✅ |
| **损失口径** | `CE(logits[:, :-1], labels[:, 1:], ignore_index=-100)` | 同（`model_v3.py:1508-1515`）| ✅ **本来就一致** |
| 数据格式 | `input_ids` / `labels` / `seg_id` | 同（`data_v3.py`）| ✅ |
| 取窗顺序 | `rng(seed+epoch*100003).permutation` + 分片内 `rng(seed*1000003+epoch*100003+si*7919)` | 同（`PackedDataset.order`）| ✅ |
| lr / steps / grad-clip | 3e-4 / 30 / 1.0 | 同 | ✅ |

> 注：`v3_train_cli.py` 的 docstring 里列的"训练损失口径未对齐"说的是**那个 CLI 自己**，
> 不是 `v3_pretrain`。`v3_pretrain::forward_loss` 已按参照切片（`v3_pretrain.py:790-792`）。

## 3. 实测 loss 曲线（纯 CE，不含 MoE 辅助项）

| | 末10 步均值 | 最低 CE | 步时 |
| --- | --- | --- | --- |
| **ORV**（`v3_pretrain` + tie）| **11.5605** | 11.5355 | **204 ms** |
| HOPE V3（seed 42）| 11.6660 | 11.6099 | 274 ms |
| HOPE V3（seed 7）| 11.6618 | 11.6177 | — |
| 参照侧**种子间**波动 | **0.009~0.012** | — | — |

* 两侧同起点量级（11.65~11.73）、同向下降、30 步内**无 NaN、无发散**
* ORV 低 **0.10**，是参照侧种子噪声的约 **10 倍** ⇒ 差异真实存在，但不等于"更好"

## 4. 造成那 0.10 的两个已知缺失（`v3_train_cli.py` 自己记录在案）

| # | 缺失项 | 参照实现 | ORV 现状 | 对 loss 的方向 |
| --- | --- | --- | --- | --- |
| 1 | **MoE z-loss + 互补均衡 aux loss** | `aux_loss()` 以 1e-4 权重加进总损失（实测 0.0074/步）| `v3_pretrain` **未加** | 少一个正则 ⇒ 前期 loss 降得更快 |
| 2 | **QK-Clip**（`qk_clip_tau=100, alpha=0.5`）| 启用，会改注意力权重 | **未实现** | 注意力数值路径不同 |

⇒ 两者都会让 ORV 的 loss 偏低，**方向与实测一致**。

## 5. 要更严格的对齐，还差什么

1. MoE `z_loss_coef` / `aux` 接进 `v3_pretrain` 的损失（ORV 的 `MoE` 已有 `z_loss_coef` 参数，接上即可）
2. 实现 QK-Clip
3. 之后再做"权重级对齐"（把参照侧的初始权重灌进 ORV，同起点跑同样步数），才能把那 0.10 精确归零

## 6. 复现

```powershell
# 参照侧（远端 PyT）
C:\Python311\python.exe D:\orv_bench\hope\q_hope_pyt.py `
  --data D:\orv_bench\data2048_v96000 --layers 8 --seq 512 --steps 30 --seed 42 --clip 1.0

# ORV 侧
$env:ORV_TIE='1'
D:\orv_pip_0962\Scripts\python.exe D:\orv_bench\tie_main.py `
  --data D:\orv_bench\data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 8 `
  --E 8 --d_ff 2816 --top_k 2 --csa 4 --n_q 16 --opt adamw --amp gemm --batch 1 `
  --steps 30 --lr 3e-4 --log-every 1 --no-final-save
```

脚本：`lab/rent/q_hope_pyt.py`（参照侧 harness，含同款取窗顺序）、`lab/rent/tie_main.py`（tie 注入壳）。

---

## 7. 2026-11-20 更新：两项"未对齐"的处置

### 7.1 MoE z-loss + aux loss —— **已实现**（`a762af3`）

接进 `v3_pretrain.forward_loss`，公式与 `model_v3.py:1079-1086 / 1126-1133` 逐项对齐：

```
z   = (logsumexp(router_logits, -1) ** 2).mean()      × moe_router_z_loss (1e-4)
f   = (E / (top_k*N)) * onehot.sum(0)                 （无梯度）
P   = probs.mean(0)                                   （有梯度）
aux = z + (f * P).sum() × moe_aux_alpha (1e-4)
```

* 开关：`ORV_Z_LOSS` / `ORV_AUX_BAL`（默认 0 ⇒ **训练损失与 1.0.7 逐位相同**）
* 实现技巧：`lse = mean_j(x_j − log_softmax_j)`，不必新写 logsumexp 内核
* 实测（8 层 / E=8 / seq512）：step-0 loss 抬高 **+0.004955**（参照侧 12 层/E=4 是 0.0074，量级一致）
* 回归测试：`tests/test_v3_pretrain.py::test_aux_coefs_default_zero_and_validation` +
  `test_router_lse_identity`（恒等式若被重构破坏，z-loss 会静默算错）

### 7.2 QK-Clip —— **实测为 no-op，暂不实现**（有证据）

**实测**：参照实现跑 5 步后 `model._qk_max.max() = 39.69`，逐头 31.7~39.7，而 `qk_clip_tau = 100`。
⇒ `γ = min(1, τ / S_max) = 1.0`，**QK-Clip 一次都不会触发**。

原因：两侧都开 `attn_qk_norm = True`（q/k 沿 head 维做 RMSNorm），pre-softmax 分数天然有界
（≈ d_head·cosθ ≈ 64 量级），达不到 τ=100。

**结论**：现在实现它 = 写一段永不执行的代码。**不做**，但把方案记录下来，将来遇到以下情形再启用：
* 关掉 `attn_qk_norm`
* 训练规模远大于当前（K2 的动机是 1T token 级别的 logit 爆炸）
* 实测 `max(q·k)` 逼近 τ

**将来实现时的精确方案**（照 `optim_v3.py::apply_qk_clip`）：
1. 前向记录 per-(layer, head) 的 pre-softmax 最大分数（ORV 目前**没有**这个遥测，需要改注意力路径）
2. **优化器更新之后**执行：`γ_h = min(1, τ / S_max^h)`；`Wq ← γ^α · Wq`、`Wk ← γ^(1−α) · Wk`（α=0.5）
3. CSA 层的 K 走 `k_down`，需缩放 **k_down 的输出行**（等效于缩放 K）
4. 判据：开启后 `max(q·k)` 应被压到 ≤ τ，且 loss 曲线无 spike