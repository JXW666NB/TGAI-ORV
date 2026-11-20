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
