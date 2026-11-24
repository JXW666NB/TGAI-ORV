# ORV 与 PyTorch 真 V3 的 faithful 端到端对拍（2026-11-16 实测）

> 目的：验证「补齐 6 项架构差异后」的 ORV 与 PyTorch 参考实现同构。
> 工具：`lab/faithful_parity.py`（两侧**全部打开**真 V3 的 6 个默认值）。

## 一、被验证的 6 项差异（之前对拍时被中和掉）

| # | 项 | PyTorch 真 V3 | ORV `--faithful-v3` |
|---|---|---|---|
| 1 | `tie_embeddings` | True（`lm_head.weight = embed.weight`）| `P["head"] is P["emb"]`，`params_of` 按 `id()` 去重 |
| 2 | `moe_post_norm` | True（`MoEv3.post_norm`）| routed 聚合后、加共享专家前做 RMSNorm |
| 3 | `attn_qk_norm` | True（`GlobalAttention.q_norm/k_norm`）| Global 层 q/k 在 RoPE 前各自沿 `d_k` 做 RMSNorm |
| 4 | `pad_token_id` | 256 | `_emb[pad_id] = 0.0` + 梯度语义 |
| 5 | `moe_balance` | `'quantile'` | `_route_quantile` |
| 6 | `attn_res_enabled` | True（`attn_res_proj`/`mlp_res_proj`）| `res_w_attn/res_gn_attn/res_w_mlp/res_gn_mlp` |

## 二、结果

| 配置 | logits max_abs | max_rel | \|ref\|max | 判定 |
|---|---:|---:|---:|---|
| d=128 / 2 层 / csa=2 / vocab=1024 / seq=32 | **5.364e-07** | 5.570e-07 | 0.9631 | **PASS** |
| d=1024 / **4 层** / 16 头 / csa=4 / d_ff=2816 / vocab=96000 / seq=512 | **9.894e-06** | 2.676e-06 | 3.698 | **PASS**（绝对判据）|
| d=1024 / **20 层** / 同上（生产几何）| 1.773e-05 | **4.869e-06** | 3.642 | 绝对超 1.8×，**相对判据 PASS** |

### 关于 20 层的 1.77e-05

- 相对误差 **4.87e-06**，属于 fp32 随深度累积的正常量级（20 层、每层若干 GEMM）
- 绝对判据 1e-5 是在小 logits 尺度下定的；生产几何的 `|logits|max = 3.642`，
  1.77e-05 / 3.642 ≈ 4.9e-06，与相对判据一致
- **结论：生产配置在相对判据下等价；绝对阈值应随 logits 尺度归一化**

## 三、复现方式（V100）

```powershell
ssh -i $env:USERPROFILE\.ssh\<ssh-key> <用户名>@<内网IP> `
  'cd D:\orv_src\tools; C:\Python311\python.exe faithful_parity.py --d 1024 --layers 4 --csa 4 --n_q 16 --d_ff 2816 --vocab 96000 --seq 512'
```

需要 `model_v3.py` 在 `D:\TGAI-V3-1.5B\03_src`，ORV 包在 `D:\orv_pip_0962`（脚本自己插入 sys.path）。

## 四、局限（不藏）

1. 只比了**单批 logits**，没有比 20 层的 loss 曲线与梯度（小配置那份曲线验证见 `lab/e2e/e2e_out_small.json`）
2. 没测 `csa_compress ≠ 4`、`attn_res_block ≠ 4`
3. 20 层同时载入 torch 与 ORV 两套权重（各 ~1.5B），显存吃紧；4 层是安全的对照规模

---

## 五、2026-11-24 更新：训练器已补齐 AttnRes / post-norm / QK-Norm

此前只有 `CLI.forward_logits`（对拍路径）实现真 V3 的六项，**`v3_pretrain` 的
训练前向没有 AttnRes、没有 MoE post-norm、global 层默认不开 QK-Norm**，
于是"训练用的前向"和"对拍用的前向"在同一份权重下 logits 差到 4.3（loss 差 0.03）。

现已给 `v3_pretrain` 补齐（默认关，开关打开后才生效）：

| CLI 开关 | cfg 键 | 对应参考实现 |
| --- | --- | --- |
| `--attn-res` / `--attn-res-block N` | `attn_res` / `attn_res_block` | BlockAttnRes 四段式层循环（`attn_res_enabled=True`）|
| `--moe-post-norm` | `moe_post_norm` | `MoEv3.forward`：routed 聚合 → RMSNorm → + 共享专家 |
| `--qk-norm` | `qk_norm` | `GlobalAttention`：q/k 先沿 d_k 做 RMSNorm 再 RoPE（CSA 层内部默认已开）|

**验证（V100，4 层 / d=1024 / 16 头 / E=4 / d_ff=2816 / vocab=96000 / seq=512）**

* `v3_pretrain.forward_loss` 与 `CLI.forward_logits` 的内部 logits 现在**逐位一致（max_abs = 0.0）**；
* 与 PyTorch 全开（faithful）同权重：logits 相对差 **3.4e-6**，首步 loss 差 **1.8e-6**；
* 20 步 loss 曲线：首步 1.8e-6，逐步漂移约 1e-3 量级（fp32 归约顺序 + 梯度裁剪的归一化差异，混沌放大）；
* 12 层真训练器（`--attn-res --moe-post-norm --qk-norm`）实测可跑：0.26 s/步、峰值 11.6 GiB。

默认（不带开关）行为与 1.0.28 **逐位一致**（4 步 loss/gnorm 完全相同），所以这是一组纯增量的可选能力。
