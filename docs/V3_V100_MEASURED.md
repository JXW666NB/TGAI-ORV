# V3 在 V100 上的实测记录（2026-11-16）

> 全部数字来自真机（Tesla V100-PCIE-32GB），不是估算。每条都注明配置与来源。
> 未测的项明确标「未测」，不要引用成结论。

## 一、ORV vs PyTorch 主表（同脚本、同配置、同数据、AdamW 真更新）

两侧都用 `tools/v3_orv_train.py` / `tools/v3_pyt_train.py`，`steps=2`，`top_k=2`，共享专家开。

| 配置 | ORV s/step | PyT s/step | 比 | ORV 峰值 | PyT 峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| d=256 / 4 层 / E4 / seq128 | **0.0729** | 0.2279 | **3.13×** | 554 MiB | 773 MiB |
| d=1024 / 8 层 / E4 / seq512 | **0.2731** | 0.5666 | **2.07×** | 11.58 GB | 10.19 GB |
| **d=1024 / 20 层 / E4 / seq256** | **0.4347** | 1.0619 | **2.44×** | **23.07 GB** | 25.24 GB |

- 第 3 行是**同精度（双方 fp32）**对比，也是最有意义的一组：ORV 快 2.44 倍、显存少 8.6%。
- 峰值口径：ORV 用 `pool_stats()["peak_live_bytes"]`；PyT 用 `torch.cuda.max_memory_allocated`。
- PyT 侧参数量（20 层配置）：1,317,787,663。

## 二、fp16 / Tensor Core：开了反而慢（反直觉，实测）

`rt.fp16_gemm` 用于让 Linear 的 GEMM 走 cuBLAS fp16（Tensor Core）。同配置 20 层：

| 设置 | s/step | tok/s | 池峰值 |
| --- | ---: | ---: | ---: |
| `fp16_gemm = False` | **0.4347** | **588.9** | **23.07 GB** |
| `fp16_gemm = True` | 0.5073 | 504.6 | 26.12 GB |

**慢 16.7%、多占 3 GB。** 推断原因（未逐段 profile）：权重影子按「代」失效，
而 AdamW 每步更新权重 ⇒ 每步都要把 1.3B 参数重新 cast，再加上 fp16 缓冲。
`TG_FP16_SPLIT=1` 同样更慢（12.50 → 18.75 ms，一个 GEMM 的前向+反向），
它本来就是**精度修复工具**而非加速开关（`docs/PRODUCTION_READINESS.md` 里 V100 实测慢 1.37×）。

**结论：V100 上当前形状下，fp16 三条路径（单次 / split / 影子）都没有正收益。**
要真正吃到 TC，需要换思路（如只前向用 fp16、或 `amp_mlp` 省激活带宽），**尚未验证**。

## 三、真 V3 架构对齐情况

真实配置取自 `model_v3.py:89-138`：`d_model=1024`、`n_layers=20`、`n_heads=16`、
`d_ff=2816`（每专家）、`vocab=96000`、`attn_global_every=4`（3:1 CSA/Global）、
`n_kv_heads=0`（MHA）、`attn_qk_norm=True`、`tie_embeddings=True`。

| 项 | 上面实测的配置 | 真 V3 | 差距 |
| --- | --- | --- | --- |
| 层数 / 宽度 | 20 / 1024 | 20 / 1024 | ✅ 一致 |
| 每专家 d_ff | 4096 | **2816** | ⚠️ 但 0.9.83 已支持 `--d_ff` |
| vocab | 4096 | **96000** | ⚠️ 已支持 `--vocab` |
| 注意力 | 全 Global | **3:1 CSA 混合** | ❌ **CSA 未实现** |
| n_kv | 2（GQA） | 0 = MHA | ⚠️ |
| QK-Norm / tie embeddings | 无 | 有 | ⚠️ |

**CSA 是唯一的大缺口**：`v3_ops.csa_compress`（:258）与 `csa_conv_causal`（:313）都是
**numpy 参考实现**（`np.asarray` 进、`np.zeros` 出，无 autograd、无 GPU），
不能进训练循环。要补需要按 ORV 算子重写（reshape+mean 做压缩、平移相加做因果卷积、
`v3_masks` 做块对角掩码），并做数值对拍。

## 四、GPU 上撞出的三个 CPU-only 算子（重要）

写 V3 移植时若照抄 PyTorch 写法，必然在这三处崩，而 **CPU 测试完全发现不了**：

| 算子 | 位置 | 症状 |
| --- | --- | --- |
| `einsum` | `__init__.py:2711` | `NotImplementedError: CUDA einsum 尚未实现` |
| `topk` | `__init__.py:2565` | `NotImplementedError: CUDA topk 尚未实现` |
| **`stack`** | `__init__.py:2453` | `NotImplementedError: CUDA stack 尚未实现` |

`v3_attnres` 已全部绕开：用逐块 matmul + `exp` + 广播乘实现 softmax-over-blocks，
GPU 上与 numpy 参考 **max_abs_err 1.788e-07**（CPU 上 2.384e-07）。

## 五、BlockAttnRes（本轮新增能力）

- `python/tgai_go/train/v3_attnres.py`：`block_attn_res` + `run_attn_res_layers` + numpy 参考
- `tools/v3_orv_attnres_loop.py`：四段式层循环（attn 前 / 收块 / mlp 前）
- `tools/v3_orv_train.py`：`--attn-res` 开关（默认关，旧路径逐字不变）
- 数值：CPU 2.384e-07 / 1.907e-06；**GPU 1.788e-07**
- 守卫测试 3 项（含「输出必须落在各 block 的凸包内」，能抓出权重未归一化）

## 六、未测 / 未验证（不要引用成结论）

1. ~~① 前向数值误差~~ **已测**：见下节「六」；但只覆盖 FFN 块，整模型端到端仍缺
2. 真 V3 尺寸（d_ff=2816 / vocab=96000）下的步时与显存
3. CSA 的 GPU 实现与对拍
4. fp16 的可行用法（只前向 / `amp_mlp` / 配合 `Fp16MomAdamW`）
5. 长 seq（≥1024）下的显存曲线


## 七、① 前向数值误差（2026-11-16 实测，共享权重）

工具：	ools/parity_orv_vs_torch.py（torch 与 ORV 同进程，同一批 numpy 权重）。

| 模块 | max_abs | max_rel | 输出量级 | 判定（≤1e-5）|
| --- | ---: | ---: | ---: | --- |
| RMSNorm(γ) → Linear(D→F) → SiLU → Linear(F→D) | **4.470e-08** | **2.400e-07** | 0.1863 | ✅ 通过，富余约两个数量级 |

配置：B=2 / S=64 / D=128 / F=256 / seed=0 / V100 / 双方 fp32。

**一个必须记住的教训**：两侧各自用 default_rng(0) 初始化时，随机数消费顺序不同
⇒ 权重不同 ⇒ 首步 loss 差 0.93%（8.5012 vs 8.4227），**这个差不含任何精度信息**。
2026-11-16 差点把它当成对拍结果。必须共享权重才能测精度。

**覆盖范围**：只测了 FFN 块。**未测**：RoPE 约定（用错会 rel≈0.44 且不报错）、
GQA 头顺序、因果掩码边界、MoE top-k 并列、整模型端到端。
