# svs-ops 插件包：算子清单与验证状态

> 2026-11-15。包版本 **0.4.2**，21 个算子，全部 `form: composed`（梯度由 autograd 自动给出）。
> 对拍脚本：`tools/verify_svs_ops.py`（51 项，全部相对误差 ≤ 2e-07）。
> 装法一行：`T.plugin.install("examples/bundles/svs-ops")` 或 `.tgop`。

## 已验证（可直接用）

| 算子 | 验证方式 | 结果 |
| --- | --- | --- |
| `stft_mag` | vs `torch.stft().abs()`，4 档分辨率，前向+反向 | ≤ 2e-07 |
| `stft_mag_fast` | 同上（hop 块分解，无分帧卷积） | ≤ 2e-07 |
| `hann_window` | vs `torch.hann_window` | 1.8e-07 |
| `pad1d` | vs `F.pad`，constant/replicate/reflect，前向+反向 | 0 |
| `cumsum` | vs `torch.cumsum`，前向+反向 | 0 |
| `sigmoid` `softplus` `log1p` `log10` `pow` `leaky_relu` | vs torch，前向+反向 | ≤ 1.2e-07 |
| `l1_loss` `mse_loss` `bce` `bce_with_logits` | vs `F.*`，前向+反向 | ≤ 1.2e-07 |
| `avg_pool1d` | vs `F.avg_pool1d` | 0 |
| `interpolate1d` | vs `F.interpolate`，**12 组**（含 96256↔189），linear+nearest | ≤ 3e-08 |
| `searchsorted_np` | vs `torch.searchsorted`，**6 组**，left/right | 逐位相等 |
| `gather_dim` | vs `torch.gather`，**4 组**（含 4 维、非末维） | 逐位相等 |

## 三个踩过的坑（都值得记）

1. **STFT 相位**：hop 块分解里，段内坐标是**局部** `u=0..hop-1`，基矩阵要取**前 hop 列**。
   取 `[s*hop:(s+1)*hop]` 等于把相位算两遍。最能骗人的是 **bin 0 恰好正确**（零频无相位）。
2. **帧数锚定**：帧数必须按**原始信号长度**算；补齐到 hop 整数倍只是为了 reshape，不能改帧数。
3. **取整精度**：`F.interpolate` 的源坐标是 **float32** 算的。用 float64 会在边界差 1 格
   （96256→189 相对差 0.84）。`gather` 则必须**连索引一起换轴**，否则位置错配。

> 共同教训：torch 的官方行为**只能靠对拍确认**，不能靠推导。这三次全是"想当然"翻的车。

## 还没做的（SGR 端到端所需）

| 缺的 | 用在哪 | 难度 | 备注 |
| --- | --- | --- | --- |
| `istft` | DDSP 谐波/噪声合成，**训练路径** | 中 | 可用 cuFFT，或用 matmul 反变换 + 重叠相加 |
| 双向 GRU | encoder（512→256×2） | 中 | ORV 无 GRU；可先用逐时刻循环 |
| MultiheadAttention | Emformer 4 层、8 头 | 中 | ORV 只有 CausalSelfAttention |
| `linalg.pinv` | 构造期一次（`__init__`） | 低 | 可用 numpy 预计算后传入 |
| `linalg.unique` | 仅 joint 阶段 | 低 | 主机侧 numpy 即可 |

## 已知性能缺口

组合式 `stft_mag` 比 cuFFT 慢 **32×**（166.7 ms vs 5.2 ms，B=2、L=96000、6 档分辨率、
前向+反向）。差距来自**算子调度次数**（一次 STFT≈40 次调用、一次损失 12 个 STFT、
正反向近千次），不是数学量。追平需把 STFT/iSTFT 做成**带反向的原生算子**
（机器上有 `cufft64_11.dll`），并配一份组合式 `impl.fallback` 让反向自动导出。
