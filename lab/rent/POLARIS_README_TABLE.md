# Polaris 精简对比表（可直接放 README）

> 口径：每个优化器**各自扫描 lr 至最优**（不是同一个 lr 硬比）；末段 = 最后 10 步均值。
> 环境：ORV 0.9.97 + `tgai_polaris` 1.0.0，V100-32GB / CPU numpy，seed=0，无 wd / 无裁剪 / 无 lr 调度。
> 详细报告见 `REPORT_polaris_bench.md`。

## 表 1｜收敛性（两个任务，排序相反）

**任务 A：玩具回归**（W 64×32 拟合 256 样本，有唯一 float64 最优解；1200 步，CPU）

| 优化器 | 最优 lr | 末10 超额损失 ↓ | 相对 float64 最优的参数误差 |
|---|---|---|---|
| AdamW | 3e-3 | **1.4e-9** | 6.4e-5 |
| Adafactor | 1e-3 | **≈0 (−3e-10)** | 1.8e-7 |
| **Polaris** | **3e-4** | **8.9e-7** | 9.7e-4 |
| Muon | 3e-3 | 3.4e-6 | 2.0e-3 |
| LAMB | 3e-3 | 4.0e-6 | 2.6e-3 |
| Lion | 1e-3 | 1.4e-5 | 3.7e-3 |

**任务 B：迷你 Transformer 固定批过拟合**（1,049,216 参数；B=8×seq64；记忆任务）

| 优化器 | 最优 lr | 末10 loss ↓（CPU 150 步） | 末10 loss ↓（CUDA 200 步） |
|---|---|---|---|
| **Polaris** | **3e-3 / 1e-3** | **9.96e-8** | **5.49e-8** |
| Muon | 1e-2 | 7.12e-7 | **1.86e-8** |
| AdamW | 1e-2 | 2.65e-4 | 2.06e-4 |
| Lion | 1e-3 | 6.61e-4 | 6.43e-4 |
| Adafactor | 1e-2 | 7.89e-4 | 4.25e-4 |

> **结论**：凸优化任务上 `AdamW ≈ Adafactor > Polaris > Muon > Lion`；
> 记忆/拟合任务上 `Muon ≈ Polaris ≫ AdamW > Lion ≈ Adafactor`（差 3~4 个数量级）。
> **选型必须看任务类型。**

## 表 2｜速度与显存（CUDA，`opt.step` 单独计时，CUDA event，预热 3 + 3 次中位数）

模型 9,441,536 参数（8 层 / d=256 / vocab 4096），B=2×seq64：

| 优化器 | opt.step ms ↓ | 整步 ms | opt 占整步 | tokens/s |
|---|---|---|---|---|
| Int8AdamWGpu | **0.261** | 20.43 | 1.3% | 6266 |
| Fp16MomAdamW | **0.332** | 21.72 | 1.5% | 5893 |
| **AdamW** | **0.341** | 19.65 | 1.7% | 6515 |
| **Polaris** | **56.79** | 75.60 | 75% | 1693 |
| Muon | 40.04 | 59.49 | 67% | 2152 |
| Lion | 149.33 | 159.78 | 93% | 801 |
| Adafactor | 160.81 | 172.80 | 93% | 741 |
| *参考：PyTorch AdamW* | *0.997* | – | – | – |

模型 1,049,216 参数（2 层 / d=128）对照：

| 优化器 | opt.step ms | 整步 ms | opt 占整步 |
|---|---|---|---|
| AdamW | 0.051 | 8.18 | 0.6% |
| **Polaris** | **14.30** | 22.05 | 65% |
| Muon | 10.68 | 17.72 | 60% |
| Lion | 11.07 | 19.69 | 56% |
| Adafactor | 13.39 | 20.21 | 66% |

> **必须一起读的三点**：① 分水岭是"融合 kernel"（AdamW/fp16mom/int8gpu 单 kernel 0.26~0.34 ms）
> vs "逐参数发 kernel"（Muon/Lion/Adafactor/Polaris 10~160 ms）——**Polaris 不是这一类里最慢的**；
> ② 模型从 1.05M 涨到 9.4M，Polaris/AdamW 的 `opt.step` 倍数只从 283× 降到 **167×**（Newton-Schulz 是 FLOP 密集型，
> 不会随规模收敛），但整步倍数只有 **2.7× → 3.8×**；
> ③ 本表**只适用于 ≤10M 参数的小模型**；大模型上 fwd/bwd 占比上升会继续摊薄，但 `opt.step` 的绝对差距不会消失。

## 表 3｜稳定性

| 项 | 结果 |
|---|---|
| NaN / Inf | 全部 4 个任务 × 6 个优化器 × 全部 lr 网格：**无 NaN/Inf** |
| AMP（`with T.amp():`） | AdamW / Polaris 开关均不掉点（Polaris: 1.22e-5 → 1.25e-5），无 NaN |
| **Polaris 稳定窗口** | 1e-3 稳定；**3e-3 先收敛到 <1e-6、80 步后失稳进入 0.05~0.36 极限环**；1e-2 第 45 步崩到 0.11+ 不恢复（CPU/CUDA 均复现） |
| 建议 | **Polaris 的 lr 要比 AdamW 小约 10×，且必须配 lr 衰减** |
