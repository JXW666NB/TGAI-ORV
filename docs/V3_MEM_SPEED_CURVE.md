# V3 骨架：ORV vs PyTorch 的显存上升 / 速度下降曲线（2026-11-13）

> 口径：两侧都是**完整训练步**（AdamW、权重真更新、反传），d=512、8 层、8 专家、vocab=4096、B=1。
> 显存用**精确计数**：ORV 取运行时池高水位 `pool_stats().peak_bytes`，PyTorch 取
> `torch.cuda.max_memory_allocated()`。**不使用 nvidia-smi 采样**（实测会漏尖峰，同配置两次差 3 倍）。
> **loss 不可比**（独立随机初始化），只作"在学"的旁证。

## 1. 显存上升曲线

| seq | ORV 物化（精确） | ORV `lowmem`（精确） | PyTorch（精确） | 物化/PyT | lowmem/PyT |
| --- | --- | --- | --- | --- | --- |
| 512（dh=128）| **3533 MiB** | —（dh≠64 不可用）| 4508 MiB | **0.78×** ✅ | — |
| 1024（dh=128）| 5276 MiB | — | 5058 MiB | **1.04×** ❌ | — |
| 1024（dh=64）| 5826 MiB | **4543 MiB** | 5058 MiB | 1.15× ❌ | **0.90×** ✅ |
| 2048（dh=64）| ≈11900 MiB* | **≈6100 MiB*** | 6516 MiB | ≈1.8× ❌ | **≈0.94×** ✅ |

\* seq2048 的两个精确字节数在输出里被截断，表中按 nvidia-smi 口径（12310 / 6536 MiB，含 330 MiB 空载）换算估计。

趋势（这是最关键的一条）：

* **PyTorch**：4508 → 5058 → 6516（seq ×4，显存 ×1.45）—— 接近平坦；
* **ORV 物化**：3533 → 5276 → ≈11900（seq ×4，显存 ×3.4）—— **超线性**；
* **ORV lowmem**：— → 4543 → ≈6100（seq ×2，显存 ×1.34）—— 与 PyTorch 同量级。

原因：物化路径要活存 `B·H·S²` 的分数矩阵，`S²` 直接进显存；PyTorch 的融合 SDPA 不物化，
`lowmem` 内核也不物化。所以**长序列下物化路径必然越线**，这不是调参能救的。

## 2. 速度下降曲线（每步耗时）

| seq | ORV 物化 | ORV `lowmem` | PyTorch | 物化/PyT | lowmem/PyT |
| --- | --- | --- | --- | --- | --- |
| 512 | 0.418 s | — | 0.265 s | 1.58× 慢 | — |
| 1024 | 0.767 s | 2.028 s | 0.265 s | 2.9× 慢 | 7.7× 慢 |
| 2048 | 1.494 s | 4.033 s | 0.320 s | 4.7× 慢 | **12.6× 慢** |

趋势：PyTorch 每步耗时几乎不随 seq 变（0.265 → 0.320），ORV 两条路径都随 seq 线性以上增长。

## 3. 结论：两样不可兼得，缺的是分块 flash 内核

| 路径 | 显存 | 速度 |
| --- | --- | --- |
| 物化 | 差（超线性）| 中 |
| `lowmem` | 好（与 PyT 同量级）| 很差（12.6× 慢）|
| **分块 flash** | 好 | 好 |

ORV 源码注释已给出 `lowmem` 慢的原因：内核"每行一个线程、串行扫 k"，并行度不如分块归约。

**因此下一步的第一优先级是写分块 flash 注意力内核**；norm 融合、路由 D2H、router 参数化都只是几个百分点。

## 4. 复现命令

```powershell
# ORV（物化 / lowmem）
python tools/v3_orv_train.py --d 512 --layers 8 --E 8 --n_q 8 --seq 2048 --steps 3
python tools/v3_orv_train.py --d 512 --layers 8 --E 8 --n_q 8 --seq 2048 --steps 3 --lowmem
# PyTorch 同配置
python tools/v3_pyt_train.py --d 512 --layers 8 --E 8 --seq 2048 --steps 3
```

判读时务必先看：**两侧架构尚未完全对齐**（ORV 全 GlobalAttention、top_k=1、无共享专家/BlockAttnRes），
所以倍数反映的是"当前实现"的差距，不是引擎的极限。
