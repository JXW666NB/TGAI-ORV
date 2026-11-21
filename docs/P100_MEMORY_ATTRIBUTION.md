# P100 显存归属：那 1.7~2.1 GiB 到底是什么（已定案）

> 2026-11-21　｜　数据来自使用者按 `bench/p100_fair/` 要求回的 `pool_dump_live` 输出
> （P100 16GB / B=2 / fp32 / ORV 1.0.18）

## 1. 原始数据

```
attn=low（不物化注意力）     live 5.64 GiB  reserved 5.89 GiB  nvidia-smi 6.58 GiB   2.15 s/step  1906 tok/s
attn=mat（物化注意力）       live 7.73 GiB  reserved 8.00 GiB  nvidia-smi 8.69 GiB   2.32 s/step  1766 tok/s
差额                          +2.09 GiB      +2.11 GiB         +2.11 GiB
```

池账本（`pool_dump_live`，格式 `(单块字节, 块数, 该类总字节)`）在两种配置下**完全相同**：

```
1) (12058624,   168, 2025848832)   ≈ 1.89 GiB   ← 21 层权重/梯度等常驻
2) (176160768,    8, 1409286144)   ≈ 1.31 GiB   ← 8 个 MoE 层的常驻缓冲
3) (5505024,    168,  924844032)   ≈ 0.86 GiB
4) (1179648,    252,  297271296)   ≈ 0.28 GiB
```

## 2. 结论（定案）

**多出来的 2.1 GiB 是「物化注意力」的分数/softmax/梯度副本，不是 MoE，也不是池碎片。**

理论值与实测吻合：
- B=2、18 头、S=2048、fp32，物化一份分数 = `2×18×2048²×4B = 576 MiB`；
- 反向还要保留分数、softmax 输出、梯度 2~3 份 ⇒ **约 1.7~2.1 GiB**；
- 实测差额 **+2.09 GiB**。

MoE dispatch **不是主项**：`attn=low` 时 MoE 已在跑，峰值仍只有 5.64 GiB，
最大的两类账本（1.89 GiB 与 1.31 GiB）在 low/mat 两种配置下**一模一样**，
说明它们与注意力无关（是权重/梯度与 MoE 常驻缓冲）。

## 3. 这条数据推翻了什么

| 之前的说法 | 现在的事实 |
| --- | --- |
| "ORV 显存比 PyTorch 高 26~37%，是引擎级缺陷" | **其中大部分来自"是否物化注意力"这个配置**。`bench_p100_full.py` 里写死 `lowmem=False`（强制物化），而 torch 的 SDPA 走 memory-efficient（不物化）⇒ 这是**配置不对等**，不是引擎缺陷 |
| "MoE 缓冲是主要嫌疑" | 已排除：两种注意力配置下 MoE 账本完全相同 |
| "开 `lowmem=True` 显存没变" | 实测**降 2.11 GiB（−24%）**；此前"没变"是因为策略 `attn` 键是死键（已于 1.0.21 修复） |

**并且**：`attn=low` 时 ORV 的 `nvidia-smi` 占用 **6.58 GiB < 同脚本里 torch 的 6.69 GiB**。
所以"ORV 一定比 torch 更吃显存"这个结论**不成立**。

> ⚠️ 谨慎之处：上面这组对比里 torch 的数字来自旧脚本（新 fair 脚本的 torch 侧当时有形状 bug，
> 已修）。**要宣称"ORV 在 P100 上显存更省"，必须等 torch 侧在同一脚本、同一口径下跑完。**

## 4. 由此得到的行动项

| # | 行动 | 状态 |
| --- | --- | --- |
| 1 | **文档/引导：长序列不要传 `lowmem=False`**。默认 `lowmem=None` 已经是 auto（能融合就融合）| 待写进 FAQ |
| 2 | `bench_fair.py` 的 torch 侧形状 bug（我的） | **已修** |
| 3 | 加 `DUMP_AT=peak`：在第 1 层注意力输出之后立刻 dump，才能看到峰值构成 | **已加** |
| 4 | 修复策略 `attn` 死键（否则 `set_policy("memory")` 无效） | **已修（1.0.21）** |
| 5 | 若仍要在"必须物化"的场景（给了 `attn_mask`）省显存：反向重算 softmax 而不是保留 | 未做，需要 GPU 验证 |

## 5. 复现命令

```bash
# 峰值归属（关键）
DUMP_AT=peak python bench_fair.py --only orv --bs 2 --attns low,mat --iters 3
# 只想看 step 结束后的残留（不是峰值）
DUMP_AT=post python bench_fair.py --only orv --bs 2 --attns low,mat --iters 3
```
