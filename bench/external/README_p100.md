# P100 上 torch vs TGAI-ORV 对比包

作者要求的复现材料。**同一份手写实现**分别用 torch 和 `tgai_go.train` 跑，
形状完全一致，差异只来自框架本身。

---

## 1. 为什么是"稠密等价"形状

我们的真实训练模型是 **MLA + SMS2-MoE**（36 专家 + 1 常驻、强 2 弱 3、W_O[1728,1152]），
ORV 里没有对应算子，无法逐算子对拍。

所以把 MoE 全替换成**稠密 FFN(2560)**，只保留公共骨架：

```
21 层 × (MLA 投影: q / q_rope / kv_down / k_up / v_up / k_rope / o
        + 注意力 (18 头 × 64)
        + FFN 1152 → 2560 → 1152)
S = 2048，d = 1152，词表 36864，梯度检查点全开
```

> ⚠ **说明**：ORV 对 MoE 有专门优化、对稠密 FFN 没有。
> 这个基准测的是稠密路径，ORV 的 MoE 优势在这里体现不出来——
> 我们这么做的原因是：这是唯一能保证"同一份代码、同一个形状"的对照方式。

---

## 2. 运行方法

```bash
# 环境：Windows / P100 16GB / torch 2.7.1+cu118 / tgai-orv 0.9.64
set HB=4               # batch（默认 8）
set FP32=1             # torch 侧 fp32（默认 fp16 autocast）
set POL=standard       # ORV 档位：standard,speed,extreme,memory
set AMP=1              # ORV 侧开 T.amp()（只 cast GEMM 输入）

python bench_p100_full.py torch     # 只测 torch
python bench_p100_full.py orv       # 只测 ORV（按 POL 列表）
python bench_p100_full.py both
```

脚本会打印：步时、tok/s、峰值显存（torch 用 `max_memory_allocated`，
ORV 用 `rt.pool_stats()["peak_live_bytes"]`）。前 1 步是预热，不计入。

---

## 3. 实测结果（P100 16GB，B=4，检查点开）

| 框架 | 步时 | 吞吐 | 峰值显存 |
|---|---:|---:|---:|
| torch fp32 | 3.33 s | 2,459 tok/s | 7.37 GB |
| torch fp16 (autocast) | 3.34 s | 2,459 tok/s | 7.37 GB |
| ORV standard | 4.47 s | 1,832 tok/s | 11.33 GB |
| ORV speed | 4.47 s | 1,832 tok/s | 11.33 GB |
| ORV extreme | 4.47 s | 1,832 tok/s | 11.33 GB |
| ORV memory | 4.60 s | 1,782 tok/s | 13.27 GB |
| ORV memory + amp | 4.63 s | 1,771 tok/s | 13.38 GB |

batch 扫描（全部 fp32 + 检查点）：

| B | torch | ORV | 比值 |
|---:|---:|---:|---:|
| 1 | 1.17 s / 1,756 tok/s / 5.45 GB | 1.18 s / 1,730 tok/s / 5.96 GB | 1.01× |
| 2 | 1.76 s / 2,331 / 5.88 | 2.30 s / 1,779 / 7.75 | 1.31× |
| 4 | 3.33 s / 2,459 / 7.37 | 4.47 s / 1,832 / 11.33 | 1.34× |
| 8 | 6.06 s / 2,706 / 11.52 | OOM | — |

---

## 4. 结论要点

```
1. P100 上 torch 的 fp16 autocast 没有收益（无 Tensor Core），fp32 就是公平基线
2. ORV 在 B=1 与 torch 打平（1.01×），B≥2 落后 30%+，且几乎不随 batch 提升
   （ORV 1,730 → 1,832；torch 1,756 → 2,706）→ 差距与 batch 正相关，是结构性的
3. 三档在 P100 上完全等价：lowmem 需要 68 KB 动态 smem，sm_60 单 block 上限 48 KB
   → 融合注意力编不出来，只能走物化注意力
4. ORV 显存高 1.3~1.5×：激活全程 fp32，amp 不省激活显存
5. 对"稠密 FFN"没有特化，本基准里 ORV 的 MoE 优化无法体现
```
