---
name: 性能回归 / 性能问题
about: 某处变慢，或与 PyTorch 对比在某配置下落后
labels: performance
---

**口径（缺一不可）**
- 模型：层数 / d / d_ff / E / top_k / vocab / seq：
- 优化器与精度（`--opt` / `--amp` / 梯度检查点开关 / batch）：
- 步数（**必须 ≥20 步取稳态**，3 步会低估，见 CONTRIBUTING）：
- 硬件：
- ORV vs 对照（PyTorch？另一版本？）：

**实测**
| 配置 | ORV | 对照 | 倍数 |
| --- | --- | --- | --- |

**已排除的方向**（这条最有价值）
-

**注意**：如果用了「调用前后 sync」给单个 kernel 计时，请先读 CONTRIBUTING 的提醒 ——
那种测法会把小 kernel 的耗时放大几十倍，用 CUDA event 重测后再报。