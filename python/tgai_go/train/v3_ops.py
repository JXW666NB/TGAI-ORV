# -*- coding: utf-8 -*-
"""V3 算子移植（第一批）：SiTU-GLU。

## 参考实现（`D:\\TGAI-V3-1.5B\\03_src\\model_v3.py` 的 `SiTUGLUFFN.forward`）

```
g = w1(x); u = w2(x)
gate = (beta1 * tanh(g / beta1)) * sigmoid(g)
up   = beta2 * tanh(u / beta2)
y    = w3(gate * up)
```

β₁ = 4（gate 分支）、β₂ = 25（up 分支）⇒ 输出上界 |f| ≤ β₁·β₂ = 100。
论文动机：SwiGLU 的两个乘子都**无界**，低精度下容易溢出；SiTU 用 tanh 把乘积硬压住，
而 tanh 在原点近似线性 ⇒ 局部行为贴近 SwiGLU，不损失表达能力。

## 为什么这里不直接用 `S.sigmoid`

ORV 的 `train` 包里 **没有** 导出 `sigmoid`（只确认了 `tanh`）。用恒等式

    sigmoid(x) = 0.5 * (1 + tanh(x / 2))

实现，数学上**精确相等**，且只依赖已确认存在的 `tanh`。
这样这个算子不会因为"某个函数没导出"而在运行时才炸。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tgai_go import train as T  # noqa: E402


def _tanh(x):
    """ORV 的 tanh **只在 Tensor 上是方法**（`x.tanh()`），没有模块级 `T.tanh`。

    实测：`T.tanh` 不存在（AttributeError），所以这里统一包一层，
    以后若补了模块级函数，只改这一个地方。
    """
    return x.tanh()


def _sigmoid(x):
    """sigmoid 用 tanh 恒等式实现，避免依赖未导出的算子。"""
    return 0.5 * (1.0 + _tanh(x * 0.5))


def situ_glu(x, w1, w2, w3, beta1=4.0, beta2=25.0):
    """SiTU-GLU 前向：`y = w3( situ(gate) * up )`，权重形状均为 (out, in)。"""
    g = x @ w1.transpose(0, 1)
    u = x @ w2.transpose(0, 1)
    gate = (_tanh(g * (1.0 / beta1)) * beta1) * _sigmoid(g)
    up = _tanh(u * (1.0 / beta2)) * beta2
    return (gate * up) @ w3.transpose(0, 1)


def situ_glu_numpy(x, w1, w2, w3, beta1=4.0, beta2=25.0):
    """同一公式的 numpy 版本，作为对拍基准（与 PyTorch 参考逐式对应）。"""
    g = x @ w1.T
    u = x @ w2.T
    gate = (beta1 * np.tanh(g / beta1)) * (1.0 / (1.0 + np.exp(-g)))
    up = beta2 * np.tanh(u / beta2)
    return (gate * up) @ w3.T
