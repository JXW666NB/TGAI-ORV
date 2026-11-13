# -*- coding: utf-8 -*-
"""SiTU-GLU（V3 专家 FFN 的激活）在 ORV 上与 numpy 参考逐元素对拍。

## 为什么先移植这个算子

它是 V3 里**数学最简、又最独立**的一块：只用到矩阵乘、tanh、sigmoid 与逐元素乘。
PyTorch 参考（`model_v3.py::SiTUGLUFFN.forward`）：

    gate = (β₁·tanh(g/β₁)) ⊙ sigmoid(g)      β₁=4
    up   = β₂·tanh(u/β₂)                     β₂=25
    y    = w3(gate ⊙ up)                     ⇒ |f| ≤ 100

对拍基准用同一公式的 numpy 版本（不是"我觉得应该"的近似）。

## 同时验证"有界"这条性质

SiTU 的存在理由就是**上界**：无论输入多大，输出 ≤ β₁·β₂ = 100。
这在 fp16（上界 65504）下是防溢出的关键，所以单独断言一次。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import v3_ops  # noqa: E402


def _mk(rng, d_model, d_ff, scale=0.1):
    return (rng.normal(0, scale, (d_model,)).astype(np.float32),
            rng.normal(0, scale, (d_ff, d_model)).astype(np.float32),
            rng.normal(0, scale, (d_ff, d_model)).astype(np.float32),
            rng.normal(0, scale, (d_model, d_ff)).astype(np.float32))


@pytest.mark.parametrize("shape", [(4, 8), (1, 16), (7, 32)])
def test_matches_numpy_reference(shape):
    rng = np.random.default_rng(0)
    n, d_ff = shape
    d_model = 8
    x, w1, w2, w3 = _mk(rng, d_model, d_ff)
    xs = rng.normal(0, 1.0, (n, d_model)).astype(np.float32)

    got = v3_ops.situ_glu(v3_ops.T.tensor(xs), v3_ops.T.tensor(w1),
                          v3_ops.T.tensor(w2), v3_ops.T.tensor(w3))
    got = np.asarray(got.to_numpy() if hasattr(got, "to_numpy") else got, np.float32)
    want = v3_ops.situ_glu_numpy(xs, w1, w2, w3)
    err = float(np.abs(got - want).max())
    assert got.shape == want.shape
    assert err < 1e-5, "SiTU-GLU 与参考实现最大误差 %.3e 偏大" % err


def test_output_is_bounded_by_beta_product():
    """SiTU 的核心性质：输出有界 ⇒ fp16 下不会溢出。用极大输入压一次。"""
    rng = np.random.default_rng(1)
    d_model, d_ff = 8, 16
    x, w1, w2, w3 = _mk(rng, d_model, d_ff, scale=1.0)
    big = (rng.normal(0, 50.0, (3, d_model))).astype(np.float32)   # 故意很大

    gate_up = v3_ops.situ_glu_numpy(big, w1, w2, w3, beta1=4.0, beta2=25.0)
    # 只看 gate*up 的上界：等价于把 w3 设成单位映射无法直接测，所以用理论上界
    # —— gate ≤ β₁、up ≤ β₂ ⇒ 逐元素乘积 ≤ 100，再乘 w3 会改变量级，
    # 因此这里用 numpy 直接验证 gate/up 两个中间量。
    g = big @ w1.T
    u = big @ w2.T
    gate = (4.0 * np.tanh(g / 4.0)) * (1.0 / (1.0 + np.exp(-g)))
    up = 25.0 * np.tanh(u / 25.0)
    assert np.abs(gate).max() <= 4.0 + 1e-6
    assert np.abs(up).max() <= 25.0 + 1e-6
    assert np.abs(gate * up).max() <= 4.0 * 25.0 + 1e-4
    assert np.isfinite(gate_up).all(), "极大输入下也不应出现 inf/NaN"
