# -*- coding: utf-8 -*-
"""half-split RoPE 融合内核与组合实现的数值对拍（需要 GPU；CI 跳过）。

背景：现有 `fused_rope` 是**交错式**（配对 2k/2k+1），Qwen/LLaMA 用的是 rotate_half
（配对 d / d+dh/2）。直接接会静默算错（实测 max diff 2.478）。所以另写了一对内核
`rope_half_ip` / `rope_half_bwd`，这里钉住它与组合实现等价。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

pytestmark = pytest.mark.skipif(
    not os.environ.get("TG_RUN_CUDA_TESTS"),
    reason="需要 GPU：设 TG_RUN_CUDA_TESTS=1 在 GPU 机器上跑")


def _relerr(a, b):
    den = float(np.linalg.norm(b)) or 1.0
    return float(np.linalg.norm(a - b)) / den


def test_rope_half_matches_composition_forward_and_backward():
    import tgai_go.train as T
    from tgai_go.train import qwen as Q

    T.set_default_device("cuda")
    B, NH, S, DH = 2, 4, 32, 64
    rng = np.random.default_rng(0)
    c, s_ = Q.rope_tables(S, DH, 1e6)
    cos = T.tensor(c[None].astype(np.float32))
    sin = T.tensor(s_[None].astype(np.float32))
    x0 = rng.normal(0, 0.3, (B * NH, S, DH)).astype(np.float32)
    g = rng.normal(0, 0.1, (B * NH, S, DH)).astype(np.float32)

    xa = T.tensor(x0.copy(), requires_grad=True)
    xb = T.tensor(x0.copy(), requires_grad=True)
    ref = Q.Qwen25._rope_apply(xa, cos, sin)
    fus = T.rope_half(xb, cos, sin, S)
    assert _relerr(np.asarray(fus.to_numpy(), np.float64),
                   np.asarray(ref.to_numpy(), np.float64)) < 1e-6, "前向不一致"

    (ref * T.tensor(g)).sum().backward()
    (fus * T.tensor(g)).sum().backward()
    assert _relerr(np.asarray(xb.grad.to_numpy(), np.float64),
                   np.asarray(xa.grad.to_numpy(), np.float64)) < 1e-6, "反向不一致"


def test_rope_half_does_not_change_semantics_of_interleaved():
    """反向用例：确认 half-split 与交错式**不等价**（防止有人把两者混用）。"""
    import tgai_go.train as T
    from tgai_go.train import qwen as Q

    T.set_default_device("cuda")
    S, DH = 8, 8
    c, s_ = Q.rope_tables(S, DH, 1e6)
    cos = T.tensor(c[None].astype(np.float32))
    sin = T.tensor(s_[None].astype(np.float32))
    x0 = np.random.default_rng(1).normal(size=(1, S, DH)).astype(np.float32)
    half = T.rope_half(T.tensor(x0.copy()), cos, sin, S)
    inter = T.fused_rope(T.tensor(x0.copy()), cos, sin)
    diff = float(np.abs(np.asarray(half.to_numpy(), np.float64)
                        - np.asarray(inter.to_numpy(), np.float64)).max())
    assert diff > 1e-3, "half-split 与交错式本应不同（实测差 %.4f）" % diff
