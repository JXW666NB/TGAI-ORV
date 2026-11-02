# -*- coding: utf-8 -*-
"""SDPA lowmem 与物化实现的数值对拍（需要 GPU；CI 里跳过）。

注意：对拍**必须用同一份输入**。第一版脚本用共享 rng 生成两组数据，
比出来 relerr 1.42（等于完全正交），差点让我误判"内核是坏的" —— 实际是输入不同。
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


def _setup(bh=8, s=64, dh=64, seed=0):
    import tgai_go.train as T
    T.set_default_device("cuda")
    rng = np.random.default_rng(seed)
    return [T.tensor(rng.normal(0, 0.3, (bh, s, dh)).astype(np.float32))
            for _ in range(3)]


def _relerr(a, b):
    den = float(np.linalg.norm(b)) or 1.0
    return float(np.linalg.norm(a - b)) / den


def test_lowmem_matches_materialized_forward():
    import tgai_go.train as T
    from tgai_go.train import functional as F
    q, k, v = _setup()
    a = np.asarray(F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                  lowmem=False).to_numpy(), np.float64)
    b = np.asarray(F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                  lowmem=True).to_numpy(), np.float64)
    assert _relerr(b, a) < 1e-5, "前向差 %.3e" % _relerr(b, a)
    del T


def test_lowmem_matches_materialized_backward():
    import tgai_go.train as T
    from tgai_go.train import functional as F

    def grads(lowmem):
        rng = np.random.default_rng(7)
        q, k, v = [T.tensor(rng.normal(0, 0.3, (2, 32, 64)).astype(np.float32),
                            requires_grad=True) for _ in range(3)]
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=lowmem)
        (o * o).sum().backward()
        return [np.asarray(x.grad.to_numpy(), np.float64) for x in (q, k, v)]

    ref = grads(False)
    lm = grads(True)
    for name, a, b in zip(("dq", "dk", "dv"), lm, ref):
        assert _relerr(a, b) < 1e-4, "%s 差 %.3e" % (name, _relerr(a, b))
