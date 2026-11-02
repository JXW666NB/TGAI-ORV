# -*- coding: utf-8 -*-
"""确定性模式：同一输入下反向必须逐位可复现（需要 GPU）。

默认关闭（快路径仍是 atomic 版）。``TG_DETERMINISTIC=1`` 或 ``rt.deterministic = True``
时反向改走无原子内核：rmsnorm 的 dw（按列升序）、词嵌入（按 id 独占一行）、
repeat_interleave（按 j 升序归约）。

实测（V100，Qwen2.5-0.5B 形状）：反向两次差异 9.5e-07 -> 0.0；
整步训练两次参数最大差 0.0；代价约 1%（717 -> 724 ms/step）。
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


def test_backward_is_bitwise_reproducible_in_deterministic_mode():
    import tgai_go.train as T
    from tgai_go.train import qwen as Q

    T.set_default_device("cuda")
    S = 32
    cfg = dict(V=256, D=64, L=2, NH=4, KV=2, FF=128, THETA=1e4, EPS=1e-6, DH=16,
               QKV_BIAS=True, TIE=True)
    rng = np.random.default_rng(0)
    w = Q.make_random_weights(cfg, seed=0)
    ids = rng.integers(0, 200, (2, S)).astype(np.int64)
    tgt = rng.integers(0, 200, (2 * S,)).astype(np.int64)
    rt = T._cuda()
    old = getattr(rt, "deterministic", False)
    try:
        rt.deterministic = True
        m = Q.Qwen25(cfg, w)

        def grads():
            opt = T.optim.AdamW(m.params(), lr=1e-4, eps=1e-8)
            loss = T.functional.cross_entropy(m.forward(ids).reshape(2 * S, cfg["V"]), tgt)
            loss.backward()
            rt.sync()
            g = {k: np.asarray(v.grad.to_numpy(), np.float64).copy()
                 for k, v in m.p.items() if v.grad is not None}
            opt.zero_grad()
            return g

        g1, g2 = grads(), grads()
        worst = max(float(np.abs(g1[k] - g2[k]).max()) for k in g1)
        assert worst == 0.0, "确定性模式下反向仍不可复现：最大差 %.3e" % worst
    finally:
        rt.deterministic = old
