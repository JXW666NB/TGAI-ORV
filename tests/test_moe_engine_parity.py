# -*- coding: utf-8 -*-
"""MoE 与产线引擎的**数值等价**（CPU，进 CI）。

依据（都是源码原话，不是推测）：
* `tgai_engine.c:1120`  —— router logits **先加 qb_bias**，再做 top-k
* `tgai_cuda.cu:301-307` —— **softmax 仅对选中的 k 个做**（减最大值后 exp 再归一化）

V2.7 是 8 专家 / 激活 2；老模型没有 qb_bias（None 时行为不变）。
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import MoE  # noqa: E402


def _np(t):
    """ORV 的 Tensor 用 to_numpy()；返回值必须是普通 ndarray 才能比对。"""
    return t.to_numpy() if hasattr(t, "to_numpy") else np.asarray(t)


def _ref(x, W, K):
    z = x @ W["router"].T + W["qb"]
    idx = np.argsort(-z, axis=1, kind="stable")[:, :K]
    sel = np.take_along_axis(z, idx, axis=1)
    sel = sel - sel.max(axis=1, keepdims=True)
    p = np.exp(sel)
    p /= p.sum(axis=1, keepdims=True)
    out = np.zeros_like(x)
    for k in range(K):
        for e in range(W["gate"].__len__()):
            m = idx[:, k] == e
            if not m.any():
                continue
            xe = x[m]
            h = xe @ W["gate"][e].T
            h = (h / (1 + np.exp(-h))) * (xe @ W["up"][e].T)
            out[m] += p[m, k:k + 1] * (h @ W["down"][e].T)
    return out


def _weights(rng, n, D, FF, E):
    return dict(router=rng.normal(0, .1, (E, D)).astype(np.float32),
                qb=rng.normal(0, .5, (E,)).astype(np.float32),
                gate=[rng.normal(0, .1, (FF, D)).astype(np.float32) for _ in range(E)],
                up=[rng.normal(0, .1, (FF, D)).astype(np.float32) for _ in range(E)],
                down=[rng.normal(0, .1, (D, FF)).astype(np.float32) for _ in range(E)])


def _build(D, FF, E, K, W):
    m = MoE(D, FF, E, top_k=K, gate="softmax_topk", qb_bias=W["qb"])
    m.router.weight.data[...] = W["router"]
    for e in range(E):
        m.experts[e].gate.weight.data[...] = W["gate"][e]
        m.experts[e].up.weight.data[...] = W["up"][e]
        m.experts[e].down.weight.data[...] = W["down"][e]
    return m


def test_moe_matches_engine_semantics():
    rng = np.random.default_rng(0)
    n, D, FF, E, K = 16, 64, 128, 8, 2
    x = rng.normal(0, .3, (n, D)).astype(np.float32)
    W = _weights(rng, n, D, FF, E)
    m = _build(D, FF, E, K, W)
    got = _np(m.forward(x))
    ref = _ref(x, W, K)
    d = np.abs(got - ref).max()
    assert d < 1e-5, "MoE 与引擎语义不一致：max|Δ|=%.3e" % d


def test_no_qb_bias_is_unchanged():
    """老模型（无 qb_bias）行为不能被这次改动影响。"""
    rng = np.random.default_rng(1)
    D, FF, E, K, n = 32, 64, 4, 2, 8
    W = _weights(rng, n, D, FF, E)
    x = rng.normal(0, .3, (n, D)).astype(np.float32)
    W0 = dict(W, qb=np.zeros(E, np.float32))
    a = _np(_build(D, FF, E, K, W0).forward(x))
    b = np.asarray(MoE(D, FF, E, top_k=K, gate="softmax_topk").forward(x)) if False else None
    # 用 qb=0 与不传 qb_bias 各跑一次，但权重要一致 ⇒ 手动构建无 bias 版本
    m2 = MoE(D, FF, E, top_k=K, gate="softmax_topk")
    m2.router.weight.data[...] = W["router"]
    for e in range(E):
        m2.experts[e].gate.weight.data[...] = W["gate"][e]
        m2.experts[e].up.weight.data[...] = W["up"][e]
        m2.experts[e].down.weight.data[...] = W["down"][e]
    got = _np(m2.forward(x))
    assert np.abs(got - a).max() < 1e-6, "qb_bias=None 与 qb_bias=0 应当等价"