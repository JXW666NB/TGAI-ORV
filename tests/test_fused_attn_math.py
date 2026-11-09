# -*- coding: utf-8 -*-
"""验证「per-head 投影 → 全宽 GEMM」的数学等价性（CPU，无需 GPU）。

为什么先做这个：改造 `V27Trainer.fwd` 时最大的风险不是性能，而是
**拼接顺序 / reshape 顺序 / 梯度回拆** 悄悄错位 —— 之前 MoE softmax 作用域、causal mask
两次事故都是"看起来等价"造成的。这里用最小例子把这两件事钉死：

1. 前向：`cat_h(h @ Wh.T)` == `h @ cat_h(Wh).T` 再 reshape 回 [T, NH, DH]
2. 反向：对全宽权重求得的梯度，按 head 切开后，与逐 head 求得的梯度**逐位相同**
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402

DT, NH, DH = 32, 4, 8          # d_model / heads / head_dim
SEQ = 5


def _weights(seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 0.2, (DH, DT)).astype(np.float32) for _ in range(NH)]


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def test_concat_order_matches_per_head_loop():
    ws = _weights()
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (SEQ, DT)).astype(np.float32)

    # 逐 head
    xt = T.tensor(x.copy(), requires_grad=True)
    hs = []
    for h in range(NH):
        w = T.tensor(ws[h].copy(), requires_grad=True)
        hs.append(xt @ w.T)
    per_head = np.stack([np.asarray(y.data, np.float32) for y in hs], axis=1)   # [T, NH, DH]

    # 全宽：按 head 顺序 concat（[NH*DH, DT]）
    wf = np.concatenate(ws, axis=0)
    xf = T.tensor(x.copy(), requires_grad=True)
    wft = T.tensor(wf.copy(), requires_grad=True)
    fused = np.asarray((xf @ wft.T).data, np.float32).reshape(SEQ, NH, DH)

    assert _rel(fused, per_head) <= 1e-6, _rel(fused, per_head)


def test_gradient_splits_back_to_per_head():
    ws = _weights(2)
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (SEQ, DT)).astype(np.float32)
    gw = rng.normal(0, 1, (SEQ, NH, DH)).astype(np.float32)

    # 逐 head 的梯度
    grads_head = []
    for h in range(NH):
        w = T.tensor(ws[h].copy(), requires_grad=True)
        y = T.tensor(x.copy()) @ w.T
        (y * T.tensor(gw[:, h, :])).sum().backward()
        grads_head.append(np.asarray(w.grad.data, np.float32).copy())

    # 全宽一次 GEMM 后再切
    wf = np.concatenate(ws, axis=0)
    wft = T.tensor(wf.copy(), requires_grad=True)
    yf = T.tensor(x.copy()) @ wft.T
    (yf.reshape((SEQ, NH, DH)) * T.tensor(gw)).sum().backward()
    g_full = np.asarray(wft.grad.data, np.float32)
    for h in range(NH):
        part = g_full[h * DH:(h + 1) * DH]
        r = _rel(part, grads_head[h])
        assert r <= 1e-6, (h, r)


def test_input_gradient_matches():
    ws = _weights(4)
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, (SEQ, DT)).astype(np.float32)
    gw = rng.normal(0, 1, (SEQ, NH, DH)).astype(np.float32)

    xa = T.tensor(x.copy(), requires_grad=True)
    acc = None
    for h in range(NH):
        w = T.tensor(ws[h].copy())
        y = xa @ w.T
        (y * T.tensor(gw[:, h, :])).sum().backward()
    g_head = np.asarray(xa.grad.data, np.float32).copy()

    xb = T.tensor(x.copy(), requires_grad=True)
    wft = T.tensor(np.concatenate(ws, axis=0).copy())
    yf = xb @ wft.T
    (yf.reshape((SEQ, NH, DH)) * T.tensor(gw)).sum().backward()
    g_fused = np.asarray(xb.grad.data, np.float32).copy()

    assert _rel(g_fused, g_head) <= 1e-6, _rel(g_fused, g_head)
