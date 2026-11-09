# -*- coding: utf-8 -*-
"""LayerNorm 与 torch 的对照 + 与互操作打通（补阶段 E 的 LayerNorm 缺口）。

判据：前向 ≤1e-5；对输入 / gamma / beta 的梯度 ≤1e-5；`from_torch`/`to_torch` 往返逐位相同。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402

torch = pytest.importorskip("torch")
tn = torch.nn


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def test_layernorm_forward_and_grads_match_torch():
    rng = np.random.default_rng(0)
    D = 16
    x = rng.normal(0, 1, (4, D)).astype(np.float32)
    w = rng.normal(1, 0.1, (D,)).astype(np.float32)
    b = rng.normal(0, 0.1, (D,)).astype(np.float32)

    ln = T.nn.LayerNorm(D, eps=1e-5)
    np.copyto(np.asarray(ln.weight.data, np.float32), w)
    np.copyto(np.asarray(ln.bias.data, np.float32), b)
    xt = T.tensor(x.copy(), requires_grad=True)
    y = ln(xt)
    y.sum().backward()

    ref = tn.LayerNorm(D, eps=1e-5)
    with torch.no_grad():
        ref.weight.copy_(torch.from_numpy(w))
        ref.bias.copy_(torch.from_numpy(b))
    xr = torch.from_numpy(x.copy()).requires_grad_(True)
    yr = ref(xr)
    yr.sum().backward()

    assert _rel(y.data, yr.detach().numpy()) <= 1e-5
    assert _rel(xt.grad.data, xr.grad.numpy()) <= 1e-5
    assert _rel(ln.weight.grad.data, ref.weight.grad.numpy()) <= 1e-5
    assert _rel(ln.bias.grad.data, ref.bias.grad.numpy()) <= 1e-5


def test_layernorm_without_affine_matches_torch():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (3, 8)).astype(np.float32)
    ours = np.asarray(T.nn.LayerNorm(8, elementwise_affine=False)(T.tensor(x)).data, np.float32)
    ref = tn.LayerNorm(8, elementwise_affine=False)(torch.from_numpy(x)).detach().numpy()
    assert _rel(ours, ref) <= 1e-5


def test_layernorm_interop_round_trip_bits():
    torch.manual_seed(7)
    m = tn.Sequential(tn.Linear(6, 6), tn.LayerNorm(6, eps=1e-5), tn.Linear(6, 3))
    x = np.random.default_rng(2).normal(0, 1, (4, 6)).astype(np.float32)

    ours = T.from_torch(m)
    got = np.asarray(ours(T.tensor(x)).data, np.float32)
    want = m(torch.from_numpy(x)).detach().numpy()
    assert _rel(got, want) <= 1e-5, _rel(got, want)

    back = T.to_torch(ours)
    a, b = m.state_dict(), back.state_dict()
    assert set(a) == set(b)
    for k in a:
        assert np.array_equal(a[k].numpy().view(np.uint32), b[k].numpy().view(np.uint32)), k


def test_layernorm_rejects_bad_shape():
    with pytest.raises(ValueError, match="修法"):
        T.nn.LayerNorm(8)(T.tensor(np.zeros((2, 4), np.float32)))
