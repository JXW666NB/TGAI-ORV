# -*- coding: utf-8 -*-
"""`nn_ops` 与 PyTorch 的数值对齐测试（纯 CPU，可选依赖 torch）。

判据：所有算子在 float32 下与对应 torch 实现 allclose(rtol=1e-5, atol=1e-6)。
没有安装 torch 时整组 skip —— CI 上仍能跑，只是少了这层对照。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import nn_ops as O

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


def _t(a):
    return T.tensor(np.ascontiguousarray(np.asarray(a, dtype=np.float32)))


def _n(x):
    return np.asarray(x.to_numpy(), dtype=np.float32)


def _both(a, rtol=1e-5, atol=1e-6):
    ref = np.asarray(a, dtype=np.float32)
    return ref


rng = np.random.default_rng(20261030)


def test_rsqrt():
    x = rng.uniform(0.5, 4.0, (3, 5)).astype(np.float32)
    got = _n(O.rsqrt(_t(x)))
    want = torch.rsqrt(torch.from_numpy(x)).numpy()
    assert np.allclose(got, want, rtol=1e-5, atol=1e-6), np.abs(got - want).max()


def test_pow_int_and_float():
    x = rng.uniform(0.5, 2.0, (4, 4)).astype(np.float32)
    for p in (2, 3, -1):
        got = _n(O.pow(_t(x), p))
        want = torch.pow(torch.from_numpy(x), p).numpy()
        assert np.allclose(got, want, rtol=1e-5, atol=1e-6), (p, np.abs(got - want).max())
    got = _n(O.pow(_t(x), 0.5))
    want = torch.pow(torch.from_numpy(x), 0.5).numpy()
    assert np.allclose(got, want, rtol=1e-4, atol=1e-6), np.abs(got - want).max()


def test_sigmoid_and_tanh():
    x = rng.normal(0, 3, (3, 7)).astype(np.float32)
    assert np.allclose(_n(O.sigmoid(_t(x))), torch.sigmoid(torch.from_numpy(x)).numpy(),
                       rtol=1e-5, atol=1e-6)
    assert np.allclose(_n(O.tanh(_t(x))), torch.tanh(torch.from_numpy(x)).numpy(),
                       rtol=1e-5, atol=1e-6)


def test_gelu_tanh_matches_torch():
    x = rng.normal(0, 2, (4, 6)).astype(np.float32)
    got = _n(O.gelu(_t(x), approximate="tanh"))
    want = torch.nn.functional.gelu(torch.from_numpy(x), approximate="tanh").numpy()
    assert np.allclose(got, want, rtol=1e-5, atol=1e-6), np.abs(got - want).max()


def test_linear_and_bias():
    x = rng.normal(0, 1, (5, 8)).astype(np.float32)
    w = rng.normal(0, 0.1, (12, 8)).astype(np.float32)   # [out, in]
    b = rng.normal(0, 0.1, (12,)).astype(np.float32)
    got = _n(O.linear(_t(x), _t(w), _t(b)))
    want = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w),
                                      torch.from_numpy(b)).numpy()
    assert np.allclose(got, want, rtol=1e-5, atol=1e-6)


def test_linear_shape_error_names_fix():
    x = _t(np.zeros((5, 8), np.float32))
    w = _t(np.zeros((12, 7), np.float32))
    with pytest.raises(ValueError) as ei:
        O.linear(x, w)
    msg = str(ei.value)
    assert "形状" in msg and "修法" in msg, msg


def test_rms_norm():
    x = rng.normal(0, 1, (6, 16)).astype(np.float32)
    w = rng.uniform(0.5, 1.5, (16,)).astype(np.float32)
    got = _n(O.rms_norm(_t(x), _t(w), eps=1e-6))
    want = torch.nn.functional.rms_norm(torch.from_numpy(x), (16,),
                                        torch.from_numpy(w), eps=1e-6).numpy()
    assert np.allclose(got, want, rtol=1e-5, atol=1e-6), np.abs(got - want).max()


def test_layer_norm():
    x = rng.normal(0, 1, (6, 16)).astype(np.float32)
    w = rng.uniform(0.5, 1.5, (16,)).astype(np.float32)
    b = rng.normal(0, 0.1, (16,)).astype(np.float32)
    got = _n(O.layer_norm(_t(x), _t(w), _t(b), eps=1e-5))
    want = torch.nn.functional.layer_norm(
        torch.from_numpy(x), (16,), torch.from_numpy(w), torch.from_numpy(b), 1e-5).numpy()
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5), np.abs(got - want).max()


def test_embedding_and_out_of_range():
    w = rng.normal(0, 1, (32, 8)).astype(np.float32)
    ids = np.array([[1, 5, 31]], dtype=np.int64)
    got = _n(O.embedding(_t(w), ids))
    want = torch.nn.functional.embedding(torch.from_numpy(ids), torch.from_numpy(w)).numpy()
    assert np.allclose(got, want, rtol=0, atol=0)
    with pytest.raises(ValueError) as ei:
        O.embedding(_t(w), np.array([99], dtype=np.int64))
    assert "越界" in str(ei.value) and "修法" in str(ei.value)


def test_stack_matches_torch():
    xs = [rng.normal(0, 1, (3, 4)).astype(np.float32) for _ in range(3)]
    for dim in (0, 1, -1):
        got = _n(O.stack([_t(x) for x in xs], dim=dim))
        want = torch.stack([torch.from_numpy(x) for x in xs], dim=dim).numpy()
        assert got.shape == want.shape and np.allclose(got, want, rtol=0, atol=0), dim


def test_split_matches_torch():
    x = rng.normal(0, 1, (10, 4)).astype(np.float32)
    for sizes in (5, [3, 7], [1, 2, 3, 4]):
        got = [ _n(p) for p in O.split(_t(x), sizes, dim=0) ]
        want = [ t.numpy() for t in torch.split(torch.from_numpy(x), sizes, dim=0) ]
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert g.shape == w.shape and np.allclose(g, w, rtol=0, atol=0)


def test_triu_matches_torch():
    x = rng.normal(0, 1, (6, 6)).astype(np.float32)
    for k in (0, 1, -1):
        got = _n(O.triu(_t(x), diagonal=k))
        want = torch.triu(torch.from_numpy(x), diagonal=k).numpy()
        assert np.allclose(got, want, rtol=0, atol=0), k


def test_where_matches_torch():
    a = rng.normal(0, 1, (4, 5)).astype(np.float32)
    b = rng.normal(0, 1, (4, 5)).astype(np.float32)
    c = (rng.uniform(0, 1, (4, 5)) > 0.5).astype(np.float32)
    got = _n(O.where(_t(c), _t(a), _t(b)))
    want = torch.where(torch.from_numpy(c) > 0, torch.from_numpy(a),
                       torch.from_numpy(b)).numpy()
    assert np.allclose(got, want, rtol=0, atol=0)


def test_clip_grad_norm_matches_torch():
    shapes = [(4, 3), (5,)]
    grads = [rng.normal(0, 1, s).astype(np.float32) for s in shapes]

    class P:
        def __init__(self, g):
            self.grad = _t(g)

    params = [P(g) for g in grads]
    norm = O.clip_grad_norm_(params, max_norm=1.0)

    tps = []
    for g in grads:
        p = torch.nn.Parameter(torch.zeros(g.shape))
        p.grad = torch.from_numpy(g.copy())
        tps.append(p)
    want_norm = torch.nn.utils.clip_grad_norm_(tps, max_norm=1.0).item()

    assert np.isclose(norm, want_norm, rtol=1e-5), (norm, want_norm)
    for p, tp in zip(params, tps):
        assert np.allclose(_n(p.grad), tp.grad.numpy(), rtol=1e-5, atol=1e-7)
