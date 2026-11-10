# -*- coding: utf-8 -*-
"""归约算子与 torch 对照：max / min / var / std / norm（前向 + 反向，CPU 与 CUDA）。

判据：相对差 ≤1e-5（fp32 求和顺序）；max/min 的反向是"只落到 argmax 位置"，
与 torch 一样存在并列取第一个的约定（测试用无并列的随机数据）。
"""
import numpy as np
import pytest

import tgai_go.train as T

torch = pytest.importorskip("torch")


def _np(t):
    return np.asarray(t.to_numpy() if hasattr(t, "to_numpy") else t, np.float32)


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(1e-9, float(np.abs(b).max())))


def _has_cuda():
    try:
        rt = T._cuda()
        return bool(rt is not None and rt.free_device_bytes() is not None)
    except Exception:
        return False


CUDA = pytest.mark.skipif(not _has_cuda(), reason="没有可用 CUDA 设备")
RNG = np.random.default_rng(0)


@pytest.mark.parametrize("axis", [0, 1, 2, None])
def test_max_min_forward_and_backward(axis):
    x = RNG.normal(0, 1, (3, 4, 5)).astype(np.float32)
    for ours, theirs in ((T.Tensor.max, torch.max), (T.Tensor.min, torch.min)):
        xt = T.tensor(x, requires_grad=True)
        if axis is None:
            y = ours(xt)
            ref = theirs(torch.from_numpy(x))
        else:
            y = ours(xt, axis=axis)
            ref = theirs(torch.from_numpy(x), dim=axis).values
        assert _rel(_np(y), ref.detach().numpy()) <= 1e-6

        g = RNG.normal(0, 1, tuple(ref.shape)).astype(np.float32)
        (y * T.tensor(g)).sum().backward()
        tr = torch.from_numpy(x).requires_grad_(True)
        if axis is None:
            out = theirs(tr)
        else:
            out = theirs(tr, dim=axis).values
        (out * torch.from_numpy(g)).sum().backward()
        assert _rel(_np(xt.grad), tr.grad.numpy()) <= 1e-6


@pytest.mark.parametrize("axis", [0, 2, None])
def test_var_std_forward(axis):
    x = RNG.normal(0, 1, (3, 4, 5)).astype(np.float32) * 2 + 1
    xt = T.tensor(x)
    tx = torch.from_numpy(x)
    for ours, theirs in (("var", torch.var), ("std", torch.std)):
        got = getattr(xt, ours)(axis=axis)
        ref = theirs(tx, dim=axis) if axis is not None else theirs(tx)
        assert _rel(_np(got), ref.numpy()) <= 1e-4


def test_var_unbiased_flag():
    x = RNG.normal(0, 1, (4, 6)).astype(np.float32)
    xt = T.tensor(x)
    tx = torch.from_numpy(x)
    assert _rel(_np(xt.var(axis=0, unbiased=True)), torch.var(tx, dim=0).numpy()) <= 1e-4
    assert _rel(_np(xt.var(axis=0, unbiased=False)), torch.var(tx, dim=0, unbiased=False).numpy()) <= 1e-4


def test_norm_p2_and_reject_other_p():
    x = RNG.normal(0, 1, (3, 4)).astype(np.float32)
    xt = T.tensor(x)
    assert _rel(_np(xt.norm(axis=1)), torch.from_numpy(x).norm(dim=1).numpy()) <= 1e-5
    with pytest.raises(NotImplementedError):
        xt.norm(p=1)


def test_keepdims_shapes():
    x = RNG.normal(0, 1, (2, 3)).astype(np.float32)
    xt = T.tensor(x)
    assert tuple(int(s) for s in xt.var(axis=1, keepdims=True).shape) == (2, 1)
    assert tuple(int(s) for s in xt.max(axis=0, keepdim=True).shape) == (1, 3)


@CUDA
def test_reductions_on_cuda_match_cpu():
    x = RNG.normal(0, 1, (3, 4, 5)).astype(np.float32)
    refs = {}
    for name, kwargs in (("max", {"axis": 1}), ("min", {"axis": 2}),
                         ("var", {"axis": 0}), ("std", {"axis": 1}), ("norm", {"axis": 2})):
        T.set_default_device("cpu")
        refs[name] = _np(getattr(T.tensor(x), name)(**kwargs))
        T.set_default_device("cuda")
        got = _np(getattr(T.tensor(x), name)(**kwargs))
        T.set_default_device("cpu")
        assert _rel(got, refs[name]) <= 1e-5, name


@CUDA
def test_max_backward_on_cuda_matches_torch():
    x = RNG.normal(0, 1, (2, 3, 4)).astype(np.float32)
    T.set_default_device("cuda")
    xt = T.tensor(x, requires_grad=True)
    y = xt.max(axis=1)
    g = RNG.normal(0, 1, tuple(_np(y).shape)).astype(np.float32)
    (y * T.tensor(g)).sum().backward()
    got = _np(xt.grad)
    T.set_default_device("cpu")
    tr = torch.from_numpy(x).requires_grad_(True)
    out = tr.max(dim=1).values
    (out * torch.from_numpy(g)).sum().backward()
    assert _rel(got, tr.grad.numpy()) <= 1e-6
    T.set_default_device("cpu")
