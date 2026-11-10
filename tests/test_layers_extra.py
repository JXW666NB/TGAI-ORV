# -*- coding: utf-8 -*-
"""补充层与 torch 对照：Dropout / MaxPool / AvgPool / BatchNorm（1d/2d）。

判据：前向与反向相对差 ≤1e-5（fp32 求和顺序差异），Dropout 用统计性质 + eval 恒等。
CUDA 部分另有一组用例（有卡才跑）。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train.layers_extra import (AvgPool1d, AvgPool2d, BatchNorm1d, BatchNorm2d,
                                        Dropout, MaxPool1d, MaxPool2d)


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
torch = pytest.importorskip("torch")


@pytest.mark.parametrize("k,s,p", [(3, None, 0), (2, 2, 0), (3, 2, 1)])
def test_maxpool2d_matches_torch(k, s, p):
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (2, 3, 6, 6)).astype(np.float32)
    m = MaxPool2d(k, stride=s, padding=p)
    xt = T.tensor(x, requires_grad=True)
    y = m(xt)
    ref = torch.nn.MaxPool2d(k, stride=s, padding=p)(torch.from_numpy(x))
    assert _rel(_np(y), ref.detach().numpy()) <= 1e-6

    g = rng.normal(0, 1, tuple(ref.shape)).astype(np.float32)
    (y * T.tensor(g)).sum().backward()
    tr = torch.from_numpy(x).requires_grad_(True)
    out = torch.nn.MaxPool2d(k, stride=s, padding=p)(tr)
    (out * torch.from_numpy(g)).sum().backward()
    assert _rel(_np(xt.grad), tr.grad.numpy()) <= 1e-5


@pytest.mark.parametrize("k,s,p", [(3, None, 0), (2, 2, 0), (3, 2, 1), (2, 1, 1)])
def test_avgpool2d_matches_torch(k, s, p):
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (2, 3, 6, 6)).astype(np.float32)
    m = AvgPool2d(k, stride=s, padding=p)
    xt = T.tensor(x, requires_grad=True)
    y = m(xt)
    ref = torch.nn.AvgPool2d(k, stride=s, padding=p)(torch.from_numpy(x))
    assert _rel(_np(y), ref.detach().numpy()) <= 1e-6

    g = rng.normal(0, 1, tuple(ref.shape)).astype(np.float32)
    (y * T.tensor(g)).sum().backward()
    tr = torch.from_numpy(x).requires_grad_(True)
    out = torch.nn.AvgPool2d(k, stride=s, padding=p)(tr)
    (out * torch.from_numpy(g)).sum().backward()
    assert _rel(_np(xt.grad), tr.grad.numpy()) <= 1e-5


def test_pool1d_matches_torch():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (2, 3, 9)).astype(np.float32)
    for cls, tcls in ((MaxPool1d, torch.nn.MaxPool1d), (AvgPool1d, torch.nn.AvgPool1d)):
        xt = T.tensor(x, requires_grad=True)
        y = cls(3, stride=2, padding=1)(xt)
        ref = tcls(3, stride=2, padding=1)(torch.from_numpy(x))
        assert _rel(_np(y), ref.detach().numpy()) <= 1e-6


def test_batchnorm2d_matches_torch_train_and_eval():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (4, 5, 3, 3)).astype(np.float32) * 2 + 0.5
    bn = BatchNorm2d(5)
    tbn = torch.nn.BatchNorm2d(5)
    bn.train(); tbn.train()
    y = bn(T.tensor(x))
    ref = tbn(torch.from_numpy(x))
    assert _rel(_np(y), ref.detach().numpy()) <= 1e-5
    assert _rel(bn.running_mean, tbn.running_mean.numpy()) <= 1e-5
    assert _rel(bn.running_var, tbn.running_var.numpy()) <= 1e-5

    bn.eval(); tbn.eval()
    y2 = bn(T.tensor(x))
    ref2 = tbn(torch.from_numpy(x))
    assert _rel(_np(y2), ref2.detach().numpy()) <= 1e-5


def test_batchnorm2d_backward_matches_torch():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, (3, 4, 3, 3)).astype(np.float32)
    bn = BatchNorm2d(4)
    bn.train()
    xt = T.tensor(x, requires_grad=True)
    y = bn(xt)
    g = rng.normal(0, 1, tuple(y.shape)).astype(np.float32)
    (y * T.tensor(g)).sum().backward()

    tbn = torch.nn.BatchNorm2d(4)
    tbn.train()
    tr = torch.from_numpy(x).requires_grad_(True)
    out = tbn(tr)
    (out * torch.from_numpy(g)).sum().backward()
    assert _rel(_np(xt.grad), tr.grad.numpy()) <= 1e-4
    assert _rel(_np(bn.weight.grad), tbn.weight.grad.numpy()) <= 1e-4
    assert _rel(_np(bn.bias.grad), tbn.bias.grad.numpy()) <= 1e-4


def test_batchnorm1d_matches_torch():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, (6, 3)).astype(np.float32)
    bn = BatchNorm1d(3); tbn = torch.nn.BatchNorm1d(3)
    bn.train(); tbn.train()
    assert _rel(_np(bn(T.tensor(x))), tbn(torch.from_numpy(x)).detach().numpy()) <= 1e-5


def test_dropout_statistics_and_eval_identity():
    d = Dropout(0.3)
    d.train()
    x = np.ones((1000, 1000), np.float32)
    y = _np(d(T.tensor(x)))
    zero_ratio = float((y == 0).mean())
    assert abs(zero_ratio - 0.3) < 0.01, zero_ratio
    assert abs(float(y.mean()) - 1.0) < 0.01, float(y.mean())
    d.eval()
    assert np.array_equal(_np(d(T.tensor(x))), x)


def test_dropout_rejects_bad_p():
    with pytest.raises(ValueError):
        Dropout(1.0)


@CUDA
def test_pool_and_bn_on_cuda_match_cpu():
    import numpy as np
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, (2, 3, 6, 6)).astype(np.float32)
    T.set_default_device("cuda")
    for cls in (MaxPool2d, AvgPool2d):
        m = cls(3, stride=2, padding=1)
        yc = _np(m(T.tensor(x)))
        T.set_default_device("cpu")
        ycpu = _np(m(T.tensor(x)))
        T.set_default_device("cuda")
        assert _rel(yc, ycpu) <= 1e-6
    bn = BatchNorm2d(3)
    yc = _np(bn(T.tensor(x)))
    T.set_default_device("cpu")
    bn2 = BatchNorm2d(3)
    ycpu = _np(bn2(T.tensor(x)))
    T.set_default_device("cuda")
    assert _rel(yc, ycpu) <= 1e-5
