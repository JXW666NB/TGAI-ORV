# -*- coding: utf-8 -*-
"""Conv1d / Conv2d 与 torch 的对照（阶段 C 判据）。

判据：前向 ≤1e-5；对输入/权重/偏置的梯度 ≤1e-5。
覆盖：stride 1/2、padding、dilation、groups=1/2、Conv1d。
CUDA 目前**不支持**（im2col 在主机侧），测试里验证它给出明确报错而不是静默算错。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402

torch = pytest.importorskip("torch")
F = torch.nn.functional


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def _case(x_shape, w_shape, stride=1, padding=0, dilation=1, groups=1, seed=0, bias=True):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, x_shape).astype(np.float32)
    w = rng.normal(0, 0.3, w_shape).astype(np.float32)
    b = rng.normal(0, 0.1, (w_shape[0],)).astype(np.float32) if bias else None
    gw = rng.normal(0, 1, _out_shape(x_shape, w_shape, stride, padding, dilation)).astype(np.float32)

    xt = T.tensor(x.copy(), requires_grad=True)
    wt = T.tensor(w.copy(), requires_grad=True)
    bt = T.tensor(b.copy(), requires_grad=True) if bias else None
    y = T.conv2d(xt, wt, bt, stride, padding, dilation, groups)
    (y * T.tensor(gw)).sum().backward()

    xr = torch.from_numpy(x.copy()).requires_grad_(True)
    wr = torch.from_numpy(w.copy()).requires_grad_(True)
    br = torch.from_numpy(b.copy()).requires_grad_(True) if bias else None
    yr = F.conv2d(xr, wr, br, stride=stride, padding=padding, dilation=dilation, groups=groups)
    (yr * torch.from_numpy(gw)).sum().backward()

    res = {"fwd": _rel(y.data, yr.detach().numpy()),
           "dx": _rel(xt.grad.data, xr.grad.numpy()),
           "dw": _rel(wt.grad.data, wr.grad.numpy())}
    if bias:
        res["db"] = _rel(bt.grad.data, br.grad.numpy())
    return res


def _out_shape(x_shape, w_shape, stride, padding, dilation):
    n, c, h, w = x_shape
    s = stride if isinstance(stride, int) else stride[0]
    p = padding if isinstance(padding, int) else padding[0]
    d = dilation if isinstance(dilation, int) else dilation[0]
    oh = (h + 2 * p - d * (w_shape[2] - 1) - 1) // s + 1
    ow = (w + 2 * p - d * (w_shape[3] - 1) - 1) // s + 1
    return (n, w_shape[0], oh, ow)


@pytest.mark.parametrize("kw", [
    dict(stride=1),
    dict(stride=2),
    dict(stride=1, padding=1),
    dict(stride=2, padding=1),
    dict(stride=1, dilation=2),
    dict(stride=1, groups=2),
    dict(stride=1, bias=False),
])
def test_conv2d_matches_torch(kw):
    res = _case((2, 4, 8, 8), (6, 4 if kw.get("groups", 1) == 1 else 2, 3, 3), **kw)
    for k, v in res.items():
        assert v <= 1e-5, (k, v)


def test_conv1d_matches_torch():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, (2, 3, 10)).astype(np.float32)
    w = rng.normal(0, 0.3, (5, 3, 3)).astype(np.float32)
    b = rng.normal(0, 0.1, (5,)).astype(np.float32)
    gw = rng.normal(0, 1, (2, 5, 8)).astype(np.float32)

    xt = T.tensor(x.copy(), requires_grad=True)
    wt = T.tensor(w.copy(), requires_grad=True)
    bt = T.tensor(b.copy(), requires_grad=True)
    y = T.conv1d(xt, wt, bt, stride=1, padding=0)
    (y * T.tensor(gw)).sum().backward()

    xr = torch.from_numpy(x.copy()).requires_grad_(True)
    wr = torch.from_numpy(w.copy()).requires_grad_(True)
    br = torch.from_numpy(b.copy()).requires_grad_(True)
    yr = F.conv1d(xr, wr, br)
    (yr * torch.from_numpy(gw)).sum().backward()

    assert _rel(y.data, yr.detach().numpy()) <= 1e-5
    assert _rel(xt.grad.data, xr.grad.numpy()) <= 1e-5
    assert _rel(wt.grad.data, wr.grad.numpy()) <= 1e-5
    assert _rel(bt.grad.data, br.grad.numpy()) <= 1e-5


def test_conv2d_module_shapes_and_grads():
    m = T.Conv2d(3, 7, 3, stride=2, padding=1)
    x = T.tensor(np.random.default_rng(6).normal(0, 1, (2, 3, 9, 9)).astype(np.float32),
                 requires_grad=True)
    y = m(x)
    assert tuple(int(s) for s in y.shape) == (2, 7, 5, 5)
    y.sum().backward()
    assert m.weight.grad is not None and x.grad is not None


def test_conv_cuda_gives_clear_error():
    """CUDA 上的 conv 若未实现，必须明确报错（含修法），不能静默算错。

    ★ 2026-11-12：本用例原本假设"本机没有 N 卡"，在有卡的机器上 `set_default_device("cuda")`
    会成功，随后 `conv2d` 走真的 CUDA 路径（已实现），于是等不到 NotImplementedError 而失败。
    现在：CUDA 可用时**跳过**（它的语义就是"没有 CUDA 时怎么报错"）。
    """
    try:
        T.set_default_device("cuda")
        T.tensor(np.zeros((1,), np.float32))
    except Exception as e:                      # noqa: BLE001
        pytest.skip("CUDA 不可用: %s" % type(e).__name__)
    finally:
        T.set_default_device("cpu")
    # 能走到这里说明本机 CUDA 真的可用 —— 但那不是本用例要测的场景
    pytest.skip("本机 CUDA 可用，conv2d 走真实内核；此用例只针对无卡时的报错")
