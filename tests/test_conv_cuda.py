# -*- coding: utf-8 -*-
"""CUDA conv2d 契约：有 GPU 时与 torch 对齐（≤1e-5）；无 GPU 时跳过。

判据来源：阶段 C2 要求"前向与输入/权重/偏置梯度 ≤1e-5，CPU 与 GPU 各测"。
实测（V100，6 组配置含 padding/stride=2/dilation=2）最差 **2.39e-07**。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train.conv import conv2d


def _has_cuda():
    try:
        rt = T._cuda()
        return bool(rt is not None and getattr(rt, "active", True) and rt.free_device_bytes() is not None)
    except Exception:
        return False


CUDA = pytest.mark.skipif(not _has_cuda(), reason="没有可用 CUDA 设备")


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(1e-9, float(np.abs(b).max())))


@CUDA
@pytest.mark.parametrize("cfg", [
    (2, 3, 5, 6, 3, 3, 1, 0, 1),
    (1, 2, 6, 6, 3, 3, 1, 1, 1),
    (2, 2, 7, 7, 3, 3, 2, 1, 1),
    (1, 1, 9, 9, 3, 3, 1, 2, 2),
])
def test_conv2d_cuda_matches_torch(cfg):
    torch = pytest.importorskip("torch")
    N, C, H, W, kh, kw, s, p, d = cfg
    cout = 4
    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    xn = rng.normal(0, 1, (N, C, H, W)).astype(np.float32)
    wn = rng.normal(0, 0.2, (cout, C, kh, kw)).astype(np.float32)
    bn = rng.normal(0, 0.1, (cout,)).astype(np.float32)

    x = T.tensor(xn, requires_grad=True)
    w = T.tensor(wn, requires_grad=True)
    b = T.tensor(bn, requires_grad=True)
    y = conv2d(x, w, b, stride=s, padding=p, dilation=d)
    yn = np.asarray(y.to_numpy(), np.float32)
    gw = rng.normal(0, 1, yn.shape).astype(np.float32)
    (y * T.tensor(gw)).sum().backward()
    T._cuda().sync()

    tx = torch.from_numpy(xn).requires_grad_(True)
    tw = torch.from_numpy(wn).requires_grad_(True)
    tb = torch.from_numpy(bn).requires_grad_(True)
    ty = torch.nn.functional.conv2d(tx, tw, tb, stride=s, padding=p, dilation=d)
    ty.backward(torch.from_numpy(gw))

    assert _rel(yn, ty.detach().numpy()) <= 1e-5
    assert _rel(x.grad.to_numpy(), tx.grad.numpy()) <= 1e-5
    assert _rel(w.grad.to_numpy(), tw.grad.numpy()) <= 1e-5
    assert _rel(b.grad.to_numpy(), tb.grad.numpy()) <= 1e-5


@CUDA
@pytest.mark.parametrize("groups", [2, 4])
def test_conv2d_cuda_groups_matches_torch(groups):
    """组卷积（0.9.17）：按组切片 + 单组卷积 + 通道拼接，与 torch 对照。

    组卷积在数学上就是"各组独立卷积再拼通道"，所以复用已验证的单组路径即可。
    """
    torch = pytest.importorskip("torch")
    N, C, H, W, kh, kw = 2, 4, 6, 6, 3, 3
    cout = 4
    T.set_default_device("cuda")
    rng = np.random.default_rng(groups)
    xn = rng.normal(0, 1, (N, C, H, W)).astype(np.float32)
    wn = rng.normal(0, 0.2, (cout, C // groups, kh, kw)).astype(np.float32)
    bn = rng.normal(0, 0.1, (cout,)).astype(np.float32)

    x = T.tensor(xn, requires_grad=True)
    w = T.tensor(wn, requires_grad=True)
    b = T.tensor(bn, requires_grad=True)
    y = conv2d(x, w, b, stride=1, padding=1, groups=groups)
    yn = np.asarray(y.to_numpy(), np.float32)
    gw = rng.normal(0, 1, yn.shape).astype(np.float32)
    (y * T.tensor(gw)).sum().backward()
    T._cuda().sync()

    tx = torch.from_numpy(xn).requires_grad_(True)
    tw = torch.from_numpy(wn).requires_grad_(True)
    tb = torch.from_numpy(bn).requires_grad_(True)
    ty = torch.nn.functional.conv2d(tx, tw, tb, stride=1, padding=1, groups=groups)
    ty.backward(torch.from_numpy(gw))

    assert _rel(yn, ty.detach().numpy()) <= 1e-5
    assert _rel(x.grad.to_numpy(), tx.grad.numpy()) <= 1e-5
    assert _rel(w.grad.to_numpy(), tw.grad.numpy()) <= 1e-5
    assert _rel(b.grad.to_numpy(), tb.grad.numpy()) <= 1e-5
