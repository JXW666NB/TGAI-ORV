# -*- coding: utf-8 -*-
"""`masked_fill` 在 CUDA 上的正确性（0.9.23 修复）。

原实现：
    out = Tensor(np.where(m, value, self.data), ...)
CUDA 张量的 `self.data` 是 **None**，于是 `np.where` 产出 object 数组，
最终**静默变成 nan** —— 既不报错也不算对。这类"静默算错"比崩溃危险得多。

修复方式：CUDA 上改用组合实现 `x*(1-m) + v*m`，梯度自动只在未掩码处流。
"""
import numpy as np
import pytest

import tgai_go.train as T


def _np(x):
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, np.float32)


def _has_cuda():
    try:
        rt = T._cuda()
        return bool(rt is not None and rt.free_device_bytes() is not None)
    except Exception:
        return False


CUDA = pytest.mark.skipif(not _has_cuda(), reason="没有可用 CUDA 设备")


@pytest.fixture(autouse=True)
def _cpu_default():
    T.set_default_device("cpu")
    yield
    T.set_default_device("cpu")


def _sample():
    x = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], np.float32)
    m = np.array([[True, False, False], [False, True, False]])
    return x, m


def test_masked_fill_cpu_values_and_grad():
    x_np, m = _sample()
    x = T.tensor(x_np, requires_grad=True)
    y = x.masked_fill(m, 0.0)
    assert np.array_equal(_np(y), np.where(m, 0.0, x_np))
    (y * T.tensor(np.ones_like(x_np))).sum().backward()
    # 梯度只在未掩码处为 1
    assert np.array_equal(_np(x.grad), np.where(m, 0.0, 1.0))


@CUDA
def test_masked_fill_cuda_is_not_nan():
    """核心回归：CUDA 上不能是 nan。"""
    x_np, m = _sample()
    T.set_default_device("cuda")
    y = T.tensor(x_np).masked_fill(m, 0.0)
    got = _np(y)
    T.set_default_device("cpu")
    assert np.all(np.isfinite(got)), "CUDA 上 masked_fill 产出了非有限值（旧 BUG 回归）"
    assert np.array_equal(got, np.where(m, 0.0, x_np)), got


@CUDA
def test_masked_fill_cuda_matches_cpu_and_grad():
    x_np, m = _sample()
    T.set_default_device("cpu")
    ref = _np(T.tensor(x_np).masked_fill(m, 0.0))

    T.set_default_device("cuda")
    x = T.tensor(x_np, requires_grad=True)
    y = x.masked_fill(m, 0.0)
    (y * T.tensor(np.ones_like(x_np))).sum().backward()
    got, g = _np(y), _np(x.grad)
    T.set_default_device("cpu")

    assert np.allclose(got, ref, atol=0, rtol=0), got
    assert np.array_equal(g, np.where(m, 0.0, 1.0)), g


@CUDA
def test_masked_fill_cuda_nonzero_value():
    x_np, m = _sample()
    T.set_default_device("cuda")
    got = _np(T.tensor(x_np).masked_fill(m, -1e4))
    T.set_default_device("cpu")
    assert np.array_equal(got, np.where(m, -1e4, x_np)), got
