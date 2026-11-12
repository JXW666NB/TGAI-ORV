# -*- coding: utf-8 -*-
"""`x[idx]` 数组索引：CPU 与 CUDA 都要与 numpy 一致，反向要处理重复下标。

## 为什么重复下标是重点

embedding 查表时同一个 token 会出现多次，反向必须**累加**而不是覆盖。
CPU 路径用 `np.add.at`，CUDA 路径也是主机 `np.add.at` —— 两条路径必须给出同样的梯度。
这一条如果错了，模型能训但会学偏，而且不报错。
"""
import numpy as np
import pytest

import tgai_go.train as T


def _has_cuda():
    try:
        T._cuda()
        return True
    except Exception:      # noqa: BLE001
        return False


def _mat(seed=0, n=6, d=4):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, d)).astype(np.float32)


def _grad_np(t):
    """统一取梯度：CPU 上是 ndarray，CUDA 上可能是 Tensor（要 to_numpy）。"""
    g = t.grad
    if g is None:
        raise AssertionError("梯度为 None")
    if hasattr(g, "to_numpy"):
        return np.asarray(g.to_numpy(), np.float32)
    return np.asarray(g, np.float32)


def test_cpu_array_index_matches_numpy():
    T.set_default_device("cpu")
    x = _mat(1)
    idx = np.array([0, 2, 2, 5], np.int64)
    got = np.asarray(T.tensor(x)[idx].to_numpy(), np.float32)
    assert np.array_equal(got, x[idx])


def test_cpu_array_index_negative_and_list():
    T.set_default_device("cpu")
    x = _mat(2)
    got = np.asarray(T.tensor(x)[[-1, 0]].to_numpy(), np.float32)
    assert np.array_equal(got, x[[-1, 0]])


def test_cpu_array_index_backward_accumulates_duplicates():
    """同一个下标出现两次，梯度必须是 2，不是 1。"""
    T.set_default_device("cpu")
    x = _mat(3, n=5)
    idx = np.array([1, 1, 3], np.int64)
    t = T.tensor(x)
    t.requires_grad = True
    t[idx].sum().backward()
    g = _grad_np(t)
    want = np.zeros_like(x)
    want[1] = 2.0
    want[3] = 1.0
    assert np.array_equal(g, want)


def test_array_index_out_of_range_raises_cpu():
    T.set_default_device("cpu")
    x = T.tensor(_mat(4, n=3))
    with pytest.raises(IndexError):
        x[np.array([0, 5], np.int64)]


def test_1d_source_index():
    T.set_default_device("cpu")
    x = np.arange(7, dtype=np.float32)
    got = np.asarray(T.tensor(x)[[3, 1]].to_numpy(), np.float32)
    assert np.array_equal(got, x[[3, 1]])


@pytest.mark.skipif(not _has_cuda(), reason="需要 CUDA")
def test_cuda_array_index_matches_cpu():
    T.set_default_device("cpu")
    x = _mat(5, n=8, d=6)
    idx = np.array([7, 0, 0, 3], np.int64)
    cpu = np.asarray(T.tensor(x)[idx].to_numpy(), np.float32)
    T.set_default_device("cuda")
    try:
        gpu = np.asarray(T.tensor(x)[idx].to_numpy(), np.float32)
    finally:
        T.set_default_device("cpu")
    assert np.array_equal(cpu, gpu)
    assert np.array_equal(gpu, x[idx])


@pytest.mark.skipif(not _has_cuda(), reason="需要 CUDA")
def test_cuda_array_index_backward_matches_cpu():
    T.set_default_device("cpu")
    x = _mat(6, n=6, d=3)
    idx = np.array([2, 2, 4], np.int64)
    t = T.tensor(x)
    t.requires_grad = True
    t[idx].sum().backward()
    cpu_g = _grad_np(t)

    T.set_default_device("cuda")
    try:
        t2 = T.tensor(x)
        t2.requires_grad = True
        t2[idx].sum().backward()
        gpu_g = _grad_np(t2)
    finally:
        T.set_default_device("cpu")
    assert np.abs(cpu_g - gpu_g).max() == 0.0


@pytest.mark.skipif(not _has_cuda(), reason="需要 CUDA")
def test_cuda_array_index_out_of_range_raises():
    T.set_default_device("cuda")
    try:
        x = T.tensor(_mat(7, n=3))
        with pytest.raises(IndexError):
            x[np.array([0, 9], np.int64)]
    finally:
        T.set_default_device("cpu")
