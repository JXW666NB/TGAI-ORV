# -*- coding: utf-8 -*-
"""cos / sin 算子：数值与反向都要与 numpy 一致。

P100 报告指出缺这两个算子导致 RoPE 类模型迁移成本高（要自己用 numpy 预生成表）。
这里验证新增实现：前向对 numpy、反向对解析解。
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


def _x(seed=0, shape=(64,), lo=-3.0, hi=3.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, shape).astype(np.float32)


def test_cos_forward_matches_numpy():
    T.set_default_device("cpu")
    a = _x(1)
    got = np.asarray(T.tensor(a).cos().to_numpy(), np.float32)
    ref = np.cos(a.astype(np.float64)).astype(np.float32)
    assert np.abs(got - ref).max() <= 1e-6


def test_sin_forward_matches_numpy():
    T.set_default_device("cpu")
    a = _x(2)
    got = np.asarray(T.tensor(a).sin().to_numpy(), np.float32)
    ref = np.sin(a.astype(np.float64)).astype(np.float32)
    assert np.abs(got - ref).max() <= 1e-6


def test_cos_backward_is_minus_sin():
    """d/dx cos = -sin。用 sum(cos(x)) 作损失，梯度应等于 -sin(x)。"""
    T.set_default_device("cpu")
    a = _x(3)
    t = T.tensor(a)
    t.requires_grad = True
    t.cos().sum().backward()
    got = np.asarray(t.grad, np.float32)
    ref = -np.sin(a.astype(np.float64)).astype(np.float32)
    assert np.abs(got - ref).max() <= 1e-6


def test_sin_backward_is_cos():
    """d/dx sin = cos。"""
    T.set_default_device("cpu")
    a = _x(4)
    t = T.tensor(a)
    t.requires_grad = True
    t.sin().sum().backward()
    got = np.asarray(t.grad, np.float32)
    ref = np.cos(a.astype(np.float64)).astype(np.float32)
    assert np.abs(got - ref).max() <= 1e-6


def test_cos_sin_compose_into_rope_like_rotation():
    """组合用法：y = x*cos(t) - z*sin(t) 能正常求导（RoPE 的核心形态）。"""
    T.set_default_device("cpu")
    x = T.tensor(_x(5))
    z = T.tensor(_x(6))
    th = T.tensor(_x(7))
    x.requires_grad = True
    z.requires_grad = True
    y = x * th.cos() - z * th.sin()
    y.sum().backward()
    # d y / d x = cos(t), d y / d z = -sin(t)
    a = np.asarray(th.data, np.float32)
    assert np.abs(np.asarray(x.grad, np.float32) - np.cos(a)).max() <= 1e-6
    assert np.abs(np.asarray(z.grad, np.float32) + np.sin(a)).max() <= 1e-6


@pytest.mark.skipif(not _has_cuda(), reason="需要 CUDA")
def test_cos_sin_cuda_matches_cpu():
    """设备路径必须与 CPU 路径一致（容差按 fp32 留 1e-6）。"""
    T.set_default_device("cpu")
    a = _x(8, (256,))
    cpu_c = np.asarray(T.tensor(a).cos().to_numpy(), np.float32)
    cpu_s = np.asarray(T.tensor(a).sin().to_numpy(), np.float32)
    T.set_default_device("cuda")
    try:
        gpu_c = np.asarray(T.tensor(a).cos().to_numpy(), np.float32)
        gpu_s = np.asarray(T.tensor(a).sin().to_numpy(), np.float32)
    finally:
        T.set_default_device("cpu")
    assert np.abs(cpu_c - gpu_c).max() <= 1e-6
    assert np.abs(cpu_s - gpu_s).max() <= 1e-6
