# -*- coding: utf-8 -*-
"""契约：只有叶子张量保留 `.grad`，中间节点不保留（PyTorch 语义）。

为什么重要：我们的反向原本给**每个**节点都建 `zeros_like` 再累加，
等于在 GPU 上又多存了一份全尺寸梯度（子智能体台账：中间梯度 11.23 GiB）。
改成"消费完即释放、只给叶子留 .grad"后，B=4/S=512 的峰值从 29.92 → 19.97 GiB（开检查点）。
这条测试防止以后有人"顺手"把中间梯度又挂回去。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

import tgai_go.train as T


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


def test_only_leaves_keep_grad():
    x = T.tensor(np.array([[1.0, 2.0], [3.0, 4.0]], np.float32), requires_grad=True)
    mid1 = x * 2.0            # 中间节点
    mid2 = mid1 * mid1        # 中间节点
    out = mid2.sum()
    out.backward()

    assert x.grad is not None, "叶子必须保留 .grad（对外契约）"
    assert np.allclose(np.asarray(x.grad, np.float64), [[8.0, 16.0], [24.0, 32.0]])
    assert mid1.grad is None, "中间节点不该保留梯度（会白占一份全尺寸内存）"
    assert mid2.grad is None


def test_grad_values_unchanged_with_shared_parent():
    """fan-out（一个节点被多处使用）时，梯度必须正确累加 —— 释放逻辑不能误删。"""
    x = T.tensor(np.array([3.0], np.float32), requires_grad=True)
    a = x * 2.0
    b = x * 5.0
    out = (a * b).sum()          # d/dx = 2*5*x + 5*2*x = 20x = 60
    out.backward()
    assert np.allclose(np.asarray(x.grad, np.float64), [60.0])
