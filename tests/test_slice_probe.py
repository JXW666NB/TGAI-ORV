# -*- coding: utf-8 -*-
"""第 0.5 步探针（docs/PLAN_FUSED_ATTN.md）：`Tensor.__getitem__` 的切片是否可用于「全宽投影 → 按头取视图」。

要回答两点：
  (a) 取值与 numpy 的切片一致；
  (b) 梯度能穿过切片回到原张量（且位置正确）。
若通过 ⇒ 第 1 步只需改投影；不通过 ⇒ 第 1 步还需要切片/批量算子。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def test_slice_values_match_numpy():
    x = np.arange(24, dtype=np.float32).reshape(6, 4)
    t = T.tensor(x)
    try:
        s = t[:, 1:3]
    except Exception as e:
        pytest.fail("Tensor 切片不可用：%s: %s。修法：改造需要用别的取视图方式（reshape+批量算子）"
                    % (type(e).__name__, str(e)[:120]))
    got = np.asarray(s.data, np.float32)
    assert _rel(got, x[:, 1:3]) <= 1e-6, (got.shape, x[:, 1:3].shape)


def test_slice_backward_lands_in_the_right_columns():
    x = np.arange(24, dtype=np.float32).reshape(6, 4)
    w = np.arange(12, dtype=np.float32).reshape(6, 2) + 1.0
    t = T.tensor(x.copy(), requires_grad=True)
    try:
        s = t[:, 1:3]
    except Exception as e:
        pytest.skip("切片不可用，跳过反向检查（上一条测试已报）：%s" % type(e).__name__)
    (s * T.tensor(w)).sum().backward()
    g = np.asarray(t.grad.data, np.float32)
    expect = np.zeros_like(x)
    expect[:, 1:3] = w
    assert _rel(g, expect) <= 1e-6, (g, expect)
