# -*- coding: utf-8 -*-
"""`ops_plugin.register_auto`：只写前向，反向自动导出。

核心机制：训练时**前向仍走 forward 快路径**，反向把数学等价的 fallback 重算一遍再反传，
因此梯度与 fallback 的数学**逐位一致**（不是有限差分）。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import ops_plugin as O


def _np_grad(t):
    """取标量/张量的梯度（CPU 后端里 .grad 是 ndarray，CUDA 上是缓冲）。"""
    g = t.grad
    if g is None:
        return None
    if hasattr(g, "to_numpy"):
        return np.asarray(g.to_numpy(), np.float32)
    return np.asarray(g, np.float32)


def test_forward_matches_reference():
    O.register_auto("ra_sp", lambda x: np.log1p(np.exp(x)),
                    lambda x: (1.0 + x.exp()).log(), replace=True)
    v = np.array([-2.0, -0.5, 0.0, 0.7, 2.0], np.float32)
    x = T.from_numpy(v)
    got = T.functional.ra_sp(x).to_numpy()
    assert np.allclose(got, np.log1p(np.exp(v)), atol=1e-6)


def test_backward_matches_analytic_formula():
    """d/dx [softplus(x)^2] = 2*softplus(x)*sigmoid(x)。"""
    O.register_auto("ra_sp2", lambda x: np.log1p(np.exp(x)),
                    lambda x: (1.0 + x.exp()).log(), replace=True)
    v = np.array([-2.0, -0.5, 0.0, 0.7, 2.0], np.float32)
    x = T.from_numpy(v)
    x.requires_grad = True
    y = T.functional.ra_sp2(x)
    (y * y).sum().backward()
    sp = np.log1p(np.exp(v))
    want = 2.0 * v * 0.0 + 2.0 * sp * (1.0 / (1.0 + np.exp(-v)))
    assert np.allclose(_np_grad(x), want, atol=1e-5)


def test_no_grad_path_does_not_build_graph():
    calls = {"n": 0}

    def fb(x):
        calls["n"] += 1
        return x * 1.0

    O.register_auto("ra_ng", lambda x: x * 2.0, fb, replace=True)
    v = np.ones((3,), np.float32)
    with T.no_grad():
        y = T.functional.ra_ng(T.from_numpy(v))
    assert y.requires_grad is False
    assert calls["n"] == 0, "no_grad 下不该走 fallback（说明快路径没生效）"
    assert np.allclose(y.to_numpy(), v * 2.0)


def test_consistency_check_catches_mismatched_fallback():
    # fallback 故意算成另一个函数：必须在首次反向时被 check 抓到
    O.register_auto("ra_bad", lambda x: np.exp(x), lambda x: x * 2.0,
                    check=True, replace=True)
    x = T.from_numpy(np.array([1.0, 2.0], np.float32))
    x.requires_grad = True
    y = T.functional.ra_bad(x)
    with pytest.raises(ValueError) as ei:
        y.sum().backward()
    assert "不一致" in str(ei.value)


def test_binary_op_with_broadcast():
    """多输入 + 广播：dx = g*y, dy = g*x（用 fallback 的 autograd 得到）。"""
    O.register_auto("ra_mul", lambda a, b: a * b, lambda a, b: a * b, replace=True)
    a = T.from_numpy(np.array([[1.0], [2.0]], np.float32))
    b = T.from_numpy(np.array([[3.0, 4.0]], np.float32))
    a.requires_grad = True
    b.requires_grad = True
    y = T.functional.ra_mul(a, b)
    y.sum().backward()
    assert np.allclose(_np_grad(a), np.array([[7.0], [7.0]], np.float32), atol=1e-6)
    assert np.allclose(_np_grad(b), np.array([[3.0, 3.0]], np.float32), atol=1e-6)


def test_native_forward_used_in_training():
    """训练路径也必须用 forward（不是 fallback），否则就白丢原生速度。"""
    hit = {"fwd": 0, "fb": 0}

    def fwd(x):
        hit["fwd"] += 1
        return np.log1p(np.exp(x))

    def fb(x):
        hit["fb"] += 1
        return (1.0 + x.exp()).log()

    O.register_auto("ra_count", fwd, fb, replace=True)
    x = T.from_numpy(np.array([0.3, 0.4], np.float32))
    x.requires_grad = True
    y = T.functional.ra_count(x)
    assert hit["fwd"] == 1 and hit["fb"] == 0, "前向应当走 forward 快路径"
    y.sum().backward()
    assert hit["fb"] >= 1, "反向应当走 fallback"
