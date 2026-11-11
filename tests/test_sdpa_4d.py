# -*- coding: utf-8 -*-
"""SDPA 的 4 维（torch 写法）与 3 维（内部展平）必须数值一致。

## 为什么必须测

用户照 torch 习惯传 `[B, H, S, Dh]`。包装器会展平再还原 ——
如果还原写错（漏了某个分支、或 reshape 顺序错），**形状可能仍是 4 维但数值已经错位**，
这种 bug 会一路算下去不报错。所以用数值对拍锁住。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import functional as Fn


def _has_cuda():
    try:
        T._cuda()
        return True
    except Exception:      # noqa: BLE001
        return False


def _rand(shape, seed=0, scale=0.5):
    rng = np.random.default_rng(seed)
    return rng.normal(0, scale, shape).astype(np.float32)


def test_4d_matches_3d_cpu():
    """`[B,H,S,Dh]` 自动展平后，结果必须与手工展平的 `[B*H,S,Dh]` 完全相同。"""
    T.set_default_device("cpu")
    B, H, S, Dh = 2, 3, 5, 4
    q = _rand((B, H, S, Dh), 1)
    k = _rand((B, H, S, Dh), 2)
    v = _rand((B, H, S, Dh), 3)

    out4 = np.asarray(Fn.scaled_dot_product_attention(
        T.tensor(q), T.tensor(k), T.tensor(v), is_causal=True).to_numpy(), np.float32)
    assert out4.shape == (B, H, S, Dh), "4 维输入应返回 4 维，实际 %s" % (out4.shape,)

    qf = q.reshape(B * H, S, Dh)
    kf = k.reshape(B * H, S, Dh)
    vf = v.reshape(B * H, S, Dh)
    out3 = np.asarray(Fn.scaled_dot_product_attention(
        T.tensor(qf), T.tensor(kf), T.tensor(vf), is_causal=True).to_numpy(),
        np.float32).reshape(B, H, S, Dh)

    rel = float(np.abs(out4 - out3).max() / max(1e-9, float(np.abs(out3).max())))
    assert rel == 0.0, "4 维与 3 维结果应完全相同，相对差 %.3e" % rel


def test_3d_still_works_and_returns_3d():
    """原有 3 维用法必须完全不受影响。"""
    T.set_default_device("cpu")
    B, S, Dh = 4, 5, 8
    q = _rand((B, S, Dh), 4)
    out = Fn.scaled_dot_product_attention(
        T.tensor(q), T.tensor(q), T.tensor(q), is_causal=False)
    assert tuple(int(x) for x in out.shape) == (B, S, Dh)


def test_4d_backward_reaches_inputs():
    """4 维路径也要能反传，且梯度形状是 4 维。"""
    T.set_default_device("cpu")
    B, H, S, Dh = 2, 2, 4, 3
    q = _rand((B, H, S, Dh), 5)
    q_t = T.tensor(q)
    q_t.requires_grad = True
    k = T.tensor(_rand((B, H, S, Dh), 6))
    v = T.tensor(_rand((B, H, S, Dh), 7))
    out = Fn.scaled_dot_product_attention(q_t, k, v, is_causal=True)
    out.sum().backward()
    g = np.asarray(q_t.grad, np.float32)
    assert g.shape == (B, H, S, Dh), "梯度形状应为 4 维，实际 %s" % (g.shape,)
    assert np.isfinite(g).all()


def test_4d_rejects_mismatched_head_counts():
    """q/k/v 的 B、H 不一致时必须报错，不能算出一个错位结果。"""
    T.set_default_device("cpu")
    q = T.tensor(_rand((2, 2, 4, 8), 8))
    k = T.tensor(_rand((2, 3, 4, 8), 9))
    v = T.tensor(_rand((2, 3, 4, 8), 10))
    with pytest.raises((ValueError, RuntimeError, IndexError, AssertionError,
                        ArithmeticError, Exception)):
        Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
