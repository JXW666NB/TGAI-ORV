# -*- coding: utf-8 -*-
"""函数式 RMSNorm 的 ORV 版与 numpy 对拍。

层的两次 norm 需要一个**不依赖模块实例**的纯函数（模块要走融合内核，签名是 `(x)->x`，
绑着参数）。这里是组合实现，正确性优先；速度差已记进 `docs/V3_PORT_STATUS.md` 的"下一步"。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import v3_ops as V  # noqa: E402


def _np(t):
    r = t.to_numpy() if hasattr(t, "to_numpy") else t
    return np.asarray(r, np.float32)


@pytest.mark.parametrize("shape", [(4, 8), (1, 16), (3, 5)])
def test_matches_numpy(shape):
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2, shape).astype(np.float32)
    w = rng.normal(1, 0.1, (shape[-1],)).astype(np.float32)
    got = _np(V.rmsnorm_orv(V.T.tensor(x), V.T.tensor(w)))
    want = V.rmsnorm_numpy(x, w)
    assert got.shape == want.shape
    err = float(np.abs(got - want).max())
    assert err < 1e-5, "RMSNorm 与参考最大误差 %.3e" % err


def test_scale_invariance():
    """RMSNorm 的定义性质：整体乘一个常数，输出**不变**（分母同步放大）。"""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (3, 8)).astype(np.float32)
    w = np.ones(8, np.float32)
    a = _np(V.rmsnorm_orv(V.T.tensor(x), V.T.tensor(w)))
    b = _np(V.rmsnorm_orv(V.T.tensor(x * 7.0), V.T.tensor(w)))
    assert np.allclose(a, b, atol=1e-5), "乘常数后输出应不变"


def test_unit_rms_after_norm():
    """未加权时，每行归一化后的均方应接近 1。"""
    rng = np.random.default_rng(2)
    x = rng.normal(0, 3, (5, 8)).astype(np.float32)
    w = np.ones(8, np.float32)
    y = _np(V.rmsnorm_orv(V.T.tensor(x), V.T.tensor(w)))
    ms = (y * y).mean(axis=-1)
    assert np.allclose(ms, 1.0, atol=1e-4)
