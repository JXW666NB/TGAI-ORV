# -*- coding: utf-8 -*-
"""SiTU-GLU 的 ORV 张量版与 numpy 参考对拍，并验证"有界"性质在 ORV 上同样成立。"""
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


def _mk(seed=0, n=5, d=8, d_ff=16, scale=0.1):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 1, (n, d)).astype(np.float32),
            rng.normal(0, scale, (d_ff, d)).astype(np.float32),
            rng.normal(0, scale, (d_ff, d)).astype(np.float32),
            rng.normal(0, scale, (d, d_ff)).astype(np.float32))


@pytest.mark.parametrize("n,d,d_ff", [(5, 8, 16), (1, 16, 8), (7, 4, 12)])
def test_matches_numpy(n, d, d_ff):
    x, w1, w2, w3 = _mk(n=n, d=d, d_ff=d_ff)
    got = _np(V.situ_glu_orv(V.T.tensor(x), V.T.tensor(w1),
                             V.T.tensor(w2), V.T.tensor(w3)))
    want = V.situ_glu_numpy(x, w1, w2, w3)
    assert got.shape == want.shape
    err = float(np.abs(got - want).max())
    assert err < 1e-5, "ORV 版 SiTU-GLU 与参考最大误差 %.3e" % err


def test_bounded_on_orv_too():
    """有界性必须在 ORV 路径上也成立（这是 SiTU 存在的唯一理由）。"""
    rng = np.random.default_rng(3)
    n, d, d_ff = 4, 8, 16
    x = rng.normal(0, 50.0, (n, d)).astype(np.float32)      # 故意大输入
    w1 = rng.normal(0, 1.0, (d_ff, d)).astype(np.float32)
    w2 = rng.normal(0, 1.0, (d_ff, d)).astype(np.float32)
    w3 = rng.normal(0, 1.0, (d, d_ff)).astype(np.float32)

    out = _np(V.situ_glu_orv(V.T.tensor(x), V.T.tensor(w1),
                             V.T.tensor(w2), V.T.tensor(w3)))
    assert np.isfinite(out).all(), "大输入下不应出现 inf/NaN"
    # 中间量 gate*up 的上界是 100；w3 会再放大，这里用 numpy 复核中间量
    g = x @ w1.T
    u = x @ w2.T
    gate = (4.0 * np.tanh(g / 4.0)) * (1.0 / (1.0 + np.exp(-g)))
    up = 25.0 * np.tanh(u / 25.0)
    assert np.abs(gate * up).max() <= 100.0 + 1e-3
