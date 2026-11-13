# -*- coding: utf-8 -*-
"""V3 单层：ORV 版与 numpy 参考逐位对拍（单专家、GlobalAttention）。

这一步验证的是**胶水**而不是算子：norm 放在残差前、残差相加、形状/通道顺序、
注意力掩码与因果的选择。这些在整模型里只会让 loss 略微偏移（最难查），
单层里则会放大成可比的差值。

限制（写在测试里）：单专家 —— 真分派需要 gather/scatter，ORV 尚未提供。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import v3_ops as V  # noqa: E402
from tgai_go.train import v3_layer as VL  # noqa: E402


def _np(t):
    r = t.to_numpy() if hasattr(t, "to_numpy") else t
    return np.asarray(r, np.float32)


def _setup(seed=0, B=2, T=6, d=16, d_ff=24, n_q=4, n_kv=2):
    rng = np.random.default_rng(seed)
    d_k = d // n_q
    w = dict(
        ln1=(1.0 + 0.1 * rng.normal(size=d)).astype(np.float32),
        ln2=(1.0 + 0.1 * rng.normal(size=d)).astype(np.float32),
        wq=rng.normal(0, .1, (n_q * d_k, d)).astype(np.float32),
        wk=rng.normal(0, .1, (n_kv * d_k, d)).astype(np.float32),
        wv=rng.normal(0, .1, (n_kv * d_k, d)).astype(np.float32),
        wo=rng.normal(0, .1, (d, n_q * d_k)).astype(np.float32),
        w1=rng.normal(0, .1, (d_ff, d)).astype(np.float32),
        w2=rng.normal(0, .1, (d_ff, d)).astype(np.float32),
        w3=rng.normal(0, .1, (d, d_ff)).astype(np.float32),
    )
    x = rng.normal(0, 1, (B, T, d)).astype(np.float32)
    cos, sin = V.rope_cos_sin(T, d_k, 10000.0)
    return x, w, cos, sin, n_q, n_kv, d_k


def test_layer_matches_reference_causal():
    x, w, cos, sin, n_q, n_kv, d_k = _setup()
    got = _np(VL.layer_orv(x, w, cos, sin, n_q, n_kv, d_k))
    want = VL.layer_numpy(x, w, cos, sin, n_q, n_kv, d_k, add_mask=None)
    assert got.shape == want.shape
    err = float(np.abs(got - want).max())
    assert err < 1e-4, "单层 ORV 与参考最大误差 %.3e" % err


def test_layer_matches_reference_with_block_diagonal_mask():
    """打包场景：块对角掩码下单层也要对齐（跨文档互不可见）。"""
    x, w, cos, sin, n_q, n_kv, d_k = _setup(seed=1)
    B, T = x.shape[0], x.shape[1]
    doc = np.zeros((B, T), np.int32)
    doc[0, 3:] = 1
    doc[1, 4:] = 1
    from tgai_go.train import v3_masks as M
    seg = M.seg_additive(M.make_seg_mask(doc))

    got = _np(VL.layer_orv(x, w, cos, sin, n_q, n_kv, d_k,
                           add_mask=V.T.tensor(seg)))
    want = VL.layer_numpy(x, w, cos, sin, n_q, n_kv, d_k, add_mask=seg)
    err = float(np.abs(got - want).max())
    assert err < 1e-4, "块对角掩码下单层最大误差 %.3e" % err


def test_residual_changes_output():
    """残差必须真的生效：把 x 整体平移，输出必须跟着变（否则残差被吞了）。"""
    x, w, cos, sin, n_q, n_kv, d_k = _setup(seed=2)
    y0 = _np(VL.layer_orv(x, w, cos, sin, n_q, n_kv, d_k))
    y1 = _np(VL.layer_orv(x + 0.5, w, cos, sin, n_q, n_kv, d_k))
    assert not np.allclose(y0, y1), "输入平移后输出不变 ⇒ 残差没接上"
