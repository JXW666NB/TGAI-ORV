# -*- coding: utf-8 -*-
"""打包训练的关键路径：**块对角掩码**在 ORV 上生效且与参考一致。

## 这条为什么重要

打包训练把多篇文档拼进一条序列，靠块对角掩码保证文档互不可见。
参考实现记录过一个 P0 缺陷：掩码被压成 `(T,T)` 广播到整个 batch，导致
**batch 1..B-1 全用 batch 0 的文档边界**，实测最大差 0.975。

ORV 原来没有加性掩码入口（`scaled_dot_product_attention` 只有 `causal` 布尔），
所以这条路径一直空着；本文件验证刚加上的 `attn_mask` 通路：
①与 numpy 参考数值一致；②跨文档位置**严格**互不可见。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import v3_ops as V  # noqa: E402
from tgai_go.train import v3_masks as M  # noqa: E402


def _np(t):
    r = t.to_numpy() if hasattr(t, "to_numpy") else t
    return np.asarray(r, np.float32)


def _setup(seed=0, B=2, T=6, dm=16, d_k=8, n_q=4, n_kv=2):
    rng = np.random.default_rng(seed)
    wq = rng.normal(0, .1, (n_q * d_k, dm)).astype(np.float32)
    wk = rng.normal(0, .1, (n_kv * d_k, dm)).astype(np.float32)
    wv = rng.normal(0, .1, (n_kv * d_k, dm)).astype(np.float32)
    wo = rng.normal(0, .1, (dm, n_q * d_k)).astype(np.float32)
    x = rng.normal(0, 1, (B, T, dm)).astype(np.float32)
    cos, sin = V.rope_cos_sin(T, d_k, 10000.0)
    return x, wq, wk, wv, wo, cos, sin, n_q, n_kv, d_k, B, T


def test_segmask_matches_reference():
    x, wq, wk, wv, wo, cos, sin, n_q, n_kv, d_k, B, T = _setup()
    doc = np.zeros((B, T), np.int32)
    doc[0, 3:] = 1            # batch0 在 3 处换文档
    doc[1, 4:] = 1            # batch1 在 4 处换文档 ⇒ 两条 batch 的边界不同
    seg = M.seg_additive(M.make_seg_mask(doc))

    got = _np(V.attention_orv(V.T.tensor(x), V.T.tensor(wq), V.T.tensor(wk),
                              V.T.tensor(wv), V.T.tensor(wo),
                              V.T.tensor(cos), V.T.tensor(sin),
                              d_k, n_q, n_kv, causal=False,
                              attn_mask=V.T.tensor(seg)))
    want = V.attention_numpy(x, wq, wk, wv, wo, cos, sin, d_k, seg, n_q, n_kv)
    err = float(np.abs(got - want).max())
    assert err < 1e-5, "块对角掩码下与参考最大误差 %.3e" % err


def test_cross_document_is_strictly_invisible():
    """把另一篇文档的 token 值整体换掉，本文档的输出必须**逐位不变**。"""
    x, wq, wk, wv, wo, cos, sin, n_q, n_kv, d_k, B, T = _setup(seed=1, B=1)
    doc = np.zeros((B, T), np.int32)
    doc[0, 4:] = 1                                    # 前 4 个属文档0，后 2 个属文档1
    seg = M.seg_additive(M.make_seg_mask(doc))

    def run(xx):
        return _np(V.attention_orv(V.T.tensor(xx), V.T.tensor(wq), V.T.tensor(wk),
                                   V.T.tensor(wv), V.T.tensor(wo),
                                   V.T.tensor(cos), V.T.tensor(sin),
                                   d_k, n_q, n_kv, causal=False,
                                   attn_mask=V.T.tensor(seg)))

    y0 = run(x)
    x2 = x.copy()
    x2[0, 4:] += 7.0                                  # 只改第二篇文档
    y1 = run(x2)
    assert np.allclose(y0[0, :4], y1[0, :4], atol=1e-6), \
        "改了另一篇文档却影响了本文档输出 ⇒ 块对角掩码没生效"
    assert not np.allclose(y0[0, 4:], y1[0, 4:]), "被改的那篇文档本应有变化"
