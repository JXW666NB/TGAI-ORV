# -*- coding: utf-8 -*-
"""CSA 的两个高风险步骤：按文档分段池化、因果深度卷积。

参考实现里这两处都记录过**真实泄漏**（P0-A），且都"不报错、只掉分"，所以必须由测试钉住：

1. **固定 chunk 池化会把两篇文档混进同一个 chunk** ⇒ 文档 B 的 token 看不到自己文档开头
   （它们被一起屏蔽了）。修法：按文档边界分段池化。
2. **卷积双向填充**（`padding=k-1` 后再切前 Tc 个位置）⇒ 每个位置还看得见未来 2 个 chunk。
   实测：把 chunk 0..7 的输入全换掉，chunk 8 的卷积输出变化 **1.450e+00**（它自己输入没动）。

这里的测试直接断言"扰动某个位置，它之前的位置必须**逐位不变**"。
"""
import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import v3_ops as V  # noqa: E402


def test_compress_length_is_ceil_T_over_c():
    rng = np.random.default_rng(0)
    for T, c in [(10, 4), (12, 4), (13, 4), (7, 2)]:
        h = rng.normal(0, 1, (2, T, 3)).astype(np.float32)
        out = V.csa_compress(h, c)
        assert out.shape == (2, int(math.ceil(T / float(c))), 3)


def test_fixed_chunk_pooling_averages_within_chunk():
    h = np.arange(1, 13, dtype=np.float32).reshape(1, 12, 1)   # 1..12
    out = V.csa_compress(h, 4)
    want = np.array([[[2.5], [6.5], [10.5]]], np.float32)      # (1+2+3+4)/4 等
    assert np.allclose(out, want), "out=%s want=%s" % (out.ravel(), want.ravel())


def test_seg_pooling_does_not_mix_documents():
    """**核心断言**：跨文档的 chunk 不允许出现。

    构造：文档 A = 位置 0..5（值 100），文档 B = 位置 6..11（值 1）。
    按 c=4 池化时，固定 chunk 会把 A/B 混进 chunk 1；分段池化则每个 chunk 只含单一文档的值。
    """
    T, c = 12, 4
    h = np.zeros((1, T, 1), np.float32)
    h[0, :6, 0] = 100.0
    h[0, 6:, 0] = 1.0
    seg = np.array([[0] * 6 + [1] * 6], np.int32)

    out = V.csa_compress(h, c, seg=seg)
    vals = out[0, :, 0].tolist()
    # 每个非零 chunk 的值必须是"纯 100"或"纯 1"，不能出现中间值
    for v in vals:
        assert v in (0.0,) or abs(v - 100.0) < 1e-5 or abs(v - 1.0) < 1e-5, \
            "出现混合值 %.3f ⇒ 跨文档池化" % v
    assert any(abs(v - 100.0) < 1e-5 for v in vals)
    assert any(abs(v - 1.0) < 1e-5 for v in vals)


def test_seg_pooling_pads_tail_with_zeros():
    T, c = 7, 4
    h = np.ones((1, T, 2), np.float32)
    seg = np.zeros((1, T), np.int32)          # 单文档、T 不是 c 的倍数
    out = V.csa_compress(h, c, seg=seg)
    assert out.shape[1] == 2
    assert np.allclose(out[0, 1], 1.0), "该文档只有 3 个 token，第二 chunk 仍是均值 1"
    assert out.shape[1] <= 2


def test_conv_is_causal_no_lookahead():
    """**泄漏回归**：扰动位置 p，则所有 < p 的输出必须逐位不变。"""
    rng = np.random.default_rng(1)
    B, Tc, L, k = 1, 8, 3, 3
    x = rng.normal(0, 1, (B, Tc, L)).astype(np.float32)
    w = rng.normal(0, 1, (L, k)).astype(np.float32)
    y0 = V.csa_conv_causal(x, w)

    for p in range(Tc):
        x2 = x.copy()
        x2[0, p] += 10.0
        y1 = V.csa_conv_causal(x2, w)
        assert np.array_equal(y0[:, :p], y1[:, :p]), \
            "扰动位置 %d 却改变了它之前的位置 ⇒ 卷积在偷看未来" % p
        # 扰动必须确实影响了它自己（否则测试可能无效）
        assert not np.array_equal(y0[:, p], y1[:, p])


def test_conv_matches_naive_loop():
    rng = np.random.default_rng(2)
    B, Tc, L, k = 2, 6, 4, 3
    x = rng.normal(0, 1, (B, Tc, L)).astype(np.float32)
    w = rng.normal(0, 1, (L, k)).astype(np.float32)
    got = V.csa_conv_causal(x, w)

    want = np.zeros_like(got)
    for b in range(B):
        for t in range(Tc):
            for j in range(k):
                if t - j >= 0:
                    want[b, t] += w[:, j] * x[b, t - j]
    assert np.abs(got - want).max() < 1e-6
