# -*- coding: utf-8 -*-
"""V3 注意力掩码的回归测试，重点是**打包训练的块对角掩码不能只用 batch 0**。

## 这条测试对应一个真实 bug

参考实现 `model_v3.py::GlobalAttention.forward` 的注释里记录了 2026-10-16 修掉的 P0 缺陷：
旧代码把 `(B,1,T,T)` 的 `seg_mask` 先 `[:, 0]` 再取 `m[0]`，最终退化成 `(T,T)`，
被 SDPA 广播到整个 batch ⇒ **batch 1..B-1 全用 batch 0 的文档边界**。

实测症状（参考实现记录）：B=2、batch0 分界=8、batch1 分界=24 时，
与正确 4D 掩码最大差 9.748e-01；batch1 第 28 行可见 key 变成 `[8..28]`（21 个），
正确应为 `[24..28]`（5 个）。

下面把它写成可自动判定的断言 —— 这个 bug 一旦回潮，测试立刻红。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import v3_masks as M  # noqa: E402


def test_seg_mask_keeps_batch_dim():
    """返回形状必须是 `(B,1,T,T)`，**不能**被压成 `(T,T)`。"""
    B, T = 2, 32
    doc = np.zeros((B, T), np.int32)
    doc[0, 16:] = 1          # batch0 在 16 处换文档
    doc[1, 8:] = 1           # batch1 在 8 处换文档
    seg = M.make_seg_mask(doc)
    add = M.seg_additive(seg)
    assert add.shape == (B, 1, T, T), "保留 batch 维是这条修复的核心"


def test_each_batch_uses_its_own_doc_boundary():
    """**回归断言**：batch1 必须用自己的边界（8），而不是 batch0 的（16）。"""
    B, T = 2, 32
    doc = np.zeros((B, T), np.int32)
    doc[0, 16:] = 1
    doc[1, 8:] = 1
    add = M.seg_additive(M.make_seg_mask(doc))
    visible1 = (add[1, 0] == 0.0)

    # batch1：位置 20 只能看到本文档 [8..31] 且因果 ⇒ 可见集 = [8..20]
    row = 20
    got = np.flatnonzero(visible1[row]).tolist()
    assert got == list(range(8, 21)), "batch1 第 %d 行可见 %s" % (row, got)

    # 若按旧 bug（用 batch0 的边界 16）可见集会从 16 开始 —— 明确排除
    assert 8 in got and 9 in got, "batch1 的文档从 8 开始，8/9 必须可见"
    assert 7 not in got


def test_buggy_batch0_only_mask_would_differ():
    """把"只用 batch0"的错误掩码显式造出来，证明它确实与正确掩码不同。

    这条不是测我们的实现，而是**固定住 bug 的存在条件**：
    只要有人把实现改回 m[0]，上面那条 `test_each_batch_uses_its_own_doc_boundary` 就会失败。
    """
    B, T = 2, 32
    doc = np.zeros((B, T), np.int32)
    doc[0, 16:] = 1
    doc[1, 8:] = 1
    seg = M.make_seg_mask(doc)
    correct = M.seg_additive(seg)
    buggy = np.repeat(M.seg_additive(seg[0])[:1], B, axis=0)   # 只用 batch0 并广播
    assert not np.array_equal(correct, buggy), "两者本应不同，否则测试失去意义"
    assert np.array_equal(correct[0], buggy[0]), "batch0 上两者一致（bug 只影响后续 batch）"


def test_causal_mask_basic_and_sliding():
    T = 8
    c = M.causal_additive(T)
    vis = (c == 0.0)
    # 行 i 可见 0..i
    assert all(vis[i, :i + 1].all() and not vis[i, i + 1:].any() for i in range(T))

    sw = M.causal_additive(T, sliding_window=3)
    v2 = (sw == 0.0)
    for i in range(T):
        want = list(range(max(0, i - 2), i + 1))
        assert np.flatnonzero(v2[i]).tolist() == want


def test_incremental_causal_right_aligned():
    """增量解码：T=1、S=5 时应看到全部 5 个 key（右侧对齐）。"""
    c = M.causal_additive(1, S=5)
    assert (c == 0.0).all()


def test_attention_weights_ignore_masked_positions():
    """掩码位置的概率必须严格为 0（不是"很小"）。"""
    T = 6
    add = M.causal_additive(T)
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 3, (T, T)).astype(np.float32)
    p = M.attention_weights(scores, add)
    for i in range(T):
        assert abs(float(p[i, i + 1:].sum())) == 0.0
        assert abs(float(p[i].sum()) - 1.0) < 1e-5
