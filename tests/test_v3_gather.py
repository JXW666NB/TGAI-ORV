# -*- coding: utf-8 -*-
"""ORV 张量的整数数组索引 = 嵌入查表原语（**含正确的 scatter-add 反向**）。

## 这条为什么重要

之前 `docs/V3_PORT_STATUS.md` 把 `gather` 列为硬缺口，理由是"嵌入查表与 MoE 分派都需要它"。
实测发现：**ORV 已经支持 `x[ids]`（ids 为整数 ndarray），且反向正确按出现次数累加**。

于是"模型无法从 token id 前向"这个结论被推翻 —— 嵌入可以用现成能力做，
不需要新写算子。MoE 的**分派**仍需另一件事（按专家分组再合并），但取行本身不再是障碍。

## 测试断言什么

1. 前向形状与数值等于 numpy 的 `w[ids]`；
2. 反向梯度等于"每个被选中的行按出现次数累加"× 每行元素个数
   （`sum()` 作为 loss 时，每行每个元素贡献 1，所以行和 = 出现次数 × 列数）；
3. 未被选中的行梯度严格为 0（稀疏性必须成立，否则嵌入更新会污染无关词）。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

import tgai_go.train as T  # noqa: E402


def _np(t):
    r = t.to_numpy() if hasattr(t, "to_numpy") else t
    return np.asarray(r, np.float32)


@pytest.mark.parametrize("ids", [[0, 2, 4], np.array([1, 1, 3], np.int64),
                                 np.array([4, 0], np.int64)])
def test_forward_matches_numpy(ids):
    w = np.arange(20, dtype=np.float32).reshape(5, 4)
    got = _np(T.tensor(w)[ids])
    want = w[np.asarray(ids, np.int64)]
    assert got.shape == want.shape
    assert np.array_equal(got, want.astype(np.float32))


def test_backward_accumulates_by_occurrence():
    """出现两次的行，梯度必须是出现一次的 2 倍（scatter-add，不是覆盖）。"""
    w = np.zeros((5, 4), np.float32)
    t = T.tensor(w, requires_grad=True)
    ids = np.array([0, 2, 2], np.int64)
    t[ids].sum().backward()

    g = _np(t.grad)
    row_sums = g.reshape(5, -1).sum(axis=1).tolist()
    assert row_sums == pytest.approx([4.0, 0.0, 8.0, 0.0, 0.0]), row_sums


def test_unselected_rows_have_zero_grad():
    w = np.zeros((6, 3), np.float32)
    t = T.tensor(w, requires_grad=True)
    ids = np.array([1, 5], np.int64)
    t[ids].sum().backward()
    g = _np(t.grad)
    for row in (0, 2, 3, 4):
        assert float(np.abs(g[row]).sum()) == 0.0, "第 %d 行不该有梯度" % row
