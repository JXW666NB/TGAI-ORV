# -*- coding: utf-8 -*-
"""`scatter_add`（分段求和）的**判定基准**：MoE 分派缺的就是这个原语。

## 为什么需要它

实测结论（见 `docs/V3_PORT_STATUS.md`）：

* 取行 `x[ids]` ✅ 可用，反向正确 scatter-add；
* 赋值 `x[ids] = v` ❌ 不支持；
* 逐元素相等 `==` 返回 Python bool ❌ 无法构造 one-hot。

于是 MoE 只剩两条坏路（算力 ×E，或无法回填），**真正缺的是 `scatter_add(ids, values, n_rows)`**：

    out[ids[i]] += values[i]          # 前向
    values.grad = out.grad[ids]       # 反向（正是已有的 gather）

本文件提供它的 numpy 判定基准（`np.add.at`），并固定三条必须成立的性质，
供未来 ORV 张量实现逐条对拍。**在 ORV 实现落地前，这里不冒充"已完成"。**
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import v3_ops as V  # noqa: E402


def test_duplicates_accumulate_not_overwrite():
    """同一个下标出现多次必须**累加**（这是 scatter_add 与赋值索引的唯一区别）。"""
    ids = np.array([0, 2, 2], np.int64)
    vals = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], np.float32)
    out = V.scatter_add_numpy(ids, vals, 4)
    assert out.shape == (4, 2)
    assert np.allclose(out[0], [1, 1])
    assert np.allclose(out[2], [5, 5]), "重复下标应累加为 5，而不是被 3 覆盖"
    assert np.allclose(out[1], [0, 0]) and np.allclose(out[3], [0, 0])


def test_matches_numpy_add_at():
    """与 `np.add.at` 逐位一致（这是公认的分段求和语义）。"""
    rng = np.random.default_rng(0)
    n_rows, rows, d = 7, 20, 3
    ids = rng.integers(0, n_rows, rows).astype(np.int64)
    vals = rng.normal(size=(rows, d)).astype(np.float32)

    want = np.zeros((n_rows, d), np.float32)
    np.add.at(want, ids, vals)
    got = V.scatter_add_numpy(ids, vals, n_rows)
    assert np.array_equal(got, want)


def test_backward_is_gather():
    """反向 = 按 ids 取行（把这条写成断言，避免未来实现把反向搞成别的）。"""
    rng = np.random.default_rng(1)
    n_rows, rows, d = 5, 9, 2
    ids = rng.integers(0, n_rows, rows).astype(np.int64)
    g_out = rng.normal(size=(n_rows, d)).astype(np.float32)      # 上游梯度

    # 反向公式：values.grad[i] = g_out[ids[i]]
    want = g_out[ids]
    got = V.scatter_add_backward_numpy(ids, g_out)
    assert np.array_equal(got, want)


def test_out_of_range_is_rejected():
    """越界下标必须报错并给修法（静默丢弃会让 MoE 少算一部分 token，最难查）。"""
    with pytest.raises(ValueError) as ei:
        V.scatter_add_numpy(np.array([0, 5], np.int64),
                            np.ones((2, 2), np.float32), 3)
    assert "修法" in str(ei.value)
