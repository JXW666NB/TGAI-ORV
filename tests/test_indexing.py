# -*- coding: utf-8 -*-
"""gather / index_select 与 torch 的对照（阶段 B 判据）。

覆盖：
  * index_select 前向 vs torch.index_select（≤1e-6）
  * index_select 反向 vs torch autograd（≤1e-6），含重复索引（atomicAdd 累加）
  * gather_flat 前向/反向 vs torch（≤1e-6）
  * 越界与非法 dim 的报错（含"修法"）
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402

torch = pytest.importorskip("torch")


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def test_index_select_forward_matches_torch():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (16, 8)).astype(np.float32)
    idx = np.array([3, 1, 1, 15, 0], np.int64)
    ours = np.asarray(T.index_select(T.tensor(x), 0, idx).data, np.float32)
    ref = torch.index_select(torch.from_numpy(x), 0, torch.from_numpy(idx)).numpy()
    assert _rel(ours, ref) <= 1e-6, _rel(ours, ref)


def test_index_select_backward_matches_torch_with_duplicates():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (10, 5)).astype(np.float32)
    idx = np.array([7, 2, 2, 2, 0, 9], np.int64)     # 重复索引 ⇒ 反向要累加
    w = rng.normal(0, 1, (6, 5)).astype(np.float32)

    xt = T.tensor(x.copy(), requires_grad=True)
    y = T.index_select(xt, 0, idx)
    (y * T.tensor(w)).sum().backward()

    xr = torch.from_numpy(x.copy()).requires_grad_(True)
    yr = torch.index_select(xr, 0, torch.from_numpy(idx))
    (yr * torch.from_numpy(w)).sum().backward()

    assert _rel(xt.grad.data, xr.grad.numpy()) <= 1e-6, _rel(xt.grad.data, xr.grad.numpy())


def test_gather_flat_forward_and_backward_match_torch():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (4, 6)).astype(np.float32)
    flat_idx = np.array([0, 7, 7, 23, 5], np.int64)
    w = rng.normal(0, 1, (5,)).astype(np.float32)

    xt = T.tensor(x.copy(), requires_grad=True)
    y = T.gather_flat(xt, flat_idx)
    (y * T.tensor(w)).sum().backward()

    flat = torch.from_numpy(x.copy()).reshape(-1).requires_grad_(True)
    yr = flat[torch.from_numpy(flat_idx)]
    (yr * torch.from_numpy(w)).sum().backward()

    assert _rel(y.data, yr.detach().numpy()) <= 1e-6
    assert _rel(xt.grad.data, flat.grad.reshape(4, 6).numpy()) <= 1e-6


def test_index_select_dim1():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (3, 9)).astype(np.float32)
    idx = np.array([8, 0, 4], np.int64)
    ours = np.asarray(T.index_select(T.tensor(x), 1, idx).data, np.float32)
    ref = torch.index_select(torch.from_numpy(x), 1, torch.from_numpy(idx)).numpy()
    assert _rel(ours, ref) <= 1e-6, _rel(ours, ref)


def test_index_select_rejects_bad_input():
    x = T.tensor(np.zeros((4, 3), np.float32))
    with pytest.raises(ValueError, match="修法"):
        T.index_select(x, 0, np.array([4], np.int64))
    with pytest.raises(ValueError, match="修法"):
        T.index_select(x, 5, np.array([0], np.int64))


def test_gather_flat_rejects_out_of_range():
    x = T.tensor(np.zeros((2, 3), np.float32))
    with pytest.raises(ValueError, match="修法"):
        T.gather_flat(x, np.array([6], np.int64))


def test_scatter_flat_forward_and_backward_match_torch():
    rng = np.random.default_rng(4)
    src = rng.normal(0, 1, (4,)).astype(np.float32)
    idx = np.array([1, 3, 3, 7], np.int64)          # 重复下标累加
    n_out = 9
    w = rng.normal(0, 1, (n_out,)).astype(np.float32)

    st = T.tensor(src.copy(), requires_grad=True)
    y = T.scatter_flat(st, idx, n_out)
    (y * T.tensor(w)).sum().backward()

    ref = torch.zeros(n_out)
    ref.index_add_(0, torch.from_numpy(idx), torch.from_numpy(src.copy()).requires_grad_(True))
    ref2 = torch.zeros(n_out)
    sr = torch.from_numpy(src.copy()).requires_grad_(True)
    ref2.index_add_(0, torch.from_numpy(idx), sr)
    (ref2 * torch.from_numpy(w)).sum().backward()
    assert _rel(y.data, ref2.detach().numpy()) <= 1e-6
    assert _rel(st.grad.data, sr.grad.numpy()) <= 1e-6
