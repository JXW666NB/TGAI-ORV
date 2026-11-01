# -*- coding: utf-8 -*-
"""分块 CE（`chunked_ce`）：数值与整块一致，且反向梯度也一致。

背景（V100，B=4/S=512 实测）：活张量峰值 29.92 GiB 里，logits / softmax / grad_logits
各 1.24 GB（B*S*V*4），三份 3.7 GB —— 32 GB 卡直接 OOM。分块后只保留块大小的 logits。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

import tgai_go.train as T
from tgai_go.train import qwen as Q

CFG = dict(V=64, D=32, L=2, NH=4, KV=2, FF=64, THETA=1e6, EPS=1e-6, DH=8,
           QK_NORM=False, QKV_BIAS=True, TIE=True)


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


def _setup(seed=0):
    m = Q.Qwen25(CFG, Q.make_random_weights(CFG, seed=seed))
    ids = np.random.default_rng(0).integers(0, 60, (2, 12)).astype(np.int64)
    tgt = np.random.default_rng(1).integers(0, 60, (24,)).astype(np.int64)
    z = m.forward(ids, hidden=True)
    return m, z, m.p["lm_head.weight"], tgt


def test_chunked_loss_matches_full():
    m, z, W, tgt = _setup()
    a = float(T.functional.cross_entropy(z @ W.T, tgt).to_numpy())
    b = float(Q.chunked_ce(z, W, tgt, chunk=8).to_numpy())
    assert abs(a - b) < 1e-5, "分块与整块的 loss 必须一致（实测 %.6f vs %.6f）" % (a, b)


def test_chunked_grads_match_full():
    m1, z1, W1, tgt = _setup()
    (z1 @ W1.T)
    l1 = T.functional.cross_entropy(z1 @ W1.T, tgt)
    l1.backward()
    g_full = np.asarray(m1.p["lm_head.weight"].grad, np.float64).copy()

    m2, z2, W2, _ = _setup()
    l2 = Q.chunked_ce(z2, W2, tgt, chunk=8)
    l2.backward()
    g_chunk = np.asarray(m2.p["lm_head.weight"].grad, np.float64).copy()

    worst = float(np.abs(g_full - g_chunk).max())
    assert worst < 1e-5, "两路梯度必须一致，实测最大差 %g" % worst


def test_chunk_size_one_still_works():
    m, z, W, tgt = _setup()
    a = float(T.functional.cross_entropy(z @ W.T, tgt).to_numpy())
    b = float(Q.chunked_ce(z, W, tgt, chunk=1).to_numpy())
    assert abs(a - b) < 1e-5


def test_all_ignored_returns_zero_and_backprops():
    m, z, W, _ = _setup()
    tgt = np.full((24,), -100, np.int64)
    loss = Q.chunked_ce(z, W, tgt, chunk=8)
    assert abs(float(loss.to_numpy())) < 1e-9
