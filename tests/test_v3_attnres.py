# -*- coding: utf-8 -*-
"""BlockAttnRes 的数值守卫：ORV 实现必须与 numpy 参考实现一致。

参考：`model_v3.py:1227-1282`；改写说明见 `v3_attnres.py` 的模块文档
（ORV 的 einsum 在 CUDA 上不可用，故换成 matmul + 广播）。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import v3_attnres as A


def _case(seed=0, B=2, Tn=5, D=16, N=3):
    rng = np.random.default_rng(seed)
    blocks = [rng.normal(0, 1, (B, Tn, D)).astype(np.float32) for _ in range(N)]
    partial = rng.normal(0, 1, (B, Tn, D)).astype(np.float32)
    w = rng.normal(0, 0.02, (D,)).astype(np.float32)
    gw = np.ones(D, np.float32)
    return blocks, partial, w, gw


def test_block_attn_res_matches_numpy():
    blocks, partial, w, gw = _case()
    ref = A.block_attn_res_numpy(blocks, partial, w, gw)
    got = A.block_attn_res(
        [T.tensor(b.copy()) for b in blocks],
        T.tensor(partial.copy()),
        T.tensor(w.copy()),
        T.tensor(gw.copy()),
    )
    err = float(np.abs(got.numpy() - ref).max())
    assert err < 1e-5, "BlockAttnRes 与参考实现不一致：max_abs_err=%.3e" % err


def test_block_attn_res_empty_blocks_returns_partial():
    _, partial, w, gw = _case()
    out = A.block_attn_res([], T.tensor(partial.copy()), T.tensor(w.copy()), T.tensor(gw.copy()))
    assert np.array_equal(out.numpy(), partial), "没有已完成 block 时必须原样返回 partial_block"


def test_block_attn_res_is_a_convex_combination():
    """softmax 权重之和为 1 ⇒ 输出必落在各 block 值域的凸包内（逐元素上下界之内）。"""
    blocks, partial, w, gw = _case(seed=7)
    got = A.block_attn_res(
        [T.tensor(b.copy()) for b in blocks],
        T.tensor(partial.copy()),
        T.tensor(w.copy()),
        T.tensor(gw.copy()),
    ).numpy()
    V = np.stack(blocks + [partial], 0)
    lo, hi = V.min(0), V.max(0)
    assert np.all(got >= lo - 1e-5) and np.all(got <= hi + 1e-5), "输出跑出了凸包，softmax 归一化有问题"
