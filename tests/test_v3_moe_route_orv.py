# -*- coding: utf-8 -*-
"""MoEv3 路由的 ORV 张量版（无偏置路径）与 numpy 参考对拍。

## 覆盖范围与限制

ORV **没有 `gather`**，所以"用偏置选出下标、再按这些下标取原始亲和度"这条路走不通。
本文件只覆盖 `expert_bias=None` 的路径 —— 训练从零开始时 `b = 0`，
这条路径与带偏置的路由**完全等价**，也是做"同权重同输入比 logits"够用的那条。

带偏置的路径必须显式报错（`NotImplementedError` + 修法），
而不是悄悄按零偏置算完给出错误结果 —— 后者会让实验数据不可信。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import v3_ops as V  # noqa: E402


def _np(t):
    r = t.to_numpy() if hasattr(t, "to_numpy") else t
    return np.asarray(r)


def _setup(n=12, d=8, E=4, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 1, (n, d)).astype(np.float32),
            rng.normal(0, 0.3, (E, d)).astype(np.float32))


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_indices_match_numpy(top_k):
    x, W = _setup()
    topi, topv, _s = V.moe_route_orv(V.T.tensor(x), V.T.tensor(W), top_k)
    ti = _np(topi)
    tv = _np(topv).astype(np.float32)

    ti_n, gate_n, _ = V.moe_route_numpy(x, W, None, top_k)
    assert np.array_equal(ti, ti_n), "专家下标必须逐个相同（否则门控会配错专家）"
    assert float(np.abs(tv - gate_n).max()) < 1e-6


def test_gate_is_raw_affinity_not_renormalised():
    """门控是**原始亲和度**，不等于在选中集合内重新归一化（那是 quantile 口径）。"""
    x, W = _setup(seed=1)
    K = 2
    topi, topv, s_all = V.moe_route_orv(V.T.tensor(x), V.T.tensor(W), K)
    tv = _np(topv).astype(np.float32)
    s = _np(s_all).astype(np.float32)

    gathered = np.take_along_axis(s, _np(topi), axis=-1)
    assert np.allclose(tv, gathered), "门控必须等于原始亲和度"
    renorm = gathered / gathered.sum(axis=-1, keepdims=True)
    assert not np.allclose(tv, renorm), "不应该在选中集合内重新归一化"


def test_bias_path_refuses_instead_of_guessing():
    """带偏置必须**明确拒绝**，不能静默按零偏置算。"""
    x, W = _setup()
    with pytest.raises(NotImplementedError) as ei:
        V.moe_route_orv(V.T.tensor(x), V.T.tensor(W), 2,
                        expert_bias=V.T.tensor(np.zeros(4, np.float32)))
    msg = str(ei.value)
    assert "gather" in msg and "修法" in msg, "错误信息必须说清原因与修法"
