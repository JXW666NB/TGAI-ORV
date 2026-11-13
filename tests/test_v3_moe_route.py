# -*- coding: utf-8 -*-
"""MoEv3 路由对拍：`b` 只影响选择、门控用原始 s、偏置符号式更新。

## 参考实现的三个"初版写错"点（原文标注，已按论文修正）

1. 偏置 `b_i` **只用于 top-k 选择**；门控权重用**原始亲和度 `s`**，不是 `s+b`
2. 偏置**符号式**更新：过载 `b←b−γ`、欠载 `b←b+γ`
3. 共享专家**直接相加、无门控**（sigmoid 门控是 Qwen 的做法）

这三条任何一条搞反，训练都能跑、loss 也降，但**路由分布与负载均衡全错**，
而且不会报错 —— 所以必须由测试钉住，而不是靠读代码。
"""
import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import v3_ops as V  # noqa: E402


def _setup(n=6, d=8, E=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, d)).astype(np.float32)
    W = rng.normal(0, 0.3, (E, d)).astype(np.float32)
    return x, W


def test_gate_uses_raw_affinity_not_biased():
    """**核心断言**：偏置改变选择结果，但门控值必须等于原始 softmax。"""
    x, W = _setup()
    E = W.shape[0]
    bias = np.zeros(E, np.float32)
    topi0, gate0, s_all = V.moe_route_numpy(x, W, bias, top_k=2)

    # 造一个会改变选择的偏置
    bias2 = np.array([0.0, 0.0, 0.5, 0.5], np.float32)
    topi2, gate2, s_all2 = V.moe_route_numpy(x, W, bias2, top_k=2)

    assert np.allclose(s_all, s_all2), "亲和度不应受偏置影响"
    # 选出来的专家可能变了
    changed = not np.array_equal(topi0, topi2)
    assert changed, "这个偏置本应改变选择（否则测试失去意义）"

    # 无论选了谁，门控值必须等于 s_all 在被选位置上的取值（不是 s+b）
    gathered = np.take_along_axis(s_all2, topi2, axis=-1)
    assert np.allclose(gate2, gathered), "门控必须用原始亲和度"


def test_topk_is_sorted_descending():
    x, W = _setup()
    topi, gate, s_all = V.moe_route_numpy(x, W, None, top_k=3)
    assert (np.diff(gate, axis=-1) <= 1e-7).all(), "top-k 必须按降序返回"


def test_softmax_affinity_sums_to_one_but_gate_does_not():
    """全专家 softmax：`s_all` 和为 1；只取 top-k 后的 gate 和必然 < 1。"""
    x, W = _setup()
    topi, gate, s_all = V.moe_route_numpy(x, W, None, top_k=2)
    assert np.allclose(s_all.sum(axis=-1), 1.0, atol=1e-6)
    assert (gate.sum(axis=-1) < 1.0 - 1e-6).all()


def test_quantile_mode_renormalises_within_selected():
    """K3 QB 口径：门控在**选中集合内**重新归一化 ⇒ 和为 1。"""
    x, W = _setup()
    topi, gate, s_all = V.moe_route_numpy(x, W, None, top_k=2, balance="quantile")
    assert np.allclose(gate.sum(axis=-1), 1.0, atol=1e-5)
    assert ((s_all > 0) & (s_all < 1)).all(), "quantile 模式用 sigmoid"


def test_bias_updates_by_sign_not_proportion():
    """偏置更新是**符号式**：过载减 γ、欠载加 γ，与超出幅度无关。"""
    E, K, gamma = 4, 2, 1e-3
    b0 = np.zeros(E, np.float32)
    # 专家 0 过载（0.9），其余欠载
    usage = np.array([0.9, 0.4, 0.4, 0.3], np.float32)
    b1 = V.moe_update_bias(b0, usage, gamma, K, E)

    ideal = K / E
    for e in range(E):
        if usage[e] < ideal:
            assert b1[e] == pytest.approx(b0[e] + gamma)
        else:
            assert b1[e] == pytest.approx(b0[e] - gamma)
    # 幅度与超出量无关：两个欠载专家加的量相同
    assert b1[1] - b0[1] == pytest.approx(b1[2] - b0[2])


def test_balance_loop_moves_usage_toward_uniform():
    """把偏置更新与路由接起来跑几十轮：负载应更均匀（这是偏置存在的唯一目的）。"""
    rng = np.random.default_rng(3)
    n, d, E, K = 64, 8, 4, 2
    x = rng.normal(0, 1, (n, d)).astype(np.float32)
    # 故意不均衡：专家 0 的 router 权重偏大
    W = rng.normal(0, 0.3, (E, d)).astype(np.float32)
    W[0] *= 3.0

    b = np.zeros(E, np.float32)

    def usage_of(bvec):
        topi, _gate, _s = V.moe_route_numpy(x, W, bvec, top_k=K)
        u = np.zeros(E, np.float32)
        for e in range(E):
            u[e] = (topi == e).sum() / float(n)
        return u

    u0 = usage_of(b)
    for _ in range(60):
        u = usage_of(b)
        b = V.moe_update_bias(b, u, 1e-2, K, E)
    u1 = usage_of(b)

    spread0 = float(u0.max() - u0.min())
    spread1 = float(u1.max() - u1.min())
    assert spread1 < spread0, "偏置更新后负载应更均匀：%s → %s" % (u0, u1)
    assert abs(float(u1.sum()) - K) < 1e-6, "各专家使用率之和应等于 top_k"
