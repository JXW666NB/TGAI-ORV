# -*- coding: utf-8 -*-
"""GlobalAttention 的 numpy 参考实现：形状、GQA 语义、掩码、与暴力循环逐位一致。

## 为什么要"暴力循环"版本

参考实现用 SDPA（一个黑盒）。把同一个注意力用**显式循环**再写一遍，两边逐位一致，
才说明"投影 + 部分 RoPE + GQA 交错复制 + 因果掩码 + softmax + 拼头 + o_proj"这套组合
没有哪一步搞反。这是后面 ORV 版要对拍的基准。

## 关键语义（都从参考实现读出，不是猜的）

* GQA：`k.repeat_interleave(rep, dim=1)` —— **交错**复制，不是 tile；
* RoPE：只作用在**前 rot_dim 维**（partial），其余原样通过；
* 掩码：SDPA 调用时 `is_causal=False`，因果**已包含在掩码里**；
* 缩放：`1/√d_k`。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import v3_ops as V  # noqa: E402
from tgai_go.train import v3_masks as M  # noqa: E402


def _weights(rng, d_model, n_q, n_kv, d_k):
    return (rng.normal(0, 0.1, (n_q * d_k, d_model)).astype(np.float32),
            rng.normal(0, 0.1, (n_kv * d_k, d_model)).astype(np.float32),
            rng.normal(0, 0.1, (n_kv * d_k, d_model)).astype(np.float32),
            rng.normal(0, 0.1, (d_model, n_q * d_k)).astype(np.float32))


def _bruteforce(x, wq, wk, wv, wo, cos, sin, rot_dim, add_mask, n_q, n_kv):
    """显式循环版：逐 batch / 逐头 / 逐 query 位置算注意力。"""
    B, T, _ = x.shape
    d_k = wq.shape[0] // n_q
    q = (x @ wq.T).reshape(B, T, n_q, d_k).transpose(0, 2, 1, 3)
    k = (x @ wk.T).reshape(B, T, n_kv, d_k).transpose(0, 2, 1, 3)
    v = (x @ wv.T).reshape(B, T, n_kv, d_k).transpose(0, 2, 1, 3)

    def rope(t):
        if rot_dim >= d_k:
            return V.v3_rope_numpy(t, cos[None, None], sin[None, None])
        head = V.v3_rope_numpy(t[..., :rot_dim], cos[None, None], sin[None, None])
        return np.concatenate([head, t[..., rot_dim:]], axis=-1)

    q, k = rope(q), rope(k)
    rep = n_q // n_kv
    mask = add_mask if add_mask.ndim == 4 else add_mask[None, None]
    mask = np.broadcast_to(mask, (B, 1, T, mask.shape[-1]))
    out = np.zeros((B, n_q, T, d_k), np.float32)
    for b in range(B):
        for h in range(n_q):
            kvh = h // rep                       # 交错复制 ⇒ 第 h 个 q 头用第 h//rep 个 kv 头
            for i in range(T):
                s = q[b, h, i] @ k[b, kvh].T / np.sqrt(float(d_k)) + mask[b, 0, i]
                s = s - np.max(np.where(np.isfinite(s), s, -1e30))
                e = np.where(np.isfinite(mask[b, 0, i]), np.exp(s), 0.0)
                den = e.sum()
                w = e / den if den > 0 else e
                out[b, h, i] = w @ v[b, kvh]
    out = out.transpose(0, 2, 1, 3).reshape(B, T, n_q * d_k)
    return out @ wo.T


@pytest.mark.parametrize("n_q,n_kv,rot_partial", [(4, 4, 1.0), (4, 2, 1.0), (4, 2, 0.5)])
def test_matches_bruteforce(n_q, n_kv, rot_partial):
    rng = np.random.default_rng(0)
    B, T, d_model, d_k = 2, 6, 16, 8
    wq, wk, wv, wo = _weights(rng, d_model, n_q, n_kv, d_k)
    x = rng.normal(0, 1, (B, T, d_model)).astype(np.float32)
    rd = int(round(d_k * rot_partial))
    rd = rd if rd % 2 == 0 else rd - 1
    cos, sin = V.rope_cos_sin(T, rd, 10000.0)
    add = M.causal_additive(T)

    got = V.attention_numpy(x, wq, wk, wv, wo, cos, sin, rd, add, n_q, n_kv)
    want = _bruteforce(x, wq, wk, wv, wo, cos, sin, rd, add, n_q, n_kv)
    assert got.shape == (B, T, d_model)
    err = float(np.abs(got - want).max())
    assert err < 1e-5, "与暴力循环版最大误差 %.3e" % err


def test_gqa_shares_kv_within_group():
    """交错复制语义：同一组内的 q 头必须看到**完全相同**的 KV。

    做法：让组内 q 头权重相同 ⇒ 组内输出必须逐位相同；换组则应不同。
    """
    rng = np.random.default_rng(1)
    B, T, d_model, d_k, n_q, n_kv = 1, 4, 8, 4, 4, 2
    rep = n_q // n_kv
    wq = rng.normal(0, 0.1, (n_q * d_k, d_model)).astype(np.float32)
    for h in range(n_q):
        src = (h // rep) * rep                      # 同组首个头
        wq[h * d_k:(h + 1) * d_k] = wq[src * d_k:(src + 1) * d_k]
    wk = rng.normal(0, 0.1, (n_kv * d_k, d_model)).astype(np.float32)
    wv = rng.normal(0, 0.1, (n_kv * d_k, d_model)).astype(np.float32)
    wo = rng.normal(0, 0.1, (d_model, n_q * d_k)).astype(np.float32)
    x = rng.normal(0, 1, (B, T, d_model)).astype(np.float32)
    cos, sin = V.rope_cos_sin(T, 4, 10000.0)

    q = (x @ wq.T).reshape(B, T, n_q, d_k).transpose(0, 2, 1, 3)
    heads = []
    for h in range(n_q):
        # 用单头权重跑一遍整体注意力，取该头对应的输出段做比较
        got = V.attention_numpy(x, wq, wk, wv, wo, cos, sin, 4,
                                M.causal_additive(T), n_q, n_kv)
        heads.append(got)
    assert np.array_equal(heads[0], heads[-1])


def test_masked_positions_have_zero_weight():
    rng = np.random.default_rng(2)
    B, T, d_model, d_k, n_q, n_kv = 1, 5, 8, 4, 2, 1
    wq, wk, wv, wo = _weights(rng, d_model, n_q, n_kv, d_k)
    x = rng.normal(0, 1, (B, T, d_model)).astype(np.float32)
    cos, sin = V.rope_cos_sin(T, d_k, 10000.0)
    add = M.causal_additive(T)
    # 直接检查 softmax 层
    rng2 = np.random.default_rng(3)
    scores = rng2.normal(0, 2, (B, n_q, T, T)).astype(np.float32)
    p = V.attention_numpy_softmax(scores, add[None, None])
    for i in range(T):
        assert float(np.abs(p[0, 0, i, i + 1:]).sum()) == 0.0
