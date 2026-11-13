# -*- coding: utf-8 -*-
"""GlobalAttention 的 **ORV 张量实现** 与 numpy 参考对拍。

这是移植里第一个"整块算子真的跑在 ORV 上"的里程碑：投影、部分 RoPE、GQA 交错复制、
缩放点积、拼头、输出投影，全部走 ORV 的张量与 `scaled_dot_product_attention`。

## 为什么值得单独一个文件

numpy 参考本身已经被"三重循环暴力版"验证过（`test_v3_attention.py`），
所以这里只要 ORV 版与 numpy 版一致，就等于 ORV 版与暴力版一致 —— 链条是闭合的。

## 已知限制（写在测试里，避免误以为已完全对齐）

ORV 的 `scaled_dot_product_attention` **只有 `causal` 布尔，没有加性掩码入口**，
所以打包训练要的块对角 `seg_mask` 还没支持。这条限制在 `attention_orv` 的
docstring 里也标了，属于"下一步"，不是"已完成"。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import v3_ops as V  # noqa: E402
from tgai_go.train import v3_masks as M  # noqa: E402


def _np(t):
    r = t.to_numpy() if hasattr(t, "to_numpy") else t
    return np.asarray(r, np.float32)


@pytest.mark.parametrize("n_q,n_kv", [(4, 4), (4, 2)])
def test_orv_attention_matches_numpy(n_q, n_kv):
    rng = np.random.default_rng(0)
    B, T, dm, d_k = 2, 6, 16, 8
    wq = rng.normal(0, .1, (n_q * d_k, dm)).astype(np.float32)
    wk = rng.normal(0, .1, (n_kv * d_k, dm)).astype(np.float32)
    wv = rng.normal(0, .1, (n_kv * d_k, dm)).astype(np.float32)
    wo = rng.normal(0, .1, (dm, n_q * d_k)).astype(np.float32)
    x = rng.normal(0, 1, (B, T, dm)).astype(np.float32)
    cos, sin = V.rope_cos_sin(T, d_k, 10000.0)

    got = _np(V.attention_orv(V.T.tensor(x), V.T.tensor(wq), V.T.tensor(wk),
                              V.T.tensor(wv), V.T.tensor(wo),
                              V.T.tensor(cos), V.T.tensor(sin),
                              d_k, n_q, n_kv, causal=True))
    want = V.attention_numpy(x, wq, wk, wv, wo, cos, sin, d_k,
                             M.causal_additive(T), n_q, n_kv)
    assert got.shape == (B, T, dm)
    err = float(np.abs(got - want).max())
    assert err < 1e-5, "ORV 版与 numpy 参考最大误差 %.3e" % err


def test_orv_attention_causal_actually_hides_future():
    """因果性必须真的生效：改最后一个位置**不能**影响前面位置的输出。

    （参考实现曾经在卷积上用双向填充泄漏未来，所以这类断言不是形式主义。）
    """
    rng = np.random.default_rng(1)
    B, T, dm, d_k, n_q, n_kv = 1, 8, 8, 4, 2, 2
    wq = rng.normal(0, .1, (n_q * d_k, dm)).astype(np.float32)
    wk = rng.normal(0, .1, (n_kv * d_k, dm)).astype(np.float32)
    wv = rng.normal(0, .1, (n_kv * d_k, dm)).astype(np.float32)
    wo = rng.normal(0, .1, (dm, n_q * d_k)).astype(np.float32)
    x = rng.normal(0, 1, (B, T, dm)).astype(np.float32)
    cos, sin = V.rope_cos_sin(T, d_k, 10000.0)

    def run(xx):
        return _np(V.attention_orv(V.T.tensor(xx), V.T.tensor(wq), V.T.tensor(wk),
                                   V.T.tensor(wv), V.T.tensor(wo),
                                   V.T.tensor(cos), V.T.tensor(sin),
                                   d_k, n_q, n_kv, causal=True))

    y0 = run(x)
    x2 = x.copy()
    x2[0, -1] += 5.0
    y1 = run(x2)
    assert np.allclose(y0[:, :-1], y1[:, :-1], atol=1e-6), \
        "改动最后一个位置却改变了前面的输出 ⇒ 因果掩码没生效"
    assert not np.allclose(y0[:, -1], y1[:, -1]), "最后一个位置本应变化"
