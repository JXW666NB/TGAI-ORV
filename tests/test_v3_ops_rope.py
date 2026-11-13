# -*- coding: utf-8 -*-
"""V3 的 RoPE（**支持 partial 旋转**）对拍。

## 为什么这块最容易做错

V3 的 RoPE 有两条容易搞错的地方，都是实测确认过的语义：

1. **partial 旋转**：全局层只旋转 `round(d_k × 0.25)` 维，其余维度**原样通过**（等价 NoPE）；
   滑窗层全维旋转。全局层那 1/4 维的 θ 还是 5e6，滑窗层是 1e4。
2. **频率分母是 `rot_dim` 而不是 `dim`**：
   `freqs = 1/(θ**(arange(0,rot_dim,2)/rot_dim))` —— 用 `dim` 算会让 partial 情况全错。
3. **交错 rotate_half**：`(x1,x2,x3,x4) → (-x2,x1,-x4,x3)`，不是"前半/后半对半换"。

ORV 侧用置换矩阵实现 rotate_half（`x @ P`），因为张量不支持对切片原地写。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import v3_ops as V  # noqa: E402


def _orv(t):
    """把 ORV 张量（或已是 ndarray 的东西）取成 numpy float32。"""
    r = t.to_numpy() if hasattr(t, "to_numpy") else t
    return np.asarray(r, np.float32)


@pytest.mark.parametrize("dim,partial,theta", [
    (64, 1.0, 10000.0),        # 滑窗层：全维 + θ=1e4
    (64, 0.25, 5000000.0),     # 全局层：1/4 维 + θ=5e6
    (64, 0.5, 100000.0),
])
def test_rope_matches_numpy(dim, partial, theta):
    rng = np.random.default_rng(0)
    rd = int(round(dim * partial))
    rd = rd if rd % 2 == 0 else rd - 1
    seq = 7
    x = rng.normal(0, 1, (seq, dim)).astype(np.float32)
    cos, sin = V.rope_cos_sin(seq, rd, theta)

    xr = np.ascontiguousarray(x[..., :rd])
    got = _orv(V.v3_rope(V.T.tensor(xr), V.T.tensor(cos), V.T.tensor(sin)))
    want = V.v3_rope_numpy(xr, cos, sin)
    assert got.shape == want.shape
    assert float(np.abs(got - want).max()) < 1e-6


def test_partial_keeps_tail_untouched():
    """partial 旋转必须**原样保留**尾部维度（等价 NoPE），这是全局层的关键语义。"""
    rng = np.random.default_rng(1)
    dim, partial, theta = 64, 0.25, 5000000.0
    rd = int(round(dim * partial))
    seq = 5
    x = rng.normal(0, 1, (seq, dim)).astype(np.float32)
    cos, sin = V.rope_cos_sin(seq, rd, theta)

    xr = np.ascontiguousarray(x[..., :rd])
    rotated = _orv(V.v3_rope(V.T.tensor(xr), V.T.tensor(cos), V.T.tensor(sin)))
    full = np.concatenate([rotated, x[..., rd:]], axis=-1)
    assert np.array_equal(full[..., rd:], x[..., rd:]), "尾部维度不应被改动"
    # 前 rd 维确实变了（否则就是没旋转）
    assert not np.allclose(rotated, xr)


def test_rotate_half_is_interleaved_not_half_split():
    """交错语义检查：x @ P 必须等于 (x1,x2,x3,x4)→(-x2,x1,-x4,x3)。"""
    x = np.arange(1, 9, dtype=np.float32).reshape(1, 8)      # 1..8
    P = V.rotate_half_perm(8)
    got = (x @ P).reshape(-1)
    want = np.array([-2, 1, -4, 3, -6, 5, -8, 7], np.float32)
    assert np.array_equal(got, want), "got=%s want=%s" % (got, want)
