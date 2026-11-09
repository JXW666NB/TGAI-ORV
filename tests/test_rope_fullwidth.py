# -*- coding: utf-8 -*-
"""第 2 步的数学前提：rope 能否一次作用在全宽张量上（表按头平铺），与逐头结果一致？

若通过 ⇒ 后续可把 16 次 rope 合成 1 次（每次少 30+ 次发射）。
若性质不同（例如 rope 依赖最后一维长度），则第 2 步只能停在"rope 仍逐头"。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402

DT, NH, DH, SEQ, THETA = 16, 4, 4, 5, 10000.0


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def _tables():
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def test_rope_full_width_equals_per_head():
    cos_np, sin_np = _tables()
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (SEQ, NH * DH)).astype(np.float32)
    cos = T.tensor(cos_np)
    sin = T.tensor(sin_np)

    # (a) 逐头
    xt = T.tensor(x.copy())
    parts = []
    for i in range(NH):
        sl = slice(i * DH, (i + 1) * DH)
        parts.append(np.asarray(T._rope_composed(xt[:, sl], cos, sin).data, np.float32))
    ref = np.concatenate(parts, axis=1)

    # (b) 全宽 + 平铺表
    tiled_cos = T.tensor(np.tile(cos_np, (1, NH)))
    tiled_sin = T.tensor(np.tile(sin_np, (1, NH)))
    got = np.asarray(T._rope_composed(T.tensor(x.copy()), tiled_cos, tiled_sin).data, np.float32)

    assert _rel(got, ref) <= 1e-6, ("全宽 rope vs 逐头", _rel(got, ref))
