# -*- coding: utf-8 -*-
"""`Module.load_state_dict_stream` 的行为（CPU，进 CI）。

为什么需要它：3B 权重的 dict 版本要求 12 GB 宿主内存（实测 OOM）；
流式版本逐个上传并释放，宿主峰值 = 单张量。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import Linear, RMSNorm  # noqa: E402


def test_stream_matches_dict():
    rng = np.random.default_rng(0)
    w = rng.normal(0, .02, (8, 16)).astype(np.float32)
    a, b = Linear(16, 8, bias=False), Linear(16, 8, bias=False)
    a.load_state_dict({"weight": w})
    b.load_state_dict_stream(iter([("weight", w)]))
    np.testing.assert_array_equal(a.weight.data, b.weight.data)


def test_stream_handles_multiple_modules():
    """按名字一次性装载多个子模块（Linear 的权重 + RMSNorm 的权重）。"""
    rng = np.random.default_rng(1)
    m = Linear(16, 8, bias=False)
    n = RMSNorm(16)
    w1 = rng.normal(0, .02, (8, 16)).astype(np.float32)
    w2 = rng.normal(0, .02, (16,)).astype(np.float32)
    items = iter([("weight", w1)])
    m.load_state_dict_stream(items)
    n.load_state_dict_stream(iter([("weight", w2)]))
    np.testing.assert_array_equal(m.weight.data, w1)
    np.testing.assert_array_equal(n.weight.data, w2)


def test_stream_rejects_missing():
    m = Linear(16, 8, bias=False)
    with pytest.raises(KeyError):
        m.load_state_dict_stream(iter([]))


def test_stream_rejects_bad_shape():
    m = Linear(16, 8, bias=False)
    with pytest.raises(ValueError):
        m.load_state_dict_stream(iter([("weight", np.zeros((4, 4), np.float32))]))


def test_stream_rejects_unknown_name():
    m = Linear(16, 8, bias=False)
    with pytest.raises(KeyError):
        m.load_state_dict_stream(iter([("nope", np.zeros((8, 16), np.float32))]))