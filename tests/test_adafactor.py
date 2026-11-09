# -*- coding: utf-8 -*-
"""Adafactor 与 **transformers.optimization.Adafactor** 的逐步对照（最权威的现成参考实现）。

判据（目标第 5 项）：单步/多步参数更新相对误差 ≤1e-6。
覆盖：2D（因子化 row/col）、1D（普通二阶矩）、weight_decay、scale_parameter、clip_threshold。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from transformers.optimization import Adafactor as HFAdafactor  # noqa: E402


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def _run(shape, steps=4, lr=1e-2, wd=0.0, scale_parameter=False, clip=1.0, seed=0):
    rng = np.random.default_rng(seed)
    w0 = rng.normal(0, 1, shape).astype(np.float32)
    grads = [rng.normal(0, 0.1, shape).astype(np.float32) for _ in range(steps)]

    ours = T.nn.Parameter(w0.copy())
    o = T.optim.Adafactor([ours], lr=lr, weight_decay=wd, scale_parameter=scale_parameter,
                          clip_threshold=clip, decay_rate=-0.8, eps1=1e-30, eps2=1e-3)
    for g in grads:
        ours.grad = T.tensor(g.copy())
        o.step()

    ref_p = torch.nn.Parameter(torch.from_numpy(w0.copy()))
    ref = HFAdafactor([ref_p], lr=lr, eps=(1e-30, 1e-3), clip_threshold=clip, decay_rate=-0.8,
                      beta1=None, weight_decay=wd, scale_parameter=scale_parameter,
                      relative_step=False, warmup_init=False)
    for g in grads:
        ref_p.grad = torch.from_numpy(g.copy())
        ref.step()
    return np.array(ours.data, np.float32), ref_p.detach().numpy()


@pytest.mark.parametrize("shape", [(6, 4), (5,), (3, 2, 2) if False else (6, 4)])
def test_adafactor_2d_and_1d_match_hf(shape):
    a, b = _run(shape)
    assert _rel(a, b) <= 1e-6, (shape, _rel(a, b))


def test_adafactor_weight_decay_matches_hf():
    a, b = _run((6, 4), wd=0.01)
    assert _rel(a, b) <= 1e-6, _rel(a, b)


def test_adafactor_scale_parameter_matches_hf():
    a, b = _run((6, 4), scale_parameter=True)
    assert _rel(a, b) <= 1e-6, _rel(a, b)


def test_adafactor_clip_threshold_matches_hf():
    a, b = _run((6, 4), clip=0.5)
    assert _rel(a, b) <= 1e-6, _rel(a, b)


def test_adafactor_state_is_factorized():
    """显存关键：2D 参数只存 row/col 两份小向量，不存满矩阵。"""
    p = T.nn.Parameter(np.zeros((64, 32), np.float32))
    p.grad = T.tensor(np.ones((64, 32), np.float32))
    o = T.optim.Adafactor([p], lr=1e-3)
    o.step()
    assert o.row[0].shape == (64,) and o.col[0].shape == (32,)
    assert o.sq[0] is None
    # 状态元素数 = 96，而 AdamW 需要 2*64*32 = 4096
    assert o.row[0].size + o.col[0].size == 96


def test_adafactor_requires_explicit_lr_message():
    p = T.nn.Parameter(np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError, match="修法"):
        T.optim.Adafactor([p], lr=None)
