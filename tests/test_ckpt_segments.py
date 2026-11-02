# -*- coding: utf-8 -*-
"""`checkpoint_sequential(..., segments=N)`：粒度可调，数值不变。

这是"给用户自由度"的具体落点（PyTorch 同签名）：
  * segments = L → 每层一次检查点（最省显存、重算最频繁）
  * segments = 1 → 整段一次（只存段边界）
数值必须与"完全不做检查点"逐位一致；变的只是峰值显存与重算次数。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

import tgai_go.train as T

CFG = dict(V=64, D=32, L=3, NH=4, KV=2, FF=64, THETA=1e6, EPS=1e-6, DH=8,
           QK_NORM=False, QKV_BIAS=True, TIE=True)


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


def _model(**cfg_over):
    from tgai_go.train import qwen as Q
    cfg = dict(CFG, **cfg_over)
    return Q.Qwen25(cfg, Q.make_random_weights(CFG, seed=0))


def _grads(model, ids):
    out = model.forward(ids)
    (out * out).sum().backward()
    return {k: np.asarray(p.grad if p.grad is not None else 0, np.float64).copy()
            for k, p in model.p.items()}


def test_segments_do_not_change_gradients():
    ids = np.random.default_rng(0).integers(0, 60, (2, 8)).astype(np.int64)
    ref = _grads(_model(GRAD_CKPT=False), ids)
    for segs in (1, 2, 3):
        g = _grads(_model(GRAD_CKPT=True, CKPT_SEGMENTS=segs), ids)
        worst = max(float(np.abs(ref[k] - g[k]).max()) for k in ref)
        assert worst == 0.0, "segments=%d 的梯度必须逐位一致（差 %g）" % (segs, worst)


def test_segment_count_controls_checkpoint_calls(monkeypatch):
    """segments=L 应产生 L 次 checkpoint 调用；segments=1 只 1 次。"""
    from tgai_go.train import qwen as Q
    ids = np.random.default_rng(1).integers(0, 60, (1, 8)).astype(np.int64)
    calls = {"n": 0}
    real = T.checkpoint

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(T, "checkpoint", counting)
    for segs, expect in ((3, 3), (1, 1), (2, 2)):
        calls["n"] = 0
        cfg = dict(CFG, GRAD_CKPT=True, CKPT_SEGMENTS=segs)
        m = Q.Qwen25(cfg, Q.make_random_weights(CFG, seed=0))
        m.forward(ids)
        assert calls["n"] == expect, "segments=%d 应产生 %d 次检查点，实际 %d" % (
            segs, expect, calls["n"])


def test_segments_out_of_range_is_clamped():
    from tgai_go.train import qwen as Q
    ids = np.random.default_rng(2).integers(0, 60, (1, 8)).astype(np.int64)
    for segs in (0, -5, 999):
        cfg = dict(CFG, GRAD_CKPT=True, CKPT_SEGMENTS=segs)
        m = Q.Qwen25(cfg, Q.make_random_weights(CFG, seed=0))
        out = m.forward(ids)                     # 不该抛异常
        assert tuple(out.shape) == (1, 8, CFG["V"])
