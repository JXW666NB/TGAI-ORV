# -*- coding: utf-8 -*-
"""梯度检查点接进模型后：前向/反向必须与不检查点**完全一致**。"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

import tgai_go.train as T
from tgai_go.train import qwen as Q

CFG = dict(V=64, D=32, L=3, NH=4, KV=2, FF=64, THETA=1e6, EPS=1e-6, DH=8,
           QK_NORM=False, QKV_BIAS=True, TIE=True)


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


def _build(ckpt):
    cfg = dict(CFG, GRAD_CKPT=ckpt)
    return Q.Qwen25(cfg, Q.make_random_weights(CFG, seed=0))


def _run(model, ids):
    out = model.forward(ids)
    s = (out * out).sum()
    s.backward()
    grads = {k: np.asarray(p.grad, np.float64).copy()
             for k, p in model.params_dict().items()} if hasattr(model, "params_dict") else {}
    return np.asarray(out.to_numpy(), np.float64), grads, model


def test_checkpoint_forward_is_identical():
    ids = np.random.default_rng(0).integers(0, 60, (2, 16)).astype(np.int64)
    a = _build(False)
    b = _build(True)
    la = np.asarray(a.forward(ids).to_numpy(), np.float64)
    lb = np.asarray(b.forward(ids).to_numpy(), np.float64)
    assert np.allclose(la, lb, rtol=0, atol=0), "前向必须逐位一致"


def test_checkpoint_backward_grads_are_identical():
    ids = np.random.default_rng(1).integers(0, 60, (2, 16)).astype(np.int64)
    a, b = _build(False), _build(True)
    for m in (a, b):
        s = (m.forward(ids) ** 2).sum()
        s.backward()
    worst = 0.0
    for k, pa in a.p.items():
        pb = b.p[k]
        ga = np.asarray(pa.grad if pa.grad is not None else 0, np.float64)
        gb = np.asarray(pb.grad if pb.grad is not None else 0, np.float64)
        worst = max(worst, float(np.abs(ga - gb).max()) if ga.size else 0.0)
    assert worst == 0.0, "梯度必须逐位一致，实测最大差 %g" % worst


def test_switch_defaults_off_and_reads_cfg():
    assert _build(False).use_ckpt is False
    assert _build(True).use_ckpt is True
