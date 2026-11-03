# -*- coding: utf-8 -*-
"""CUDA Graph 训练封装：参数校验与失败路径（纯 CPU 可测）。

真实捕获/重放需要 NVIDIA GPU，放在 GPU-gated 用例里；结论见
`docs/CUDA_GRAPH_FINDINGS.md`（当前实测只有 1.01×，已不是优化方向）。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

from tgai_go.train.graph_train import GraphStep, GraphStepError  # noqa: E402


class _FakeOpt:
    b1 = 0.9
    b2 = 0.999
    t = 0

    def step(self):
        self.t += 1

    def zero_grad(self):
        pass


def _make(vocab=32):
    return GraphStep(None, _FakeOpt(), ids_shape=(2, 8), tgt_shape=(16,), vocab=vocab)


def test_shape_mismatch_message_says_recapture():
    gs = _make()
    gs.captured = True
    with pytest.raises(GraphStepError) as ei:
        gs.step(np.zeros((4, 8), np.int64), np.zeros(32, np.int64))
    msg = str(ei.value)
    assert "重新 capture" in msg and "修法" in msg


def test_step_before_capture_is_rejected():
    gs = _make()
    with pytest.raises(GraphStepError) as ei:
        gs.step(np.zeros((2, 8), np.int64), np.zeros(16, np.int64))
    assert "capture" in str(ei.value)


def test_host_buffers_are_fixed_address():
    """图里刻的是宿主地址，所以这两块内存必须是同一个对象、只原地改写。"""
    gs = _make()
    a = gs.ids_host
    gs.ids_host[...] = 7
    assert gs.ids_host is a and int(gs.ids_host[0, 0]) == 7


def test_bias_correction_params_captured_from_optimizer():
    gs = _make()
    assert abs(gs._b1 - 0.9) < 1e-12 and abs(gs._b2 - 0.999) < 1e-12


@pytest.mark.skipif(not os.environ.get("TG_RUN_CUDA_TESTS"),
                    reason="需要 GPU：设 TG_RUN_CUDA_TESTS=1")
def test_capture_and_replay_on_gpu():
    import tgai_go.train as T
    from tgai_go.train import qwen as Q

    T.set_default_device("cuda")
    S = 16
    cfg = dict(V=128, D=64, L=1, NH=4, KV=2, FF=128, THETA=1e4, EPS=1e-6, DH=16,
               QKV_BIAS=True, TIE=True)
    w = Q.make_random_weights(cfg, seed=0)
    m = Q.Qwen25(cfg, w)
    opt = T.optim.AdamW(m.params(), lr=1e-4, eps=1e-8)
    gs = GraphStep(m, opt, (2, S), (2 * S,), cfg["V"], clip=0.0)
    nodes = gs.capture(warmup=1)
    assert nodes and nodes > 100
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 100, (2, S)).astype(np.int64)
    tgt = rng.integers(0, 100, (2 * S,)).astype(np.int64)
    gs.step(ids, tgt)
    T._cuda().sync()
    assert opt.t == 2
    gs.close()
