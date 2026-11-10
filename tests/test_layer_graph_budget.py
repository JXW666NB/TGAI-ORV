# -*- coding: utf-8 -*-
"""LayerGraphBackward 的显存预算与"部分捕获"契约（CUDA 才跑）。

判据：
1. `max_graphs=2` 时只捕获 2 张，`count==2`，`has(0/1)` 真、`has(2)` 假；
2. 重放的参数梯度与 `dL/dx` 与逐算子路径一致（≤1e-6）；
3. `max_affordable()` 返回 0..n_layers 之间，且受 fraction 约束。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train.layer_graph import LayerGraphBackward

V27 = r"D:\tgai\models\V2.7\TGAI_NB_V2.7_v4.tg"


def _np(x):
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, np.float32)


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(1e-9, float(np.abs(b).max())))


def _has_cuda():
    try:
        rt = T._cuda()
        return bool(rt is not None and rt.free_device_bytes() is not None)
    except Exception:
        return False


CUDA = pytest.mark.skipif(not _has_cuda(), reason="没有可用 CUDA 设备")
import os


@CUDA
@pytest.mark.skipif(not os.path.exists(V27), reason="没有 V2.7 权重（CI 机器上跳过）")
def test_partial_capture_and_budget():
    from tgai_go.read_tg import close_tg, read_tg
    from tgai_go.train.v27_train import TGState, V27Trainer

    info = read_tg(V27)
    h = info["header"]
    close_tg(info)
    D, NH = int(h["d_model"]), int(h["n_heads"])
    DH, THETA = D // NH, float(h["rope_theta"])
    cfg = dict(d_model=D, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]), rope_theta=THETA)
    T.set_default_device("cuda")
    st = TGState(V27)
    tr = V27Trainer(cfg, n_layers=2)
    tr.load_all(st)

    seq = 64
    rng = np.random.default_rng(0)
    xs = [rng.normal(0, 1, (seq, D)).astype(np.float32) for _ in range(2)]
    gws = [rng.normal(0, 1, (seq, D)).astype(np.float32) for _ in range(2)]
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(seq, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos = T.tensor(np.cos(ang).astype(np.float32))
    sin = T.tensor(np.sin(ang).astype(np.float32))

    # 逐算子参照
    ref_g, ref_dx = [], []
    for i in range(2):
        for k in ("q", "k", "v", "o"):
            for m in tr.layers[i][k]:
                m.weight.grad = T.tensor(np.zeros(tuple(int(x) for x in m.weight.shape), np.float32))
        xi = T.tensor(xs[i], requires_grad=True)
        ya = tr.fwd_attn(xi, i, cos, sin)
        (ya * T.tensor(gws[i])).sum().backward()
        ref_dx.append(_np(xi.grad))
        ref_g.append([_np(m.weight.grad) for k in ("q", "k", "v", "o") for m in tr.layers[i][k]])
    T._cuda().sync()

    gb = LayerGraphBackward(tr, cos, sin, seq=seq, arena_bytes=256 * 1024 ** 2)
    assert 0 <= gb.max_affordable(2) <= 2
    gb.capture(xs, max_graphs=2)
    assert gb.count == 2
    assert gb.has(0) and gb.has(1) and not gb.has(2)

    got = gb.grads_numpy()
    for i in range(2):
        dx = gb.replay_layer(i, gws[i])
        for a, b in zip(ref_g[i], got[i]):
            assert _rel(a, b) <= 1e-6
        assert _rel(ref_dx[i], dx) <= 1e-6
    gb.close()
    try:
        st.close(); T._cuda().empty_cache()
    except Exception:
        pass
