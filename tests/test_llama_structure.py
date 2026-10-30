# -*- coding: utf-8 -*-
"""把「LLaMA 结构训练对齐」固化成 CI 用例。

为什么要有：`examples/llama_tiny.py` 是一次性验证，算子一改就可能悄悄回归。
这里用 3 步、极小配置跑一遍，与 PyTorch 孪生对比 loss 曲线 —— 纯 CPU，CI 可跑。
"""
import importlib.util
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_example():
    path = os.path.join(REPO, "examples", "llama_tiny.py")
    spec = importlib.util.spec_from_file_location("llama_tiny", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_llama_structure_trains_like_torch():
    m = _load_example()
    cfg = dict(m.CFG)
    cfg.update(D=32, L=1, NH=2, KV=1, FF=64, S=8, V=32)   # 缩小以保持 CI 快

    rng = np.random.default_rng(5)
    ids = rng.integers(0, cfg["V"], size=cfg["S"]).astype(np.int64)
    tgt = rng.integers(0, cfg["V"], size=cfg["S"]).astype(np.int64)
    w = m.init_weights(cfg)

    steps, lr = 3, 0.05
    ours_loss, ours_w = m.ours_train(w, ids, tgt, steps, lr, cfg)
    ref_loss, ref_w = m.torch_train(w, ids, tgt, steps, lr, cfg, torch)

    for o, r in zip(ours_loss, ref_loss):
        rel = abs(o - r) / max(abs(r), 1e-9)
        assert rel <= 1e-3, (o, r, rel)

    param_rel = max(
        float(np.abs(ours_w[k].astype(np.float64) - ref_w[k].astype(np.float64)).max()
              / (np.abs(ref_w[k]).max() + 1e-12)) for k in ours_w)
    assert param_rel <= 1e-3, param_rel

    # 梯度确实在起作用：loss 必须下降
    assert ours_loss[-1] < ours_loss[0]
