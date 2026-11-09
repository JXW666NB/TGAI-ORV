# -*- coding: utf-8 -*-
"""激活重算（`T.checkpoint`）的**梯度等价性**验证（目标阶段一第 4 项）。

仓库里已有 `checkpoint` 实现（前向不建图 + 反向重算，语义与 torch.utils.checkpoint 一致），
`tests/test_checkpoint_cuda.py` 已覆盖"确实重算了两次 / 输出不连 _parents"。
本文件只补目标要求的**数值判据**：开启 checkpoint 后梯度与不开启一致（≤1e-5）。

显存收益的量化在 GPU 上单独做（见 docs/TRAIN_V27_ORV.md §31），这里只保证正确性。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402

torch = pytest.importorskip("torch")


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def _block_grads(use_ckpt, depth=3):
    """同一组参数、同一输入，分别用/不用 checkpoint 求梯度。"""
    rng = np.random.default_rng(0)
    D, H, N = 8, 12, 6
    X = rng.normal(0, 1, (N, D)).astype(np.float32)
    W1 = T.nn.Parameter(rng.normal(0, 0.3, (D, H)).astype(np.float32))
    W2 = T.nn.Parameter(rng.normal(0, 0.3, (H, D)).astype(np.float32))

    def seg(h):
        out = h
        for _ in range(depth):
            out = (out @ W1)
            out = out * out if False else out            # 保持线性，避免依赖不存在的算子
            out = out @ W2
        return out

    x = T.tensor(np.ascontiguousarray(X), requires_grad=True)
    y = T.checkpoint(seg, x) if use_ckpt else seg(x)
    y.sum().backward()
    return {"W1": np.array(W1.grad.data, np.float32).copy(),
            "W2": np.array(W2.grad.data, np.float32).copy(),
            "x": np.array(x.grad.data, np.float32).copy()}


def test_checkpoint_gradients_match_without_checkpoint():
    a = _block_grads(False)
    b = _block_grads(True)
    for k in a:
        r = _rel(b[k], a[k])
        assert r <= 1e-5, "%s 相对误差 %.2e" % (k, r)


def test_checkpoint_passes_through_without_grad_input():
    """没有任何**输入**需要梯度时直接透传（不建 checkpoint 图），但参数仍然可求导。"""
    W = T.nn.Parameter(np.ones((2, 2), np.float32))
    x = T.tensor(np.ones((1, 2), np.float32))          # 输入不需要梯度
    y = T.checkpoint(lambda z: z @ W, x)
    assert np.allclose(np.asarray(y.data, np.float32),
                       np.ones((1, 2), np.float32) @ np.ones((2, 2), np.float32))
    y.sum().backward()
    assert W.grad is not None                          # 参数侧梯度正常
    assert getattr(x, "grad", None) is None            # 输入侧本来就不需要
