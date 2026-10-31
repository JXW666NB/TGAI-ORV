# -*- coding: utf-8 -*-
"""梯度检查点：数值必须与不检查点**完全一致**，而保留的中间张量要大幅减少。

为什么必须测数值：checkpoint 一旦写错（比如重算时没打开 grad、或梯度被算两次），
训练**照样能跑**、loss 照样降，只是学错了 —— 和"反向写错"同一类问题。

为什么也要测内存：如果只测数值，一个"什么都不省"的实现也能通过 ——
那就失去了这项功能的意义。
"""
import gc

import numpy as np
import pytest

import tgai_go.train as T


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


def _layer(h):
    return (h * 1.7 + 0.3).tanh()


def _grad(x):
    g = x.grad
    return np.asarray(g.to_numpy() if hasattr(g, "to_numpy") else g, np.float64)


def _live_tensors():
    return sum(1 for o in gc.get_objects() if isinstance(o, T.Tensor))


def test_checkpoint_grads_match_plain():
    vals = np.random.default_rng(0).random((3, 5)).astype(np.float32)

    x1 = T.tensor(vals.copy(), requires_grad=True)
    h = x1
    for _ in range(5):
        h = _layer(h)
    (h * h).sum().backward()

    x2 = T.tensor(vals.copy(), requires_grad=True)
    g = x2
    for _ in range(5):
        g = T.checkpoint(_layer, g)
    (g * g).sum().backward()

    assert np.allclose(h.to_numpy(), g.to_numpy(), rtol=1e-6)
    assert np.allclose(_grad(x1), _grad(x2), rtol=1e-6, atol=1e-8)


def test_nested_checkpoint():
    vals = np.random.default_rng(1).random((2, 4)).astype(np.float32)

    def block(h):
        return T.checkpoint(_layer, T.checkpoint(_layer, h))

    x1 = T.tensor(vals.copy(), requires_grad=True)
    y1 = block(_layer(x1))
    (y1 * y1).sum().backward()

    x2 = T.tensor(vals.copy(), requires_grad=True)
    y2 = _layer(_layer(_layer(x2)))
    (y2 * y2).sum().backward()

    assert np.allclose(_grad(x1), _grad(x2), rtol=1e-5, atol=1e-7)


def test_checkpoint_sequential_matches():
    vals = np.random.default_rng(2).random((2, 3)).astype(np.float32)
    fns = [_layer, _layer, _layer]

    x1 = T.tensor(vals.copy(), requires_grad=True)
    y1 = T.checkpoint_sequential(fns, x1)
    (y1 * y1).sum().backward()

    x2 = T.tensor(vals.copy(), requires_grad=True)
    y2 = x2
    for fn in fns:
        y2 = fn(y2)
    (y2 * y2).sum().backward()
    assert np.allclose(_grad(x1), _grad(x2), rtol=1e-6)


def test_checkpoint_without_grad_input_is_passthrough():
    x = T.tensor(np.ones((2, 2), np.float32))
    y = T.checkpoint(_layer, x)
    assert np.allclose(y.to_numpy(), _layer(x).to_numpy())


def test_checkpoint_rejects_multi_output_with_fix():
    x = T.tensor(np.ones((2, 2), np.float32), requires_grad=True)
    with pytest.raises(TypeError) as ei:
        T.checkpoint(lambda t: (t * 2.0, t * 3.0), x)
    msg = str(ei.value)
    assert "checkpoint" in msg and "修法" in msg


def test_checkpoint_actually_reduces_retained_intermediates():
    """保留的中间张量必须显著减少 —— 否则这个功能没有意义。"""
    n = 30
    gc.collect()
    base = _live_tensors()

    x = T.tensor(np.random.rand(32, 64).astype(np.float32), requires_grad=True)
    h = x
    for _ in range(n):
        h = _layer(h)
    plain = _live_tensors() - base
    del h, x
    gc.collect()

    base = _live_tensors()
    x2 = T.tensor(np.random.rand(32, 64).astype(np.float32), requires_grad=True)
    g = x2
    for _ in range(n):
        g = T.checkpoint(_layer, g)
    ckpt = _live_tensors() - base

    assert ckpt < plain * 0.5, (
        "checkpoint 之后保留的张量 %d，不检查点是 %d —— 没省下显存，实现有问题"
        % (ckpt, plain))
