# -*- coding: utf-8 -*-
"""第二批核心算子（归约/损失/形状/采样/SGD）与 PyTorch 的对照测试。"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import nn_ops as O

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


def _t(a, grad=True):
    return T.tensor(np.ascontiguousarray(np.asarray(a, dtype=np.float32)),
                    requires_grad=grad)


X = np.array([[-1.0, 3.0, 2.0], [0.5, -2.0, 1.0], [3.0, 3.0, 3.0]], dtype=np.float32)


def test_argmax():
    x = _t(X, grad=False)
    for dim in (None, 0, 1):
        got = np.asarray(O.argmax(x, dim=dim).to_numpy(), dtype=np.int64)
        want = torch.argmax(torch.from_numpy(X), dim=dim).numpy()
        assert np.array_equal(got, want), dim


@pytest.mark.parametrize("name,tfn", [("amax", torch.amax), ("amin", torch.amin)])
def test_amax_amin_forward_bwd_with_ties(name, tfn):
    """并列最大值时 torch 会把梯度均分 —— 我们的实现必须一致。"""
    x = _t(X)
    y = getattr(O, name)(x, dim=1)
    y.sum().backward()

    tx = torch.from_numpy(X.copy()).requires_grad_(True)
    ty = tfn(tx, dim=1)
    ty.sum().backward()
    assert np.allclose(y.to_numpy(), ty.detach().numpy(), rtol=1e-6, atol=1e-6)
    assert np.allclose(x.grad, tx.grad.numpy(), rtol=1e-6, atol=1e-6)


def test_norm_forward_and_backward():
    for kw in (dict(), dict(dim=1), dict(dim=0, keepdim=True)):
        x = _t(X)
        y = O.norm(x, **kw)
        y.sum().backward()
        tx = torch.from_numpy(X.copy()).requires_grad_(True)
        ty = torch.norm(tx, **kw)
        ty.sum().backward()
        assert np.allclose(y.to_numpy(), ty.detach().numpy(), rtol=1e-5, atol=1e-6), kw
        assert np.allclose(x.grad, tx.grad.numpy(), rtol=1e-5, atol=1e-6), kw


def test_norm_only_supports_p2():
    with pytest.raises(ValueError) as ei:
        O.norm(_t(X), p=1.0)
    assert "p=2" in str(ei.value)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_mse_loss(reduction):
    a_np = X
    b_np = np.zeros_like(X)
    a, b = _t(a_np), _t(b_np)
    loss = O.mse_loss(a, b, reduction=reduction)
    loss.backward()
    ta = torch.from_numpy(a_np.copy()).requires_grad_(True)
    tb = torch.from_numpy(b_np.copy()).requires_grad_(True)
    tloss = torch.nn.functional.mse_loss(ta, tb, reduction=reduction)
    tloss.backward()
    assert np.allclose(float(loss.to_numpy()), float(tloss), rtol=1e-6)
    assert np.allclose(a.grad, ta.grad.numpy(), rtol=1e-5, atol=1e-6)


def test_nll_loss_forward_and_backward():
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(4, 6)).astype(np.float32)
    tgt = np.array([0, 5, 2, 3], dtype=np.int64)
    lp = torch.log_softmax(torch.from_numpy(logits), dim=-1).numpy()

    x = _t(lp)
    loss = O.nll_loss(x, tgt)
    loss.backward()

    tlp = torch.from_numpy(lp.copy()).requires_grad_(True)
    tloss = torch.nn.functional.nll_loss(tlp, torch.from_numpy(tgt))
    tloss.backward()
    assert np.allclose(float(loss.to_numpy()), float(tloss), rtol=1e-6)
    assert np.allclose(x.grad, tlp.grad.numpy(), rtol=1e-6, atol=1e-7)


def test_nll_loss_errors_name_fix():
    x = _t(np.zeros((4, 6), np.float32))
    with pytest.raises(ValueError) as ei:
        O.nll_loss(x, np.array([0, 9, 1, 2], np.int64))
    msg = str(ei.value)
    assert "越界" in msg and "修法" in msg, msg

    with pytest.raises(ValueError) as ei2:
        O.nll_loss(x, np.array([0, 1], np.int64))
    assert "样本数" in str(ei2.value)


def test_chunk_matches_torch():
    x_np = np.arange(24, dtype=np.float32).reshape(6, 4)
    for chunks in (2, 3, 4):
        got = [_n(p) for p in O.chunk(_t(x_np), chunks, dim=0)]
        want = [t.numpy() for t in torch.chunk(torch.from_numpy(x_np), chunks, dim=0)]
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert g.shape == w.shape and np.allclose(g, w), (chunks, g.shape, w.shape)


def test_tril_matches_torch():
    x_np = np.arange(16, dtype=np.float32).reshape(4, 4)
    for k in (0, 1, -1):
        got = _n(O.tril(_t(x_np), diagonal=k))
        want = torch.tril(torch.from_numpy(x_np), diagonal=k).numpy()
        assert np.allclose(got, want), k


def test_arange_matches_torch():
    for a, b, s in ((0, 5, 1), (1, 10, 3), (2.5, 0, -0.5)):
        got = _n(O.arange(a, b, s))
        want = torch.arange(a, b, s).numpy()
        assert np.allclose(got, want), (a, b, s)
    with pytest.raises(ValueError):
        O.arange(0, 5, 0)


def test_topk_indices():
    x = _t(np.array([[0.1, 5.0, 3.0, 4.0]], np.float32), grad=False)
    idx = O.topk_indices(x, 2, dim=1)
    assert idx.shape == (1, 2)
    assert set(idx[0].tolist()) == {1, 3}          # 5.0 与 4.0 的位置
    with pytest.raises(ValueError) as ei:
        O.topk_indices(x, 9, dim=1)
    assert "k=9" in str(ei.value) and "修法" in str(ei.value)


def test_sgd_step_matches_torch():
    """几步之后参数应与 torch.optim.SGD(momentum) 一致。"""
    p0 = np.array([[1.0, -2.0], [0.5, 3.0]], np.float32)
    grads = [np.full((2, 2), 0.1, np.float32), np.full((2, 2), -0.2, np.float32),
             np.full((2, 2), 0.05, np.float32)]

    # ★ T.tensor 与 numpy **共享内存**，SGD 会原地改写 p0；
    #   所以 torch 侧必须用副本，否则会拿"被改过的初值"去算（我第一版就踩了）
    p0_ref = p0.copy()
    ours = _t(p0)
    state = None
    for g in grads:
        ours.grad = g
        state = O.sgd_step([ours], lr=0.1, momentum=0.9, weight_decay=0.01, state=state)

    tp = torch.nn.Parameter(torch.from_numpy(p0_ref))
    opt = torch.optim.SGD([tp], lr=0.1, momentum=0.9, weight_decay=0.01)
    for g in grads:
        tp.grad = torch.from_numpy(g.copy())
        opt.step()
    assert np.allclose(_n(ours), tp.detach().numpy(), rtol=1e-5, atol=1e-6)


def _n(x):
    return np.asarray(x.to_numpy(), dtype=np.float32)
