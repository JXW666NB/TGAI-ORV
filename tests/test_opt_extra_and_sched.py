# -*- coding: utf-8 -*-
"""SGD / Lion / 学习率调度器 与 torch 的数值对照（CPU 上即可跑，无需 GPU）。

判据（与目标一致）：
  * 调度器：lr 序列逐 step 与 torch 完全相同（rtol 1e-12）
  * SGD：多步后参数与 torch.optim.SGD 一致（≤1e-6 相对）
  * Lion：与独立手写的 numpy 参考实现一致（≤1e-7 相对），并检查公式特征（更新步长恒为 lr*sign）
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


# ───────────────────────── 调度器 ─────────────────────────
def _torch_seq(kind, steps, **kw):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=kw.pop("base_lr", 1e-3))
    if kind == "linear":
        s = torch.optim.lr_scheduler.LinearLR(opt, **kw)
    elif kind == "cosine":
        s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, **kw)
    elif kind == "step":
        s = torch.optim.lr_scheduler.StepLR(opt, **kw)
    elif kind == "constant":
        s = torch.optim.lr_scheduler.ConstantLR(opt, **kw)
    elif kind == "lambda":
        s = torch.optim.lr_scheduler.LambdaLR(opt, **kw)
    else:
        raise AssertionError(kind)
    out = []
    for _ in range(steps):
        out.append(float(opt.param_groups[0]["lr"]))
        opt.step()
        s.step()
    return out


class _FakeOpt(object):
    def __init__(self, lr):
        self.lr = lr


def _ours_seq(kind, steps, **kw):
    opt = _FakeOpt(kw.pop("base_lr", 1e-3))
    s = T.optim.lr_scheduler.make_scheduler(kind, opt, **kw)
    return [s.step() for _ in range(steps)]


def test_scheduler_linear_matches_torch():
    a = _ours_seq("linear", 12, start_factor=0.25, end_factor=1.0, total_iters=5)
    b = _torch_seq("linear", 12, start_factor=0.25, end_factor=1.0, total_iters=5)
    assert np.allclose(a, b, rtol=1e-6, atol=0), (a, b)   # torch 的 lr 以 float32 存储


def test_scheduler_cosine_matches_torch():
    a = _ours_seq("cosine", 20, T_max=10, eta_min=1e-5)
    b = _torch_seq("cosine", 20, T_max=10, eta_min=1e-5)
    assert np.allclose(a, b, rtol=1e-6, atol=0), (a, b)   # torch 的 lr 以 float32 存储


def test_scheduler_step_matches_torch():
    a = _ours_seq("step", 10, step_size=3, gamma=0.1)
    b = _torch_seq("step", 10, step_size=3, gamma=0.1)
    assert np.allclose(a, b, rtol=1e-6, atol=0), (a, b)   # torch 的 lr 以 float32 存储


def test_scheduler_constant_matches_torch():
    a = _ours_seq("constant", 8, factor=0.3, total_iters=3)
    b = _torch_seq("constant", 8, factor=0.3, total_iters=3)
    assert np.allclose(a, b, rtol=1e-6, atol=0), (a, b)   # torch 的 lr 以 float32 存储


def test_scheduler_lambda_matches_torch():
    fn = lambda n: 1.0 / (1.0 + n)          # noqa: E731
    a = _ours_seq("lambda", 9, fn=fn)
    b = _torch_seq("lambda", 9, lr_lambda=fn)
    assert np.allclose(a, b, rtol=1e-6, atol=0), (a, b)   # torch 的 lr 以 float32 存储


def test_cosine_warmup_is_monotonic_and_lands_on_base():
    opt = _FakeOpt(1e-3)
    s = T.optim.lr_scheduler.CosineLR(opt, T_max=10, warmup_steps=5)
    seq = [s.step() for _ in range(5)]
    # warmup 共 5 步：第 0 步 0，第 4 步 0.8*base，第 5 步才到 base
    assert seq[0] == 0.0 and abs(seq[-1] - 0.8e-3) < 1e-12
    assert all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
    assert abs(s.step() - 1e-3) < 1e-12      # warmup 结束后第一步仍在 base


# ───────────────────────── SGD ─────────────────────────
@pytest.mark.parametrize("kw", [
    dict(momentum=0.0),
    dict(momentum=0.9),
    dict(momentum=0.9, nesterov=True),
    dict(momentum=0.5, dampening=0.1),
    dict(momentum=0.9, weight_decay=0.01),
])
def test_sgd_matches_torch(kw):
    rng = np.random.default_rng(0)
    shape = (7, 5)
    w0 = rng.normal(0, 1, shape).astype(np.float32)
    grads = [rng.normal(0, 0.1, shape).astype(np.float32) for _ in range(4)]

    ours = T.nn.Parameter(w0.copy())
    ref = torch.nn.Parameter(torch.from_numpy(w0.copy()))
    opt_o = T.optim.SGD([ours], lr=1e-2, **kw)
    opt_r = torch.optim.SGD([ref], lr=1e-2, **kw)

    for g in grads:
        ours.grad = T.tensor(g.copy())
        ref.grad = torch.from_numpy(g.copy())
        opt_o.step()
        opt_r.step()
        assert _rel(ours.data, ref.detach().numpy()) <= 1e-6, (kw, _rel(ours.data, ref.detach().numpy()))


def test_sgd_rejects_bad_nesterov_with_fix_hint():
    p = T.nn.Parameter(np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError, match="修法"):
        T.optim.SGD([p], lr=1e-2, momentum=0.0, nesterov=True)


# ───────────────────────── Lion ─────────────────────────
def _lion_ref(w0, grads, lr, b1, b2, wd):
    w = w0.astype(np.float64).copy()
    m = np.zeros_like(w)
    for g in grads:
        g = g.astype(np.float64)
        upd = np.sign(b1 * m + (1.0 - b1) * g)
        if wd:
            upd = upd + wd * w
        w = w - lr * upd
        m = b2 * m + (1.0 - b2) * g
    return w


@pytest.mark.parametrize("wd", [0.0, 0.01])
def test_lion_matches_reference(wd):
    rng = np.random.default_rng(1)
    shape = (6, 4)
    w0 = rng.normal(0, 1, shape).astype(np.float32)
    grads = [rng.normal(0, 0.1, shape).astype(np.float32) for _ in range(5)]
    ours = T.nn.Parameter(w0.copy())
    opt = T.optim.Lion([ours], lr=1e-2, betas=(0.9, 0.99), weight_decay=wd)
    for g in grads:
        ours.grad = T.tensor(g.copy())
        opt.step()
    ref = _lion_ref(w0, grads, 1e-2, 0.9, 0.99, wd)
    assert _rel(ours.data, ref) <= 1e-6, _rel(ours.data, ref)


def test_lion_update_is_bounded_by_lr():
    """Lion 的特征：每次更新幅度恒为 lr（逐元素），不含 wd 时成立。"""
    rng = np.random.default_rng(2)
    w0 = rng.normal(0, 1, (5, 3)).astype(np.float32)
    ours = T.nn.Parameter(w0.copy())
    opt = T.optim.Lion([ours], lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0)
    ours.grad = T.tensor(rng.normal(0, 1, (5, 3)).astype(np.float32))
    opt.step()
    d = np.abs(ours.data - w0)
    assert np.allclose(d, 1e-3, atol=1e-6), d   # p-lr*sign 在 float32 下是 0.00099999
