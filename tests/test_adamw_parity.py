# -*- coding: utf-8 -*-
"""优化器对齐：`tgai_go.train.optim.AdamW` 与 `torch.optim.AdamW` 逐步一致。

为什么单独测这个：模型梯度对了，优化器错了照样训不出预期结果，而且**错得很隐蔽** ——
loss 曲线只是"差点"，不像崩。容易错的四处：
  1. decoupled weight decay 的施加顺序（先加 wd*lr*p 还是先做动量更新）；
  2. bias correction 用的 t 是"第几步"（从 1 开始）；
  3. eps 加在 sqrt 之后还是之前；
  4. weight_decay=0 时不能退化成 L2（会进梯度）。
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tgai_go.train as T


def _make_ours(shape, seed=0):
    rng = np.random.default_rng(seed)
    p = T.nn.Parameter(rng.normal(0, 1, shape).astype(np.float32))
    return p


@pytest.mark.parametrize("wd", [0.0, 0.01, 0.1])
def test_adamw_matches_torch_step_by_step(wd):
    T.set_default_device("cpu")
    shape = (4, 5)
    rng = np.random.default_rng(7)
    init = rng.normal(0, 1, shape).astype(np.float32)

    ours = _make_ours(shape)
    ours.data[...] = init
    ref = torch.tensor(init.copy(), requires_grad=True)

    lr, betas, eps = 3e-3, (0.9, 0.95), 1e-8
    opt_o = T.optim.AdamW([ours], lr=lr, betas=betas, eps=eps, weight_decay=wd)
    opt_r = torch.optim.AdamW([ref], lr=lr, betas=betas, eps=eps, weight_decay=wd)

    for step in range(6):
        g = rng.normal(0, 0.5, shape).astype(np.float32)
        ours.grad = g.copy()
        ref.grad = torch.tensor(g.copy())
        opt_o.step()
        opt_r.step()
        opt_o.zero_grad()
        opt_r.zero_grad()

        got = np.asarray(ours.to_numpy(), np.float64)
        want = ref.detach().numpy().astype(np.float64)
        scale = max(float(np.abs(want).max()), 1e-9)
        rel = float(np.abs(got - want).max()) / scale
        assert rel < 1e-5, (
            "AdamW 第 %d 步不一致（wd=%g，相对误差 %.2e）。"
            "常见原因：weight_decay 的施加顺序、bias correction 的 t 起点、eps 的位置"
            % (step + 1, wd, rel))


def test_adamw_weight_decay_zero_is_not_l2():
    """wd=0 时参数变化必须完全来自梯度；若把 wd 误加进梯度，wd=0 就不会有问题、
    但 wd>0 会与 torch 分叉（上面那条已经覆盖）。这里额外确认 wd=0 与梯度同向。"""
    T.set_default_device("cpu")
    p = _make_ours((3,), seed=1)
    p.data[...] = np.array([1.0, -1.0, 0.5], np.float32)
    opt = T.optim.AdamW([p], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    p.grad = np.array([1.0, 1.0, 1.0], np.float32)
    opt.step()
    got = np.asarray(p.to_numpy(), np.float64)
    want = np.array([1.0, -1.0, 0.5]) - 1e-2      # 第一步 Adam 的更新量≈lr（bias correction 后）
    assert np.allclose(got, want, atol=2e-3), (got, want)
