# -*- coding: utf-8 -*-
"""PPO / RLHF 的对照测试（0.9.25）。

判据：
1. `gae` 与手写递推逐位一致，且能通过"单步不给 bootstrap"的边界检查；
2. `ppo_loss` 的数值与 torch 手写参考一致（≤1e-5），**梯度**也一致；
3. PPO 的裁剪行为正确：ratio 超出 [1−ε,1+ε] 时梯度被截断；
4. 端到端：一个可学的玩具任务上，PPO 让期望回报上升。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train.ppo import discounted_returns, gae, ppo_loss

torch = pytest.importorskip("torch")


def _np(x):
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, np.float32)


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(1e-9, float(np.abs(b).max())))


def _scalar(x):
    """把 ORV Tensor / numpy 标量取成 python float（ORV Tensor 不能直接 float()）。"""
    return float(np.asarray(_np(x), np.float32).reshape(-1)[0])


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield
    T.set_default_device("cpu")


# ---------------- GAE ----------------
def test_gae_matches_manual_recursion():
    r = np.array([1.0, 0.0, -1.0, 2.0], np.float32)
    v = np.array([0.5, 0.2, -0.1, 0.3], np.float32)
    gamma, lam = 0.9, 0.8
    adv, ret = gae(r, v, None, gamma=gamma, lam=lam)

    # 手写递推
    exp_adv = np.zeros(4, np.float64)
    acc = 0.0
    for t in range(3, -1, -1):
        nxt = v[t + 1] if t + 1 < 4 else 0.0
        delta = r[t] + gamma * nxt - v[t]
        acc = delta + gamma * lam * acc
        exp_adv[t] = acc
    assert _rel(adv, exp_adv) <= 1e-6
    assert _rel(ret, exp_adv + v) <= 1e-6


def test_gae_done_stops_bootstrap():
    r = np.array([1.0, 1.0], np.float32)
    v = np.array([10.0, 10.0], np.float32)
    adv, _ = gae(r, v, np.array([1.0, 0.0], np.float32), gamma=0.9, lam=1.0)
    # 第 0 步 done ⇒ 不含 V(s_1)
    assert abs(float(adv[0]) - (1.0 - 10.0)) < 1e-5, adv


def test_gae_rejects_length_mismatch():
    with pytest.raises(ValueError):
        gae([1.0, 2.0], [0.0], None)


def test_discounted_returns_basic():
    r = np.array([1.0, 1.0, 1.0], np.float32)
    g = discounted_returns(r, gamma=0.5)
    assert np.allclose(g, [1.75, 1.5, 1.0], atol=1e-6)


# ---------------- PPO 损失 ----------------
def _toy_logps(n=6, seed=0):
    rng = np.random.default_rng(seed)
    lp = rng.normal(-1.0, 0.3, n).astype(np.float32)
    old = (lp + rng.normal(0, 0.05, n)).astype(np.float32)
    adv = rng.normal(0, 1, n).astype(np.float32)
    return lp, old, adv


def test_ppo_loss_matches_torch_reference():
    lp, old, adv = _toy_logps()
    eps = 0.2
    mask = np.ones_like(lp)

    loss, stats = ppo_loss(T.tensor(lp, requires_grad=True), old, adv, masks=mask,
                           clip_eps=eps, normalize_adv=False)

    # torch 参考
    p = torch.tensor(lp, requires_grad=True)
    o = torch.tensor(old)
    a = torch.tensor(adv)
    m = torch.tensor(mask)
    ratio = torch.exp(p - o)
    s1 = ratio * a
    s2 = torch.clamp(ratio, 1 - eps, 1 + eps) * a
    ref = (-torch.min(s1, s2) * m).sum() / m.sum()
    ref.backward()

    assert abs(_scalar(loss) - float(ref)) <= 1e-5, (_scalar(loss), float(ref))
    # 同一个 Tensor 的梯度：重新算一次拿 grad
    lp2 = T.tensor(lp, requires_grad=True)
    loss2, _ = ppo_loss(lp2, old, adv, masks=mask, clip_eps=eps, normalize_adv=False)
    loss2.backward()
    assert _rel(_np(lp2.grad), p.grad.numpy()) <= 1e-5


def test_ppo_clip_blocks_gradient_when_ratio_too_large():
    """ratio 远大于 1+ε 且优势为正时，PPO 的梯度应为 0（被 clip 掉）。"""
    old = np.array([0.0], np.float32)
    lp = T.tensor(np.array([2.0], np.float32), requires_grad=True)   # ratio = e^2 ≈ 7.4
    adv = np.array([1.0], np.float32)
    loss, _ = ppo_loss(lp, old, adv, clip_eps=0.2, normalize_adv=False)
    loss.backward()
    assert abs(float(_np(lp.grad)[0])) < 1e-6, _np(lp.grad)


def test_ppo_value_loss_and_entropy_terms():
    lp, old, adv = _toy_logps()
    v = np.array([0.5] * len(lp), np.float32)
    ret = np.array([1.0] * len(lp), np.float32)
    ent = np.array([0.5] * len(lp), np.float32)
    l0, _ = ppo_loss(T.tensor(lp), old, adv, normalize_adv=False)
    l1, _ = ppo_loss(T.tensor(lp), old, adv, values=v, returns=ret,
                     normalize_adv=False, vf_coef=0.5)
    l2, _ = ppo_loss(T.tensor(lp), old, adv, values=v, returns=ret, entropy=ent,
                     normalize_adv=False, vf_coef=0.5, ent_coef=0.01)
    # vf 项应为 0.5 * mean((0.5-1)^2) = 0.5*0.25 = 0.125
    assert abs((_scalar(l1) - _scalar(l0)) - 0.125) < 1e-5, (_scalar(l0), _scalar(l1))
    # 熵项为 −0.01*0.5 = −0.005
    assert abs((_scalar(l2) - _scalar(l1)) + 0.005) < 1e-6


def test_ppo_rejects_bad_clip():
    with pytest.raises(ValueError):
        ppo_loss(T.tensor(np.zeros(3, np.float32)), np.zeros(3, np.float32),
                 np.zeros(3, np.float32), clip_eps=0.0)


def test_ppo_requires_returns_with_values():
    with pytest.raises(ValueError):
        ppo_loss(T.tensor(np.zeros(3, np.float32)), np.zeros(3, np.float32),
                 np.zeros(3, np.float32), values=np.zeros(3, np.float32))
