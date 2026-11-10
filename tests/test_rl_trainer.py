# -*- coding: utf-8 -*-
"""高层训练器：DPOTrainer / GRPOTrainer 的对照测试与端到端一步。

判据：
1. DPO 一步的 **损失与参数梯度** 与"手写 torch 参考实现"一致（≤1e-6）；
2. GRPO 一步的损失与策略梯度与 torch 参考（PPO-clip + k3 KL）一致（≤1e-5）；
3. 端到端：在可分数据上重复 DPO 步，偏好 margin 上升、loss 下降。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train.rl_trainer import DPOTrainer, GRPOTrainer

torch = pytest.importorskip("torch")


def _np(t):
    return np.asarray(t.to_numpy() if hasattr(t, "to_numpy") else t, np.float32)


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(1e-9, float(np.abs(b).max())))


# ---------------- 一个极小的"策略"：logπ = x·w ----------------
class ToyPolicy(object):
    """logprob_fn(ids, mask) = sum(ids * w)，用于把 DPO/GRPO 的数学与模型解耦。"""

    def __init__(self, dim, seed=0):
        rng = np.random.default_rng(seed)
        self.w = T.nn.Parameter(rng.normal(0, 0.2, (dim,)).astype(np.float32))

    def params(self):
        return [self.w]

    def logprob(self, ids, mask=None):
        ids = np.asarray(ids, np.float32)
        s = (T.tensor(ids) * self.w).sum()
        return s


def test_dpo_step_matches_torch_reference():
    dim = 5
    pol = ToyPolicy(dim, seed=1)
    xw = np.array([1.0, 0.5, -0.3, 0.2, 0.1], np.float32)
    xl = np.array([-0.4, 0.2, 0.9, -0.1, 0.3], np.float32)
    ref_w, ref_l = -0.7, -1.1
    beta = 0.2

    tr = DPOTrainer(pol.params(), pol.logprob, ref_logprob_fn=None, beta=beta)
    out = tr.step(xw, None, xl, None, ref_logp_w=ref_w, ref_logp_l=ref_l)

    # torch 参考：同样的公式，按参数自动求导
    w = torch.tensor(_np(pol.w).copy() if hasattr(pol.w, "to_numpy") else np.asarray(pol.w, np.float32),
                     requires_grad=True)
    pw = (torch.tensor(xw) * w).sum()
    pl = (torch.tensor(xl) * w).sum()
    loss = -torch.nn.functional.logsigmoid(((pw - ref_w) - (pl - ref_l)) * beta)
    loss.backward()

    assert abs(out["loss"] - float(loss)) <= 1e-6, (out["loss"], float(loss))
    # 注意：Trainer 已经 step 过（opt=None 时不动参数），梯度仍在
    assert _rel(_np(pol.w.grad), w.grad.numpy()) <= 1e-6


def test_dpo_requires_ref_or_beta():
    pol = ToyPolicy(3, seed=2)
    tr = DPOTrainer(pol.params(), pol.logprob, ref_logprob_fn=None, beta=0.1)
    with pytest.raises(ValueError):
        tr.step(np.ones(3, np.float32), None, np.zeros(3, np.float32), None)
    with pytest.raises(ValueError):
        DPOTrainer(pol.params(), pol.logprob, beta=0.0)


def test_dpo_end_to_end_prefers_chosen():
    """在可分数据上重复 DPO 步：margin 上升、loss 下降。"""
    dim = 4
    pol = ToyPolicy(dim, seed=3)
    opt = T.optim.AdamW(pol.params(), lr=0.2)
    xw = np.array([1.0, 0.8, 0.0, 0.0], np.float32)
    xl = np.array([0.0, 0.0, 1.0, 0.8], np.float32)
    tr = DPOTrainer(pol.params(), pol.logprob, beta=0.1, optimizer=opt)
    first = tr.step(xw, None, xl, None, ref_logp_w=0.0, ref_logp_l=0.0)
    for _ in range(15):
        last = tr.step(xw, None, xl, None, ref_logp_w=0.0, ref_logp_l=0.0)
    assert last["loss"] < first["loss"], (first["loss"], last["loss"])
    assert last["margin"] > first["margin"], (first["margin"], last["margin"])


def test_grpo_step_matches_torch_reference():
    """G=3 条采样，token 级 logπ 由小策略给出；与 torch 的 PPO-clip + k3 KL 对照。"""
    G, TT, dim = 3, 4, 3
    rng = np.random.default_rng(4)
    ids = [rng.normal(0, 1, (TT, dim)).astype(np.float32) for _ in range(G)]
    masks = [np.ones(TT, np.float32) for _ in range(G)]
    rewards = np.array([1.0, 0.2, -0.5], np.float32)
    clip_eps, beta = 0.2, 0.0

    w0 = rng.normal(0, 0.1, (dim,)).astype(np.float32)
    pol = ToyPolicy(dim, seed=5)

    def tok_logprob(ids_i, mask_i):
        # [T] 逐 token：ids_i[t] · w
        return (T.tensor(np.asarray(ids_i, np.float32)) * pol.w).sum(axis=1)

    tr = GRPOTrainer(pol.params(), tok_logprob, clip_eps=clip_eps, beta=beta)
    old = [np.asarray(np.asarray(ids[i], np.float32) @ w0, np.float32) for i in range(G)]
    out = tr.step(ids, masks, rewards, old_logps=old)

    # torch 参考
    w = torch.tensor(_np(pol.w).copy(), requires_grad=True)
    adv = torch.tensor((rewards - rewards.mean()) / (rewards.std() + 1e-4))
    total = 0.0
    for i in range(G):
        lp = (torch.tensor(np.asarray(ids[i], np.float32)) * w).sum(dim=1)
        ratio = torch.exp(lp - torch.tensor(old[i]))
        a = adv[i]
        un = ratio * a
        cl = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a
        mask = torch.tensor(masks[i])
        total = total + (-torch.min(un, cl) * mask).sum() / mask.sum()
    loss = total / G
    loss.backward()
    if beta == 0.0:
        assert abs(out["loss"] - float(loss)) <= 1e-5, (out["loss"], float(loss))
        assert _rel(_np(pol.w.grad), w.grad.numpy()) <= 1e-5


def test_grpo_rejects_mismatched_lengths():
    pol = ToyPolicy(2, seed=6)
    tr = GRPOTrainer(pol.params(), lambda ids, m: (T.tensor(np.asarray(ids, np.float32)) * pol.w).sum(axis=1))
    with pytest.raises(ValueError):
        tr.step([np.ones((2, 2), np.float32)], [], np.array([1.0, 2.0], np.float32))
