# -*- coding: utf-8 -*-
"""GRPO（含 GPRO 口径）与手写 torch 参考的对照（目标阶段二第 7 项）。

判据：loss 与对 logits 的梯度相对误差 ≤1e-5。
覆盖：组内归一化优势、token 级 PPO 裁剪、KL 的 k1/k3 两种估计、mask、clip 触发统计。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402

torch = pytest.importorskip("torch")
F = torch.nn.functional


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def test_group_advantages_matches_formula():
    r = np.array([1.0, 3.0, 2.0, 5.0], np.float32)
    a = T.group_advantages(r)
    ref = (r - r.mean()) / (r.std() + 1e-4)
    assert _rel(a, ref) <= 1e-6, (a, ref)
    # torch 同口径（总体标准差）
    rt = torch.from_numpy(r.astype(np.float64))
    ref_t = ((rt - rt.mean()) / (rt.std(unbiased=False) + 1e-4)).numpy()
    assert _rel(a, ref_t) <= 1e-6


def test_group_advantages_rejects_singleton():
    with pytest.raises(ValueError, match="修法"):
        T.group_advantages([1.0])


def test_token_logprobs_matches_torch():
    rng = np.random.default_rng(0)
    Tlen, V = 5, 7
    logits = rng.normal(0, 1, (Tlen, V)).astype(np.float32)
    labels = rng.integers(0, V, Tlen).astype(np.int64)
    labels[:2] = -100
    ours = np.asarray(T.token_logprobs(T.tensor(logits), labels).data, np.float32)
    lp = F.log_softmax(torch.from_numpy(logits), dim=-1)
    idx = torch.from_numpy(np.where(labels == -100, 0, labels)).long()
    picked = lp.gather(1, idx[:, None]).squeeze(1).numpy()
    ref = picked * (labels != -100).astype(np.float32)
    assert _rel(ours, ref) <= 1e-5, (ours, ref)


def _grpo_case(beta, kl, seed=0, G=3, Tlen=4, V=6, eps=0.2):
    rng = np.random.default_rng(seed)
    logits = [rng.normal(0, 1, (Tlen, V)).astype(np.float32) for _ in range(G)]
    labels = [rng.integers(0, V, Tlen).astype(np.int64) for _ in range(G)]
    masks = [np.array([0, 1, 1, 1], np.float32) for _ in range(G)]
    old = [rng.normal(-1.5, 0.2, Tlen).astype(np.float32) for _ in range(G)]
    ref = [rng.normal(-1.7, 0.2, Tlen).astype(np.float32) for _ in range(G)]
    adv = T.group_advantages(np.array([1.0, 0.0, -1.0], np.float32))

    ours_logits = [T.tensor(a.copy(), requires_grad=True) for a in logits]
    pls = [T.token_logprobs(ours_logits[i], labels[i]) for i in range(G)]
    loss, stats = T.grpo_loss(pls, old, ref, adv, masks, clip_eps=eps, beta=beta, kl=kl)
    loss.backward()

    # torch 参考（独立实现，同样的公式）
    tl = [torch.from_numpy(a.copy()).requires_grad_(True) for a in logits]
    total = 0.0
    ntok = 0.0
    for i in range(G):
        p = torch.log_softmax(tl[i], dim=-1).gather(
            1, torch.from_numpy(np.where(labels[i] == -100, 0, labels[i])).long()[:, None]).squeeze(1)
        m = torch.from_numpy(masks[i])
        oldt = torch.from_numpy(old[i])
        reft = torch.from_numpy(ref[i])
        ratio = (p - oldt).exp()
        s1 = ratio * float(adv[i])
        s2 = torch.clamp(ratio, 1 - eps, 1 + eps) * float(adv[i])
        per = -(torch.minimum(s1, s2) * m).sum()
        if beta:
            d = reft - p
            klt = (d.exp() - d - 1.0) if kl == "k3" else d
            per = per + beta * (klt * m).sum()
        total = total + per
        ntok += float(m.sum())
    ref_loss = total / ntok
    ref_loss.backward()

    return loss, ref_loss, ours_logits, tl, stats


@pytest.mark.parametrize("beta,kl", [(0.0, "k3"), (0.1, "k3"), (0.1, "k1")])
def test_grpo_loss_and_grad_match_torch(beta, kl):
    loss, ref_loss, ours, ref, stats = _grpo_case(beta, kl)
    assert abs(float(np.asarray(loss.data)) - float(ref_loss)) / max(1e-9, abs(float(ref_loss))) <= 1e-5
    for i in range(len(ours)):
        assert _rel(ours[i].grad.data, ref[i].grad.numpy()) <= 1e-5, i
    assert stats["tokens"] == 9.0 and 0.0 <= stats["clip_frac"] <= 1.0


def test_grpo_clip_actually_triggers():
    """ratio 远离 1 时应当被裁剪：clip_frac > 0，且 loss 与不裁剪不同。"""
    rng = np.random.default_rng(3)
    Tlen, V = 3, 5
    logits = rng.normal(0, 1, (Tlen, V)).astype(np.float32)
    labels = rng.integers(0, V, Tlen).astype(np.int64)
    mask = [np.ones(Tlen, np.float32)]
    old = [rng.normal(-6.0, 0.1, Tlen).astype(np.float32)]   # 与当前策略差很多 ⇒ ratio 远离 1
    ref = [old[0].copy()]
    adv = np.array([1.0], np.float32)

    lg = T.tensor(logits.copy(), requires_grad=True)
    loss_clip, st = T.grpo_loss([T.token_logprobs(lg, labels)], old, ref, adv, mask,
                                clip_eps=0.2, beta=0.0)
    assert st["clip_frac"] > 0.0, st


def test_grpo_rejects_bad_inputs():
    with pytest.raises(ValueError, match="修法"):
        T.grpo_loss([T.tensor(np.zeros((2, 3), np.float32))], [], [], [], [], clip_eps=0.2)
    with pytest.raises(ValueError, match="修法"):
        T.grpo_loss([T.tensor(np.zeros((2, 3), np.float32))], [np.zeros(2, np.float32)],
                    [np.zeros(2, np.float32)], np.array([0.0], np.float32),
                    [np.zeros(2, np.float32)], clip_eps=-0.1)
