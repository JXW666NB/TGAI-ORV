# -*- coding: utf-8 -*-
"""DPO 与手写 torch 参考的对照（目标阶段二第 6 项）。

判据：loss 与 **对 logits 的梯度** 相对误差 ≤1e-5。
覆盖：logsigmoid 逐元素、单对 loss、多对 batch 平均、mask（ignore_index）、beta 取值。
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


def test_logsigmoid_matches_torch():
    x = np.array([-40.0, -5.0, -0.3, 0.0, 0.7, 12.0, 40.0], np.float32)
    a = np.asarray(T.logsigmoid(T.tensor(x)).data, np.float32)
    b = F.logsigmoid(torch.from_numpy(x)).numpy()
    assert _rel(a, b) <= 1e-6, (a, b)


def _seq_logit_pair(rng, Tlen=6, V=11, mask_prompt=2, seed=0):
    logits = rng.normal(0, 1, (Tlen, V)).astype(np.float32)
    labels = rng.integers(0, V, Tlen).astype(np.int64)
    labels[:mask_prompt] = -100          # 前 2 个位置是 prompt/padding
    return logits, labels


def test_seq_logprob_matches_torch_reference():
    rng = np.random.default_rng(0)
    logits, labels = _seq_logit_pair(rng)
    ours = float(np.asarray(T.seq_logprob(T.tensor(logits), labels).data, np.float32))

    lp = F.log_softmax(torch.from_numpy(logits), dim=-1)
    m = torch.from_numpy((labels != -100).astype(np.float32))
    idx = torch.from_numpy(np.where(labels == -100, 0, labels)).long()
    picked = lp.gather(1, idx[:, None]).squeeze(1)
    ref = float((picked * m).sum())
    assert abs(ours - ref) / max(1e-9, abs(ref)) <= 1e-5, (ours, ref)


def test_dpo_pair_loss_and_grad_match_torch():
    rng = np.random.default_rng(1)
    lw, yw = _seq_logit_pair(rng, seed=1)
    ll, yl = _seq_logit_pair(rng, seed=2)
    ref_w, ref_l = -7.5, -9.25             # 参考模型的 logp（标量，冻结）
    beta = 0.2

    lw_t = T.tensor(lw.copy(), requires_grad=True)
    ll_t = T.tensor(ll.copy(), requires_grad=True)
    pw = T.seq_logprob(lw_t, yw)
    pl = T.seq_logprob(ll_t, yl)
    loss, marg = T.dpo_pair_loss(pw, pl, ref_w, ref_l, beta)
    loss.backward()

    # torch 参考（同样的公式，独立实现）
    def torch_logp(logits, labels):
        lp = F.log_softmax(logits, dim=-1)
        idx = torch.from_numpy(np.where(labels == -100, 0, labels)).long()
        m = torch.from_numpy((labels != -100).astype(np.float32))
        return (lp.gather(1, idx[:, None]).squeeze(1) * m).sum()

    tw = torch.from_numpy(lw.copy()).requires_grad_(True)
    tl = torch.from_numpy(ll.copy()).requires_grad_(True)
    tloss = -F.logsigmoid(beta * ((torch_logp(tw, yw) - ref_w) - (torch_logp(tl, yl) - ref_l)))
    tloss.backward()

    assert abs(float(np.asarray(loss.data)) - float(tloss)) / max(1e-9, abs(float(tloss))) <= 1e-5
    assert _rel(lw_t.grad.data, tw.grad.numpy()) <= 1e-5
    assert _rel(ll_t.grad.data, tl.grad.numpy()) <= 1e-5
    assert abs(float(np.asarray(marg.data)) - float(beta * ((torch_logp(torch.from_numpy(lw), yw)
                                                            - ref_w)
                                                           - (torch_logp(torch.from_numpy(ll), yl)
                                                              - ref_l)))) <= 1e-4


@pytest.mark.parametrize("beta", [0.01, 0.1, 0.5])
def test_dpo_batch_loss_matches_torch(beta):
    rng = np.random.default_rng(2)
    n = 3
    lws, yws, lls, yls = [], [], [], []
    for i in range(n):
        a, b = _seq_logit_pair(rng, seed=10 + i)
        c, d = _seq_logit_pair(rng, seed=20 + i)
        lws.append(T.tensor(a.copy(), requires_grad=True))
        yws.append(b)
        lls.append(T.tensor(c.copy(), requires_grad=True))
        yls.append(d)
    refw = [-8.0, -6.5, -7.2]
    refl = [-9.1, -8.8, -9.5]

    loss, margin_list = T.dpo_batch_loss(lws, yws, lls, yls, refw, refl, beta)
    loss.backward()

    def torch_logp(logits, labels):
        lp = F.log_softmax(logits, dim=-1)
        idx = torch.from_numpy(np.where(labels == -100, 0, labels)).long()
        m = torch.from_numpy((labels != -100).astype(np.float32))
        return (lp.gather(1, idx[:, None]).squeeze(1) * m).sum()

    tot = 0.0
    for i in range(n):
        pw = torch_logp(torch.from_numpy(np.asarray(lws[i].data, np.float32)), yws[i])
        pl = torch_logp(torch.from_numpy(np.asarray(lls[i].data, np.float32)), yls[i])
        tot = tot - float(F.logsigmoid(beta * ((pw - refw[i]) - (pl - refl[i]))))
    ref_loss = tot / n
    assert abs(float(np.asarray(loss.data)) - ref_loss) / max(1e-9, abs(ref_loss)) <= 1e-5
    assert len(margin_list) == n


def test_dpo_rejects_nonpositive_beta():
    with pytest.raises(ValueError, match="修法"):
        T.dpo_pair_loss(0.0, 0.0, 0.0, 0.0, beta=0.0)


def test_seq_logprob_rejects_all_ignored():
    logits = T.tensor(np.zeros((3, 5), np.float32))
    with pytest.raises(ValueError, match="修法"):
        T.seq_logprob(logits, np.array([-100, -100, -100]))
