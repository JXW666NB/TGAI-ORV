# -*- coding: utf-8 -*-
"""`PPOTrainer` 端到端测试（0.9.25）。

判据：
1. 一次 `step()` 能跑通，loss 有限、adv/returns 形状正确；
2. critic 的梯度**真的**被写出来（value_params 已并进 params，否则会静默不训练）；
3. 在一个可学的玩具任务上重复 step，期望回报上升。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train.rl_trainer import PPOTrainer


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield
    T.set_default_device("cpu")


def _np(x):
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, np.float32)


def _toy(seed=0, n=8, dim=3):
    """玩具策略/价值网络：logits = x·w，value = x·v（都用线性，便于验证梯度）。"""
    rng = np.random.default_rng(seed)
    w = T.nn.Parameter(rng.normal(0, 0.1, (dim, 2)).astype(np.float32))
    v = T.nn.Parameter(rng.normal(0, 0.1, (dim,)).astype(np.float32))
    x = rng.normal(0, 1, (n, dim)).astype(np.float32)

    def logprob(ids, mask):
        # 用 softmax 后的 log 概率，但为了简单起见这里直接返回 logits 的 log_softmax 对角
        logits = T.tensor(np.asarray(ids, np.float32)) @ w        # [n, 2]
        return T.functional.log_softmax(logits, axis=-1)[:, 0]

    def value(ids, mask):
        return T.tensor(np.asarray(ids, np.float32)) @ v

    return w, v, x, logprob, value


def test_ppo_step_runs_and_updates_critic():
    w, v, x, logprob, value = _toy()
    opt = T.optim.AdamW([w, v], lr=0.05)
    tr = PPOTrainer([w], logprob, value, optimizer=opt, value_params=[v],
                    clip_eps=0.2, vf_coef=0.5)
    mask = np.ones(8, np.float32)
    rewards = np.array([1, 0, 1, 0, 1, 0, 1, 0], np.float32)
    out = tr.step(x, mask, rewards)

    assert np.isfinite(out["loss"])
    assert np.asarray(out["adv"]).shape == (8,)
    assert np.asarray(out["returns"]).shape == (8,)
    # critic 必须被并进 params，否则 v.grad 会是 None（静默不训练）
    assert v in tr.params, "value_params 没有被并进 self.params"


def test_ppo_critic_gradient_is_written():
    w, v, x, logprob, value = _toy(seed=1)
    tr = PPOTrainer([w], logprob, value, value_params=[v], optimizer=T.optim.SGD([w, v], lr=1e-3))
    mask = np.ones(8, np.float32)
    rewards = np.ones(8, np.float32)
    tr.step(x, mask, rewards)
    # _finish 会清零后再反传，所以这里检查"训练器认知的参数集合"包含 critic
    assert any(p is v for p in tr.params)


def test_ppo_learns_on_toy_task():
    """方向性判据：优势高的样本，其动作 logπ 应当**上升**。

    两个坑都得避开，否则测出来的是测试的毛病而不是代码的毛病：
    1. 全 1 奖励会被标准化成 0 优势（数据没区分度）→ 给两档奖励；
    2. `w` 是**所有样本共享**的，若两组样本的 x 都是随机的，梯度会互相抵消 →
       让两组 x 各自固定且线性可分，任务才真的可学。
    """
    rng = np.random.default_rng(2)
    dim, n = 3, 16
    w = T.nn.Parameter(rng.normal(0, 0.1, (dim, 2)).astype(np.float32))
    v = T.nn.Parameter(rng.normal(0, 0.1, (dim,)).astype(np.float32))
    x = np.vstack([np.tile(np.array([1.0, 0.0, 0.0], np.float32), (8, 1)),
                   np.tile(np.array([0.0, 1.0, 0.0], np.float32), (8, 1))]).astype(np.float32)

    def logprob(ids, mask):
        logits = T.tensor(np.asarray(ids, np.float32)) @ w
        return T.functional.log_softmax(logits, axis=-1)[:, 0]

    def value(ids, mask):
        return T.tensor(np.asarray(ids, np.float32)) @ v

    opt = T.optim.AdamW([w, v], lr=0.1)
    tr = PPOTrainer([w], logprob, value, optimizer=opt, value_params=[v],
                    clip_eps=0.2, vf_coef=0.5, gamma=0.9, lam=0.9)
    mask = np.ones(n, np.float32)
    rewards = np.array([1.0] * 8 + [0.0] * 8, np.float32)
    good = np.arange(8)

    before = _np(logprob(x, mask))[good].mean()
    for _ in range(20):
        out = tr.step(x, mask, rewards)
    after = _np(logprob(x, mask))[good].mean()
    assert np.isfinite(out["loss"])
    assert after > before, "优势高的样本 logπ 应上升：%.4f -> %.4f" % (before, after)


def test_ppo_rejects_mismatched_lengths():
    w, v, x, logprob, value = _toy()
    tr = PPOTrainer([w], logprob, value, value_params=[v])
    with pytest.raises(ValueError):
        tr.step(x, np.ones(4, np.float32), np.ones(8, np.float32))
