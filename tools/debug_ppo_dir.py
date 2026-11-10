# -*- coding: utf-8 -*-
"""调试 PPO 方向：打印优势符号、logπ 变化与单步梯度方向。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T
from tgai_go.train.ppo import gae
from tgai_go.train.rl_trainer import PPOTrainer

T.set_default_device("cpu")
rng = np.random.default_rng(2)
n, dim = 16, 3
w = T.nn.Parameter(rng.normal(0, 0.1, (dim, 2)).astype(np.float32))
v = T.nn.Parameter(rng.normal(0, 0.1, (dim,)).astype(np.float32))
x = rng.normal(0, 1, (n, dim)).astype(np.float32)
mask = np.ones(n, np.float32)
rewards = np.array([1.0] * 8 + [0.0] * 8, np.float32)


def logprob(ids, mask):
    logits = T.tensor(np.asarray(ids, np.float32)) @ w
    return T.functional.log_softmax(logits, axis=-1)[:, 0]


def value(ids, mask):
    return T.tensor(np.asarray(ids, np.float32)) @ v


vals = np.asarray(value(x, mask).to_numpy(), np.float32)
adv, ret = gae(rewards, vals, None, gamma=0.9, lam=0.9)
print("values[:4] =", np.round(vals[:4], 3))
print("adv 前8 =", np.round(adv[:8], 3))
print("adv 后8 =", np.round(adv[8:], 3))
adn = (adv - adv.mean()) / (adv.std() + 1e-8)
print("标准化后 前8 =", np.round(adn[:8], 3))
print("标准化后 后8 =", np.round(adn[8:], 3))

# 单步：看无优化器时 logp 的梯度符号
lp = logprob(x, mask)
before = np.asarray(lp.to_numpy(), np.float32).copy()
tr = PPOTrainer([w], logprob, value, value_params=[v],
                optimizer=T.optim.SGD([w, v], lr=0.1), clip_eps=0.2, vf_coef=0.5)
out = tr.step(x, mask, rewards)
after = np.asarray(logprob(x, mask).to_numpy(), np.float32)
print("单步后 前8 logπ 变化:", np.round(after[:8] - before[:8], 4))
print("单步后 后8 logπ 变化:", np.round(after[8:] - before[8:], 4))
print("trainer 报的 adv 前8 =", np.round(np.asarray(out["adv"])[:8], 3))
