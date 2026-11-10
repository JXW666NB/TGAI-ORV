# -*- coding: utf-8 -*-
"""在 CPU 上复现 GRPO smoke 的 nan，判断是否与 CUDA 有关。"""
import sys

import numpy as np

sys.path.insert(0, r"D:\DSH-Workspace\TGAI-对话档案\_tgai_orv\python")

import tgai_go.train as T
from tgai_go.train.rl_trainer import GRPOTrainer

rng = np.random.default_rng(0)
dim = 6
w = T.nn.Parameter(rng.normal(0, 0.2, (dim,)).astype(np.float32))
G, TT = 4, 5
ids = [rng.normal(0, 1, (TT, dim)).astype(np.float32) for _ in range(G)]
masks = [np.ones(TT, np.float32) for _ in range(G)]
rewards = np.array([1.0, 0.5, -0.2, -0.9], np.float32)


def tok(ids_i, mask_i):
    return (T.tensor(np.asarray(ids_i, np.float32)) * w).sum(axis=1)


gt = GRPOTrainer([w], tok, clip_eps=0.2, beta=0.0,
                 optimizer=T.optim.AdamW([w], lr=0.05), max_grad_norm=1.0)
out = gt.step(ids, masks, rewards)
print("loss =", out["loss"])
print("adv  =", out["adv"])
print("grad =", None if w.grad is None else np.asarray(w.grad if not hasattr(w.grad, "to_numpy") else w.grad.to_numpy()))

# 逐步拆解：policy/old 是否相等、ratio 是否 1
pol = [tok(ids[i], masks[i]) for i in range(G)]
for i in range(G):
    p = np.asarray(pol[i].to_numpy() if hasattr(pol[i], "to_numpy") else pol[i], np.float32)
    old = np.asarray(gt._old[i], np.float32)
    print("i=%d  max|p-old| = %.3e   p[0]=%.4f old[0]=%.4f" % (i, float(np.abs(p - old).max()), p[0], old[0]))
