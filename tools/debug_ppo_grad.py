# -*- coding: utf-8 -*-
"""直接验证 ppo_loss 对 logπ 的梯度符号：优势为正时，梯度必须为负（下降会提高 logπ）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T
from tgai_go.train.ppo import ppo_loss

T.set_default_device("cpu")

for adv_val in (1.0, -1.0):
    lp = T.tensor(np.array([0.0], np.float32), requires_grad=True)
    loss, _ = ppo_loss(lp, np.array([0.0], np.float32), np.array([adv_val], np.float32),
                       clip_eps=0.2, normalize_adv=False)
    loss.backward()
    g = float(np.asarray(lp.grad, np.float32)[0])
    print("adv=%+.1f  ratio=1  =>  loss=%.4f  dloss/dlogp=%+.4f  %s"
          % (adv_val, float(np.asarray(loss.to_numpy())), g,
             "正确（应为负）" if (adv_val > 0) == (g < 0) else "**符号错误**"))

# 再看 ratio != 1 的情形
for r in (0.5, 2.0):
    lp = T.tensor(np.array([np.log(r)], np.float32), requires_grad=True)
    loss, _ = ppo_loss(lp, np.array([0.0], np.float32), np.array([1.0], np.float32),
                       clip_eps=0.2, normalize_adv=False)
    loss.backward()
    g = float(np.asarray(lp.grad, np.float32)[0])
    print("adv=+1  ratio=%.1f  =>  dloss/dlogp=%+.4f  （ratio<0.8 或 >1.2 时应为 0，被 clip）" % (r, g))
