# -*- coding: utf-8 -*-
"""CSA 反向冒烟：验证 csa_forward 能建图并反传（x 与权重都有梯度）。"""
import sys

sys.path.insert(0, r"D:\orv_pip_0962\Lib\site-packages")

import numpy as np

import tgai_go.train as T
import v3_csa as C

T.set_default_device("cuda")

cfg = C.CSAConfig(d_model=64, n_heads=4, csa_compress=4)
w = C.init_weights(np.random.default_rng(0), cfg)
plan = C.make_plan(cfg, 1, 32)
cos, sin = C.rope_tables(cfg, 32)

x_n = np.random.default_rng(1).normal(0, 1, (1, 32, 64)).astype(np.float32)
x = T.tensor(x_n, requires_grad=True)
y = C.csa_forward(x, w, plan, cos, sin)
loss = y.sum()
loss.backward()

gx = x.grad
gmax = 0.0 if gx is None else float(np.abs(gx.to_numpy()).max())
n_grad_w = 0
for k, v in (w.items() if isinstance(w, dict) else []):
    if getattr(v, "grad", None) is not None:
        n_grad_w += 1
print("CSA_BWD x_grad_max=%.3e  weights_with_grad=%d/%d  y_shape=%s"
      % (gmax, n_grad_w, len(w) if isinstance(w, dict) else -1, tuple(y.shape)))
