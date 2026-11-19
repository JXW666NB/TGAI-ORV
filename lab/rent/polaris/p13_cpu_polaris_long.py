# -*- coding: utf-8 -*-
"""Polaris CPU 长跑：lr=3e-3 / 1e-2 各 250 步，看"先收敛后极限环"在 CPU 上是否同样出现。

与 g10b（CUDA 200 步）同任务、同初始化、同批数据。
"""
import json
import os
import sys

import numpy as np

import pbench_lib as L

L.utf8()
import tgai_go.train as T

assert T.get_device() == "cpu"
import polaris  # noqa: F401,E402

P = T.plugin.resolve_optimizer("Polaris")
SEQ, NB = 64, 8
xs, ys = L.model_blocks(SEQ, NB)
X = np.stack(xs)
Y = np.stack(ys)
STEPS = 250

res = {}
for lr in (3e-3, 1e-2):
    T.manual_seed(1000)
    model = T.preset.transformer(vocab=4096, d_model=128, n_layers=2, n_heads=4, d_ff=512)
    ps = list(model.parameters())
    opt = P(ps, lr=lr)
    curve = []
    for i in range(STEPS):
        logits = model(X)
        loss = L.ce_vec(T, logits.reshape(NB * SEQ, 4096), Y.reshape(-1))
        for p in ps:
            p.grad = None
        loss.backward()
        opt.step()
        curve.append(float(np.asarray(loss.to_numpy())))
    res["lr_%.0e" % lr] = {
        "last10_mean": float(np.mean(curve[-10:])),
        "min": float(min(curve)), "argmin_step": int(np.argmin(curve)) + 1,
        "loss_at_50": curve[49], "loss_at_100": curve[99],
        "loss_at_150": curve[149], "loss_at_200": curve[199],
        "curve": [round(v, 6) for v in curve],
        "status": opt.status(),
    }
    print("CPU Polaris lr=%.0e 末10=%.4g min=%.4g@%d 50:%.4g 100:%.4g 150:%.4g 200:%.4g"
          % (lr, res["lr_%.0e" % lr]["last10_mean"], res["lr_%.0e" % lr]["min"],
             res["lr_%.0e" % lr]["argmin_step"], curve[49], curve[99],
             curve[149], curve[199]), flush=True)
    print("  tail100-250: " + " ".join("%.4g" % v for v in curve[99:250:10]), flush=True)
L.dump("p13_cpu_polaris_long", res)
