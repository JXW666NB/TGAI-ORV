# -*- coding: utf-8 -*-
"""验证 clip_grad_norm_ 的数值正确性 + 单独计时。"""
import json
import time

import numpy as np

import tgai_go.train as T
from tgai_go.train import nn_ops

T.set_default_device("cuda")
rt = T._cuda()

# ① 数值：321 个随机梯度，范数必须等于 float64 主机计算
rng = np.random.default_rng(0)
params = []
host_sq = 0.0
for i in range(321):
    n = int(rng.integers(64, 50000))
    arr = rng.normal(0, 1.0, n).astype(np.float32)
    p = T.tensor(np.zeros(n, np.float32), requires_grad=True)
    p.grad = T.tensor(arr)
    params.append(p)
    host_sq += float(np.sum(arr.astype(np.float64) ** 2))

ref = float(np.sqrt(host_sq))
got = float(nn_ops.clip_grad_norm_(params, None))     # None = 只求范数不裁剪
rel = abs(got - ref) / max(1e-12, ref)

# ② 计时（不裁剪，只求范数）
ts = []
for _ in range(5):
    rt.sync()
    t0 = time.perf_counter()
    nn_ops.clip_grad_norm_(params, None)
    rt.sync()
    ts.append(time.perf_counter() - t0)
ts.sort()

# ③ 裁剪语义：设一个很小的 max_norm，裁剪后范数应等于 max_norm
got2 = float(nn_ops.clip_grad_norm_(params, 0.5))
after_sq = 0.0
for p in params:
    d = np.asarray(rt.d2h(p.grad._gpu), np.float64)
    after_sq += float((d * d).sum())
after = float(np.sqrt(after_sq))

print("[clip] " + json.dumps({
    "n_params": 321,
    "norm_dev": round(got, 6), "norm_host_fp64": round(ref, 6),
    "rel_err": rel, "median_ms": round(ts[2] * 1e3, 3),
    "clip_to_0.5_returned": round(got2, 6),
    "norm_after_clip": round(after, 6),
    "host_change": round(abs(after - 0.5) / 0.5, 6),
}))
