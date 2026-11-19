# -*- coding: utf-8 -*-
"""用**真实规模**验证 clip（98.3M 的 emb/head 梯度）—— 上次用 5 万元素测，漏掉了带宽病。

判据：
  1) 范数与主机 float64 相对误差 ≤ 1e-6
  2) 裁剪后范数精确等于 max_norm
  3) 321 张量（含 2×98.3M）的耗时 —— 旧路径应 ≈10 ms 量级，而不是 286 ms
"""
import json
import time

import numpy as np

import tgai_go.train as T
from tgai_go.train import nn_ops

T.set_default_device("cuda")
rt = T._cuda()

rng = np.random.default_rng(0)
params = []
host_sq = 0.0
# 两个 98.3M（= 96000×1024，emb 与 head 的真实形状）
for n in (96000 * 1024, 96000 * 1024):
    arr = rng.normal(0, 1.0, n).astype(np.float32)
    p = T.tensor(np.zeros(n, np.float32), requires_grad=True)
    p.grad = T.tensor(arr)
    params.append(p)
    host_sq += float(np.sum(arr.astype(np.float64) ** 2))
# 其余 319 个小张量
for _ in range(319):
    n = int(rng.integers(64, 50000))
    arr = rng.normal(0, 1.0, n).astype(np.float32)
    p = T.tensor(np.zeros(n, np.float32), requires_grad=True)
    p.grad = T.tensor(arr)
    params.append(p)
    host_sq += float(np.sum(arr.astype(np.float64) ** 2))

ref = float(np.sqrt(host_sq))
got = float(nn_ops.clip_grad_norm_(params, None))
rel = abs(got - ref) / max(1e-12, ref)

ts = []
for _ in range(3):
    rt.sync()
    t0 = time.perf_counter()
    nn_ops.clip_grad_norm_(params, None)
    rt.sync()
    ts.append(time.perf_counter() - t0)
ts.sort()

got2 = float(nn_ops.clip_grad_norm_(params, 0.5))
after_sq = 0.0
for p in params[:4]:                       # 大张量抽样验裁剪
    d = np.asarray(rt.d2h(p.grad._gpu), np.float64)
    after_sq += float((d * d).sum())
scale_expect = 0.5 / (ref + 1e-6)
after_expect = scale_expect * np.sqrt(host_sq)

print("[clip98m] " + json.dumps({
    "n_params": len(params),
    "elems": sum(int(p.numel()) for p in params),
    "norm_dev": round(got, 4), "norm_host": round(ref, 4), "rel_err": rel,
    "median_ms": round(ts[1] * 1e3, 2),
    "clip_returned": round(got2, 4),
    "clip_after_norm": round(float(np.sqrt(after_sq * (ref ** 2) / max(1e-12, host_sq))), 6),
}))
