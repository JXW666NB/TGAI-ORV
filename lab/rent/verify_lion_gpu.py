# -*- coding: utf-8 -*-
"""Lion 设备端单 kernel 的验证：数值等价 + 速度。

判据：
  1) 与 numpy 参考在**同一份数据**上逐位/近逐位一致（rel <= 1e-6）
  2) 9.4M 参数规模的 opt.step 耗时（对照：修前 149 ms）
"""
import json
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()

rng = np.random.default_rng(0)


def numpy_ref(ps, gs, ms, b1, b2, lr, wd):
    out = []
    for p, g, m in zip(ps, gs, ms):
        u = b1 * m + (1.0 - b1) * g
        s = np.sign(u)
        if wd:
            s = s + wd * p
        pn = p - lr * s
        mn = b2 * m + (1.0 - b2) * g
        out.append((pn, mn))
    return out


# ---- 1) 数值等价（小规模，真实形状组合）----
shapes = [(64, 32), (128,), (33, 17), (1,), (7, 5)]
ps_np, gs_np, ms_np = [], [], []
params = []
for sh in shapes:
    p = rng.normal(0, 0.1, sh).astype(np.float32)
    g = rng.normal(0, 1.0, sh).astype(np.float32)
    m = rng.normal(0, 0.01, sh).astype(np.float32)
    ps_np.append(p.copy()); gs_np.append(g.copy()); ms_np.append(m.copy())


# 直接在设备上搭张量
dev_params = []
for p, g in zip(ps_np, gs_np):
    t = T.tensor(p.copy(), requires_grad=True)
    t.grad = T.tensor(g.copy())
    dev_params.append(t)

b1, b2, lr, wd = 0.9, 0.99, 1e-3, 0.01
opt = T.optim.Lion(dev_params, lr=lr, betas=(b1, b2), weight_decay=wd)
# 把初值动量灌成 numpy 参考里那份（Lion.__init__ 建的是 0）
for i, m0 in enumerate(ms_np):
    opt.m[i] = rt.h2d(np.ascontiguousarray(m0), dtype=np.float32)
opt.step()
rt.sync()

ref = numpy_ref(ps_np, gs_np, ms_np, b1, b2, lr, wd)
max_p, max_m = 0.0, 0.0
for i in range(len(ref)):
    p_got = np.asarray(rt.d2h(dev_params[i]._gpu), np.float64)
    m_got = np.asarray(rt.d2h(opt.m[i]), np.float64)
    max_p = max(max_p, float(np.abs(p_got - ref[i][0]).max()))
    max_m = max(max_m, float(np.abs(m_got - ref[i][1]).max()))

# ---- 2) 速度：9.4M 参数 ----
big = []
sizes = [256 * 1024] * 30 + [1024 * 512] + [128 * 1024] * 8   # ≈ 9.4M
tot = sum(sizes)
for n in sizes:
    p = T.tensor(rng.normal(0, 0.01, n).astype(np.float32), requires_grad=True)
    p.grad = T.tensor(rng.normal(0, 1.0, n).astype(np.float32))
    big.append(p)
opt2 = T.optim.Lion(big, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.01)
for _ in range(3):
    opt2.step()
rt.sync()
ts = []
for _ in range(3):
    rt.sync()
    t0 = time.perf_counter()
    opt2.step()
    rt.sync()
    ts.append((time.perf_counter() - t0) * 1e3)
ts.sort()

print("[lion_gpu] " + json.dumps({
    "params": int(tot),
    "n_tensors": len(big),
    "max_abs_diff_param": max_p,
    "max_abs_diff_moment": max_m,
    "opt_step_ms_median": round(ts[1], 3),
    "min": round(ts[0], 3), "max": round(ts[-1], 3),
}))
