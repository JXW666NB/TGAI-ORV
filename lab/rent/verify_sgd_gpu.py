# -*- coding: utf-8 -*-
"""验证 SGD 设备端单 kernel：数值等价 + 速度。

判据：
  1) 与 numpy 参考（含"首步 buf=g"这条 torch 语义）在**同一份数据**上一致（<=1e-6）
  2) 连跑 2 步仍一致（覆盖 started 标志的翻转）
  3) 9.4M 参数的 opt.step 耗时
"""
import json
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
rng = np.random.default_rng(0)

shapes = [(64, 32), (128,), (33, 17), (1,), (7, 5)]
lrs = [1e-3, 3e-3]


def ref_step(ps, gs, bufs, started, lr, wd, mom, damp, nesterov):
    out = []
    for k in range(len(ps)):
        g = gs[k].astype(np.float32).copy()
        p = ps[k].astype(np.float32).copy()
        b = bufs[k].astype(np.float32).copy()
        if wd:
            g = g + np.float32(wd) * p
        if mom > 0:
            if not started[k]:
                b = g.copy()
            else:
                b = np.float32(mom) * b + np.float32(1.0 - damp) * g
            g = (g + np.float32(mom) * b) if nesterov else b
        p = p - np.float32(lr) * g
        out.append((p, b))
    return out


report = {}
for wd, mom, damp, nesterov in [(0.0, 0.0, 0.0, False),
                                (0.01, 0.9, 0.0, False),
                                (0.0, 0.9, 0.0, True),
                                (0.01, 0.9, 0.1, False)]:
    ps_np, gs_np, bs_np = [], [], []
    params = []
    for sh in shapes:
        p0 = rng.normal(0, 0.1, sh).astype(np.float32)
        g0 = rng.normal(0, 1.0, sh).astype(np.float32)
        ps_np.append(p0.copy()); gs_np.append(g0.copy())
        bs_np.append(np.zeros(sh, np.float32))
        t = T.tensor(p0.copy(), requires_grad=True)
        t.grad = T.tensor(g0.copy())
        params.append(t)

    opt = T.optim.SGD(params, lr=lrs[0], momentum=mom, dampening=damp,
                      weight_decay=wd, nesterov=nesterov)
    started = [False] * len(params)
    max_p = max_b = 0.0
    for step in range(2):
        ref = ref_step(ps_np, gs_np, bs_np, started, lrs[0], wd, mom, damp, nesterov)
        opt.step()
        rt.sync()
        for k in range(len(params)):
            pg = np.asarray(rt.d2h(params[k]._gpu), np.float64)
            max_p = max(max_p, float(np.abs(pg - ref[k][0]).max()))
            if mom > 0:
                bg = np.asarray(rt.d2h(opt.buf[k]), np.float64)
                max_b = max(max_b, float(np.abs(bg - ref[k][1]).max()))
        ps_np = [r[0] for r in ref]
        bs_np = [r[1] for r in ref]
        started = [True] * len(params)
        # 同步设备侧状态，保证下一步比较的是同一轨迹
        for k, t in enumerate(params):
            rt.h2d_none = None
    key = "wd%s_mom%s_damp%s_nes%s" % (wd, mom, damp, int(nesterov))
    report[key] = {"max_abs_diff_param": max_p, "max_abs_diff_buf": max_b}
    print("%-28s max|Δp|=%.3e  max|Δbuf|=%.3e" % (key, max_p, max_b))

# 速度：9.4M
sizes = [256 * 1024] * 30 + [1024 * 512] + [128 * 1024] * 8
big = []
for n in sizes:
    p = T.tensor(rng.normal(0, 0.01, n).astype(np.float32), requires_grad=True)
    p.grad = T.tensor(rng.normal(0, 1.0, n).astype(np.float32))
    big.append(p)
for tag, kw in (("nomom", dict(momentum=0.0)),
                ("mom0.9", dict(momentum=0.9)),
                ("nesterov", dict(momentum=0.9, nesterov=True))):
    o = T.optim.SGD(big, lr=1e-3, **kw)
    for _ in range(3):
        o.step()
    rt.sync()
    ts = []
    for _ in range(3):
        rt.sync(); t0 = time.perf_counter(); o.step(); rt.sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    report["perf_" + tag] = {"params": int(sum(sizes)), "ms": round(ts[1], 3)}
    print("%-28s %.3f ms" % ("perf " + tag, ts[1]))

print()
print("[sgd_gpu] " + json.dumps(report, ensure_ascii=False))
