# -*- coding: utf-8 -*-
"""CPU 侧正式矩阵（与 GPU 的 g10 完全同口径）：

  S1 玩具回归 600 步（与 GPU 逐配置对齐，用于 CPU/CUDA 路径差异）
  S2 迷你 V3 固定批过拟合 150 步 × 4 lr × 5 优化器
  S3 迷你 V3 流式（每步换一批）100 步 @ 各自最优 lr（泛化方向，不作主结论）
"""
import json
import os
import sys
import time

import numpy as np

import pbench_lib as L

L.utf8()
import tgai_go.train as T

assert T.get_device() == "cpu"
import polaris  # noqa: F401,E402

CPU_ORDS = ["AdamW", "Adafactor", "Muon", "Lion", "Polaris"]


def fmt(v, p=6):
    return ("%." + str(p) + "g") % v if isinstance(v, (int, float)) else str(v)


def calib():
    a = np.random.rand(512, 512).astype(np.float32)
    b = np.random.rand(512, 512).astype(np.float32)
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(5):
            a @ b
        best = min(best, (time.perf_counter() - t0) / 5)
    t0 = time.perf_counter()
    acc = np.float32(0.0)
    arr = np.ones(200000, np.float32)
    for v in arr:
        acc = np.float32(acc + np.float32(v * v))
    return {"matmul512_gflops": round(2 * 512 ** 3 / best / 1e9, 1),
            "py_sum_200k_s": round(time.perf_counter() - t0, 4)}


CAL = calib()
print("calib " + json.dumps(CAL), flush=True)

# ---------------------------------------------------------------- S1 toy 600
S1 = 600
recs1 = []
t0 = time.perf_counter()
for name in CPU_ORDS:
    for lr in L.GRID[name]:
        kw = {"ns_fast": True} if name == "Muon" else {}
        try:
            r = L.run_toy(T, name, lr, seed=0, steps=S1, opt_kw=kw)
            recs1.append(r)
            print("S1 %-9s lr=%-7.0e l10=%-11s excess=%-11s st<=1e-3=%-5s"
                  % (name, lr, fmt(r["last10_mean"], 7), fmt(r["excess_last10"], 4),
                     fmt(r["steps_to_excess_1e-3"], 5)), flush=True)
        except Exception as e:
            recs1.append({"task": "toy", "opt": name, "lr": lr,
                          "error": "%s: %s" % (type(e).__name__, e)})
            print("S1 %-9s lr=%-7.0e ERR %s" % (name, lr, e), flush=True)
L.dump("p12a_cpu_toy600", {"records": recs1, "steps": S1, "device": "cpu",
                           "loss_star": L.toy_problem()[5], "calib": CAL,
                           "wall_s": round(time.perf_counter() - t0, 1)})

# ------------------------------------------------------ S2 mini 过拟合 150 步
SEQ, NB, STEPS = 64, 8, 150
xs, ys = L.model_blocks(SEQ, NB)
X = np.stack(xs)
Y = np.stack(ys)
GRID_B = {"AdamW": [1e-3, 3e-3, 1e-2, 3e-2],
          "Adafactor": [3e-3, 1e-2, 3e-2, 1e-1],
          "Muon": [1e-3, 3e-3, 1e-2, 3e-2],
          "Lion": [3e-4, 1e-3, 3e-3, 1e-2],
          "Polaris": [3e-4, 1e-3, 3e-3, 1e-2]}


def train_overfit(name, lr, steps=STEPS):
    T.manual_seed(1000)
    model = T.preset.transformer(vocab=4096, d_model=128, n_layers=2, n_heads=4, d_ff=512)
    ps = list(model.parameters())
    kw = {"ns_fast": True} if name == "Muon" else {}
    opt = L.make_opt(T, name, ps, lr, **kw)
    curve, ts = [], []
    for i in range(steps):
        t0 = time.perf_counter()
        logits = model(X)
        loss = L.ce_vec(T, logits.reshape(NB * SEQ, 4096), Y.reshape(-1))
        for p in ps:
            p.grad = None
        loss.backward()
        opt.step()
        curve.append(float(np.asarray(loss.to_numpy())))
        ts.append(time.perf_counter() - t0)
    r = L.summarize(curve, t_steps=ts)
    r.update({"task": "mini_v3_overfit", "opt": name, "lr": lr, "seed": 0,
              "steps": steps, "batch": NB, "seq": SEQ, "device": "cpu",
              "tokens_per_step": NB * SEQ,
              "tokens_per_s": (NB * SEQ / r["t_step_s"]) if r.get("t_step_s") else None,
              "curve": [round(v, 6) if v == v else None for v in curve]})
    if name == "Polaris" and hasattr(opt, "status"):
        r["opt_status"] = opt.status()
    return r


recs2 = []
t0 = time.perf_counter()
for name in CPU_ORDS:
    for lr in GRID_B[name]:
        try:
            r = train_overfit(name, lr)
            recs2.append(r)
            print("S2 %-9s lr=%-7.0e l10=%-10s l25=%-10s l50=%-10s min=%-10s "
                  "ms/step=%-7.1f%s"
                  % (name, lr, fmt(r["last10_mean"], 6), fmt(r.get("loss_at_25"), 6),
                     fmt(r.get("loss_at_50"), 6), fmt(r["min"], 6),
                     (r["t_step_s"] or 0) * 1e3,
                     (" nssaved=%s" % r["opt_status"]["ns_saved_pct"])
                     if r.get("opt_status") else ""), flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            recs2.append({"task": "mini_v3_overfit", "opt": name, "lr": lr,
                          "error": "%s: %s" % (type(e).__name__, e)})
L.dump("p12b_cpu_overfit", {"records": recs2, "steps": STEPS, "device": "cpu",
                            "batch": NB, "seq": SEQ, "calib": CAL,
                            "wall_s": round(time.perf_counter() - t0, 1)})

# ------------------------------------------------- S3 流式（每步新数据）@最优 lr
best = L.table(recs2, "last10_mean")
STREAM_STEPS = 100
xs2, ys2 = L.model_blocks(SEQ, STREAM_STEPS)
recs3 = []
for name, r0 in best.items():
    lr = r0["lr"]
    try:
        T.manual_seed(1000)
        model = T.preset.transformer(vocab=4096, d_model=128, n_layers=2,
                                     n_heads=4, d_ff=512)
        ps = list(model.parameters())
        kw = {"ns_fast": True} if name == "Muon" else {}
        opt = L.make_opt(T, name, ps, lr, **kw)
        curve, ts = [], []
        for i in range(STREAM_STEPS):
            x, y = xs2[i][None, :], ys2[i][None, :]
            t0 = time.perf_counter()
            logits = model(x)
            loss = L.ce_vec(T, logits.reshape(SEQ, 4096), y.reshape(-1))
            for p in ps:
                p.grad = None
            loss.backward()
            opt.step()
            curve.append(float(np.asarray(loss.to_numpy())))
            ts.append(time.perf_counter() - t0)
        rr = L.summarize(curve, t_steps=ts)
        rr.update({"task": "mini_v3_stream", "opt": name, "lr": lr, "steps":
                   STREAM_STEPS, "seq": SEQ, "device": "cpu",
                   "curve": [round(v, 6) for v in curve]})
        recs3.append(rr)
        print("S3 %-9s lr=%-7.0e l10=%-10s l25=%-10s l50=%-10s"
              % (name, lr, fmt(rr["last10_mean"], 6), fmt(rr.get("loss_at_25"), 6),
                 fmt(rr.get("loss_at_50"), 6)), flush=True)
    except Exception as e:
        recs3.append({"task": "mini_v3_stream", "opt": name, "lr": lr,
                      "error": "%s: %s" % (type(e).__name__, e)})
L.dump("p12c_cpu_stream", {"records": recs3, "steps": STREAM_STEPS, "device": "cpu",
                           "calib": CAL})
print("DONE cpu matrix")
