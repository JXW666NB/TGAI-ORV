# -*- coding: utf-8 -*-
"""关键 pilot：固定 8 个 batch 反复过拟合，loss 会不会动？顺带标定 CPU 算力。"""
import json
import os
import sys
import time

import numpy as np

import pbench_lib as L

L.utf8()
import tgai_go.train as T

try:
    import polaris  # noqa: F401
    P = T.plugin.resolve_optimizer("Polaris")
except Exception as e:
    P = None
    print("no polaris %s" % e)

out = {}

# ---- CPU 算力标定（同一窗口内可比）----
a = np.random.rand(512, 512).astype(np.float32)
b = np.random.rand(512, 512).astype(np.float32)
best = 1e9
for _ in range(5):
    t0 = time.perf_counter()
    for _ in range(5):
        c = a @ b
    best = min(best, (time.perf_counter() - t0) / 5)
out["calib_matmul512_gflops"] = round(2 * 512 ** 3 / best / 1e9, 1)
out["calib_matmul512_ms"] = round(best * 1e3, 3)

# 纯 python 标定（极性化的 sumsq 循环那种）
t0 = time.perf_counter()
acc = np.float32(0.0)
arr = np.ones(200000, np.float32)
for v in arr:
    acc = np.float32(acc + np.float32(v * v))
out["calib_py_sum_200k_s"] = round(time.perf_counter() - t0, 4)

SEQ, NB = 64, 8
xs, ys = L.model_blocks(SEQ, NB)
X = np.stack(xs)          # [8,64]
Y = np.stack(ys)
out["n_blocks"] = NB

STEPS = 300


def train(name, lr, steps=STEPS, batch=NB):
    T.manual_seed(1000)
    model = T.preset.transformer(vocab=4096, d_model=128, n_layers=2, n_heads=4, d_ff=512)
    ps = list(model.parameters())
    opt = L.make_opt(T, name, ps, lr)
    curve, ts = [], []
    for i in range(steps):
        t0 = time.perf_counter()
        logits = model(X[:batch])                    # [B,S,V]
        lg = logits.reshape(batch * SEQ, 4096)
        loss = L.ce_vec(T, lg, Y[:batch].reshape(-1))
        for p in ps:
            p.grad = None
        loss.backward()
        opt.step()
        curve.append(float(np.asarray(loss.to_numpy())))
        ts.append(time.perf_counter() - t0)
    r = {"curve": [round(v, 5) for v in curve],
         "at10": curve[9], "at25": curve[24], "at50": curve[49],
         "at100": curve[99] if steps > 100 else None,
         "at200": curve[199] if steps > 200 else None,
         "final": curve[-1], "min": float(min(curve)),
         "t_step_s": float(np.mean(ts[20:])) if steps > 25 else float(np.mean(ts))}
    if name == "Polaris" and hasattr(opt, "status"):
        r["status"] = opt.status()
    return r


tests = [("AdamW", 1e-3, 300), ("AdamW", 3e-3, 300), ("AdamW", 1e-2, 300)]
if P is not None:
    tests.append(("Polaris", 1e-3, 150))
for name, lr, st in tests:
    try:
        r = train(name, lr, steps=st)
        out["%s@%.0e" % (name, lr)] = r
        print("%-9s lr=%-7.0e 10:%.3f 25:%.3f 50:%.3f 100:%s 200:%s fin:%.3f min:%.3f "
              "s/step:%.3f %s"
              % (name, lr, r["at10"], r["at25"], r["at50"],
                 ("%.3f" % r["at100"]) if r["at100"] else "-",
                 ("%.3f" % r["at200"]) if r["at200"] else "-",
                 r["final"], r["min"], r["t_step_s"],
                 r.get("status", {})), flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        out["%s@%.0e" % (name, lr)] = "ERR %s: %s" % (type(e).__name__, e)

# 单优化器 step 成本（同一模型，仅 opt.step 计时）
T.manual_seed(1000)
model = T.preset.transformer(vocab=4096, d_model=128, n_layers=2, n_heads=4, d_ff=512)
ps = list(model.parameters())
costs = {}
for name, lr in (("AdamW", 1e-3), ("Adafactor", 1e-2), ("Muon", 1e-2),
                 ("LAMB", 1e-3), ("Polaris", 1e-3)):
    try:
        opt = L.make_opt(T, name, ps, lr)
        ts = []
        for _ in range(6):
            logits = model(X[:2])
            loss = L.ce_vec(T, logits.reshape(2 * SEQ, 4096), Y[:2].reshape(-1))
            for p in ps:
                p.grad = None
            loss.backward()
            t0 = time.perf_counter()
            opt.step()
            ts.append(time.perf_counter() - t0)
        costs[name] = {"opt_step_ms": round(float(np.median(ts[2:])) * 1e3, 1),
                       "all_ms": [round(t * 1e3, 1) for t in ts]}
    except Exception as e:
        costs[name] = "ERR %s: %s" % (type(e).__name__, e)
out["opt_step_cost"] = costs
print(json.dumps(costs, ensure_ascii=True, indent=1))

L.dump("p06_pilot2", out)
