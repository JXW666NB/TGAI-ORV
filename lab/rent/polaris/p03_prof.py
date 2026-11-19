# -*- coding: utf-8 -*-
"""为什么 CPU 每步 10 秒？分解 forward / backward / step，并 cProfile 热点。"""
import cProfile
import io
import json
import os
import pstats
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import tgai_go.train as T

out = {}
ROOT = r"D:\orv_bench"
DATA = os.path.join(ROOT, "data2048_v4096")
seq = 64
f = np.load(os.path.join(DATA, "shard_000.npz"))
ids = np.asarray(f["input_ids"]).reshape(-1)
x = ids[:seq].astype(np.int64)
y = ids[1:seq + 1].astype(np.int64)

model = T.preset.transformer(vocab=4096, d_model=128, n_layers=2, n_heads=4, d_ff=512)
ps = list(model.parameters())

# numpy 本身的算力（排除 CPU 太慢）
a = np.random.rand(512, 512).astype(np.float32)
b = np.random.rand(512, 512).astype(np.float32)
t0 = time.perf_counter()
for _ in range(20):
    c = a @ b
dt = (time.perf_counter() - t0) / 20
out["np_matmul_512_gflops"] = round(2 * 512 ** 3 / dt / 1e9, 2)
out["np_matmul_512_ms"] = round(dt * 1e3, 2)

# forward only
with T.no_grad():
    model(x, y)
    t0 = time.perf_counter()
    for _ in range(2):
        model(x, y)
    out["fwd_s"] = round((time.perf_counter() - t0) / 2, 3)

# forward + backward
t0 = time.perf_counter()
loss = model(x, y)
out["fwd1_s"] = round(time.perf_counter() - t0, 3)
t0 = time.perf_counter()
loss.backward()
out["bwd_s"] = round(time.perf_counter() - t0, 3)

# optimizer cost per class (1 step)
opt_times = {}
for name, mk in (("AdamW", lambda: T.optim.AdamW(ps, lr=1e-3)),
                 ("Muon", lambda: T.optim.Muon(ps, lr=1e-2)),
                 ("Adafactor", lambda: T.optim.Adafactor(ps, lr=1e-2)),
                 ("LAMB", lambda: T.optim.LAMB(ps, lr=1e-3))):
    try:
        o = mk()
        t0 = time.perf_counter()
        o.step()
        opt_times[name] = round(time.perf_counter() - t0, 3)
    except Exception as e:
        opt_times[name] = "ERR:%s: %s" % (type(e).__name__, e)
out["opt_step_s"] = opt_times

try:
    import polaris  # noqa: F401
    P = T.plugin.resolve_optimizer("Polaris")
    o = P(ps, lr=1e-3)
    t0 = time.perf_counter()
    o.step()
    opt_times["Polaris"] = round(time.perf_counter() - t0, 3)
    out["polaris_status_1"] = o.status()
except Exception as e:
    opt_times["Polaris"] = "ERR:%s: %s" % (type(e).__name__, e)

# cProfile 热点（forward+backward+AdamW step）
pr = cProfile.Profile()
pr.enable()
for _ in range(2):
    loss = model(x, y)
    for p in ps:
        p.grad = None
    loss.backward()
    T.optim.AdamW(ps, lr=1e-3).step()
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(28)
out["profile"] = s.getvalue()

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
p = os.path.join(OUT, "p03_prof.json")
with open(p, "w", encoding="utf-8") as fp:
    json.dump(out, fp, ensure_ascii=False, indent=1, default=str)
print("WROTE " + p)
print(json.dumps({k: v for k, v in out.items() if k != "profile"},
                 ensure_ascii=True, indent=1, default=str))
print(out["profile"][:4000])
