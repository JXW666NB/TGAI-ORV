# -*- coding: utf-8 -*-
"""GPU 冒烟：preset Transformer 能不能在 CUDA 上 fwd/bwd；Polaris 能不能 step。"""
import json
import os
import sys
import time

import numpy as np

import pbench_lib as L

L.utf8()
import tgai_go.train as T

out = {}
try:
    T.set_default_device("cuda")
    out["device"] = str(T.get_device())
    rt = T._cuda()
    out["caps"] = T.device_caps() if hasattr(T, "device_caps") else None
    out["pool_stats_keys"] = list(rt.pool_stats().keys())
except Exception as e:
    out["set_device_err"] = "%s: %s" % (type(e).__name__, e)
    L.dump("g00_smoke", out)
    raise SystemExit(1)

# 1) 极简张量算子
try:
    w = T.tensor(np.random.randn(4, 3).astype(np.float32), requires_grad=True)
    x = T.tensor(np.random.randn(3, 2).astype(np.float32))
    y = (T.matmul(w, x)).sum()
    y.backward()
    out["matmul_backward"] = "ok grad_type=%s" % type(w.grad).__name__
except Exception as e:
    out["matmul_backward"] = "ERR %s: %s" % (type(e).__name__, e)

# 2) preset Transformer 前反向
SEQ = 64
xs, ys = L.model_blocks(SEQ, 2, shard_dir=None) if False else (None, None)
D = r"D:\orv_bench\data2048_v4096"
f = np.load(os.path.join(D, "shard_000.npz"))
ids = np.asarray(f["input_ids"]).reshape(-1)
x = ids[:SEQ].astype(np.int64)
y = ids[1:SEQ + 1].astype(np.int64)

model = None
try:
    T.manual_seed(1000)
    T.set_default_device("cuda")
    model = T.preset.transformer(vocab=4096, d_model=128, n_layers=2, n_heads=4, d_ff=512)
    dev0 = str(model.embed.weight.device)
    out["model_device"] = dev0
    t0 = time.perf_counter()
    logits = model(x)
    out["fwd_ok"] = True
    out["fwd_s"] = round(time.perf_counter() - t0, 3)
    out["logits_shape"] = list(logits.shape)
    lg = logits.reshape(SEQ, 4096)
    loss = L.ce_vec(T, lg, y)
    out["loss"] = float(np.asarray(loss.to_numpy()))
    ps = list(model.parameters())
    for p in ps:
        p.grad = None
    loss.backward()
    out["bwd_ok"] = True
    out["grad0_type"] = type(ps[0].grad).__name__
except Exception as e:
    import traceback
    out["model_err"] = "%s: %s" % (type(e).__name__, e)
    out["tb"] = traceback.format_exc()[-2000:]

# 3) 各优化器在 CUDA 上 step 一次
if model is not None:
    ps = list(model.parameters())
    steps = {}
    T.manual_seed(1000)
    for name, lr in (("AdamW", 1e-3), ("Adafactor", 1e-2), ("Muon", 1e-2),
                     ("LAMB", 1e-3), ("Lion", 1e-3), ("Polaris", 1e-3)):
        try:
            opt = L.make_opt(T, name, ps, lr)
            t0 = time.perf_counter()
            opt.step()
            rt.sync()
            steps[name] = round((time.perf_counter() - t0) * 1e3, 1)
        except Exception as e:
            steps[name] = "ERR %s: %s" % (type(e).__name__, e)
    out["first_step_ms"] = steps

try:
    out["mem_peak_gib"] = round(rt.pool_stats()["peak_live_bytes"] / 2 ** 30, 3)
except Exception as e:
    out["mem_err"] = str(e)

L.dump("g00_smoke", out)
print(json.dumps(out, ensure_ascii=True, indent=1, default=str)[:4000])
