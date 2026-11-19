# -*- coding: utf-8 -*-
"""模型/数据探针：npz 结构、迷你 V3 模型参数表、初始化分布、单步 timing。"""
import json
import os
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

# ---- npz 结构 ----
try:
    f = np.load(os.path.join(DATA, "shard_000.npz"))
    info = {}
    for k in f.files:
        a = f[k]
        info[k] = {"shape": list(a.shape), "dtype": str(a.dtype),
                   "min": int(a.min()) if a.size else None,
                   "max": int(a.max()) if a.size else None}
    out["npz"] = info
    out["npz_files"] = list(f.files)
except Exception as e:
    out["npz_err"] = "%s: %s" % (type(e).__name__, e)

# ---- 迷你 V3 模型 ----
CFG = dict(vocab=4096, d_model=128, n_layers=2, n_heads=4, d_ff=512, norm="rms",
           tie_embedding=True)
try:
    model = T.preset.transformer(**CFG)
    params = list(model.parameters())
    seen = {}
    dup = 0
    rows = []
    for p in params:
        if id(p) in seen:
            dup += 1
        seen[id(p)] = 1
        rows.append({"shape": list(p.shape), "numel": int(p.numel()),
                     "device": str(getattr(p, "device", "?")),
                     "dtype": str(getattr(p, "dtype", "?"))})
    out["model"] = {"cfg": CFG, "n_params_listed": len(params), "n_unique": len(seen),
                    "n_duplicate_entries": dup, "n_params_model": model.n_params(),
                    "total_numel_listed": sum(r["numel"] for r in rows),
                    "params": rows}
    # 初始化分布
    dist = []
    for p in params:
        a = np.asarray(p.data, np.float32)
        dist.append({"shape": list(p.shape), "std": float(a.std()),
                     "mean": float(a.mean()), "absmax": float(np.abs(a).max())})
    out["init_dist"] = dist
except Exception as e:
    out["model_err"] = "%s: %s" % (type(e).__name__, e)

# ---- 单步（CPU, AdamW）----
try:
    f = np.load(os.path.join(DATA, "shard_000.npz"))
    arr = f[f.files[0]]
    flat = np.asarray(arr).reshape(-1)
    out["tokens_total_shard0"] = int(flat.size)
    seq = 64
    x = flat[:seq].astype(np.int64)
    y = flat[1:seq + 1].astype(np.int64)
    out["x_first8"] = [int(v) for v in x[:8]]
    opt = T.optim.AdamW(list(model.parameters()), lr=1e-3)
    ps = list(model.parameters())
    l0 = float(np.asarray(model(x, y).to_numpy() if hasattr(model(x, y), "to_numpy") else 0))
    ts = []
    for i in range(3):
        t0 = time.perf_counter()
        loss = model(x, y)
        for p in ps:
            p.grad = None
        loss.backward()
        opt.step()
        ts.append(time.perf_counter() - t0)
    out["step_s"] = ts
    out["loss_at_init"] = l0
    out["loss_after3"] = float(np.asarray(loss.to_numpy()))
except Exception as e:
    out["step_err"] = "%s: %s" % (type(e).__name__, e)

out["has_clip_grad_norm_"] = hasattr(T, "clip_grad_norm_")
out["has_amp"] = hasattr(T, "amp")
out["default_device"] = str(T.get_device())
out["torch_mod_used"] = "tgai_go.train"

# ---- numpy 线程/CPU 信息 ----
try:
    out["cpu_count"] = os.cpu_count()
    out["np_show_config"] = str(np.show_config.__doc__)[:0]
    out["numpy_threads"] = os.environ.get("OMP_NUM_THREADS", "unset")
except Exception:
    pass

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "p02_probe_model.json")
with open(p, "w", encoding="utf-8") as fp:
    json.dump(out, fp, ensure_ascii=False, indent=1, default=str)
print("WROTE " + p)
print(json.dumps({k: v for k, v in out.items()
                  if k in ("npz", "npz_files", "step_s", "loss_at_init",
                           "loss_after3", "cpu_count", "default_device",
                           "has_clip_grad_norm_")},
                 ensure_ascii=True, indent=1, default=str))
