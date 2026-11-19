# -*- coding: utf-8 -*-
"""API 探针：help / 优化器签名 / 小模型构建路径 / 数据文件。"""
import inspect
import io
import json
import os
import sys

import numpy as np

import tgai_go.train as T

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

out = {}

# ---- T.help() ----
buf = io.StringIO()
try:
    old = sys.stdout
    T.help()
    out["help_ok"] = True
except Exception as e:
    out["help_err"] = "%s: %s" % (type(e).__name__, e)
out["help_sig"] = str(inspect.signature(T.help))

# ---- 优化器签名 ----
sigs = {}
for n in ("AdamW", "Adafactor", "Muon", "LAMB", "Lion", "SGD"):
    c = getattr(T.optim, n, None)
    if c is None:
        continue
    try:
        sigs[n] = str(inspect.signature(c.__init__))
    except Exception as e:
        sigs[n] = "ERR:%s" % e
out["optim_sig"] = sigs

# ---- 小模型构建路径 ----
cand = {}
try:
    import tgai_go.train.v3_pretrain as vp
    cand["v3_pretrain_file"] = vp.__file__
    cand["v3_pretrain_names"] = [n for n in dir(vp) if not n.startswith("_")][:120]
except Exception as e:
    cand["v3_pretrain_err"] = "%s: %s" % (type(e).__name__, e)

# 找所有带 build/model 的候选函数
hits = []
for modname in ("tgai_go.train", "tgai_go.train.v3_pretrain", "tgai_go.train.preset",
                "tgai_go.train.v3_layer", "tgai_go.train.v3_moe"):
    try:
        m = __import__(modname, fromlist=["*"])
    except Exception:
        continue
    for n in dir(m):
        if n.startswith("_"):
            continue
        if ("build" in n.lower() or "model" in n.lower() or "arch" in n.lower()):
            f = getattr(m, n, None)
            if callable(f):
                try:
                    hits.append([modname, n, str(inspect.signature(f))])
                except Exception:
                    hits.append([modname, n, "?"])
out["build_candidates"] = hits

# ---- preset ----
try:
    out["preset_names"] = [n for n in dir(T.preset) if not n.startswith("_")]
except Exception as e:
    out["preset_names"] = "ERR:%s" % e

# ---- 数据 ----
data = {}
for d in (r"D:\orv_bench", r"D:\orv_bench\data2048_v4096", r"D:\orv_bench\data2048_v96000"):
    try:
        items = []
        for n in sorted(os.listdir(d))[:40]:
            p = os.path.join(d, n)
            items.append([n, os.path.getsize(p) if os.path.isfile(p) else "DIR"])
        data[d] = items
    except Exception as e:
        data[d] = "ERR:%s" % e
out["data"] = data

# ---- 是否有 fit / 训练循环原语 ----
out["loop_api"] = {n: hasattr(T, n) for n in
                   ("fit", "train_step", "backward", "no_grad", "clip_grad_norm",
                    "clip_grads", "lr_scheduler", "set_device", "set_default_device")}

# ---- torch ----
try:
    import subprocess
    r = subprocess.run([r"C:\Python311\python.exe", "-c",
                        "import torch;print(torch.__version__, torch.cuda.is_available(),"
                        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],
                       capture_output=True, text=True, timeout=180)
    out["torch"] = (r.stdout or "").strip() + (r.stderr or "").strip()
except Exception as e:
    out["torch"] = "ERR:%s" % e

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "p01_probe_api.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=str)
print("WROTE " + p)
print(json.dumps({k: (v if k in ("help_sig", "loop_api", "torch", "optim_sig") else "...")
                 for k, v in out.items()}, ensure_ascii=True, indent=1, default=str))
