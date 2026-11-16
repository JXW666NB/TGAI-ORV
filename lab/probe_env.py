# -*- coding: utf-8 -*-
"""Probe remote tgai_go install: version, module presence, hashes, disk, GPU."""
import hashlib
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

PKG = r"D:\orv_pip_0962\Lib\site-packages\tgai_go"
print("PKG exists", os.path.isdir(PKG))
TRAIN = os.path.join(PKG, "train")
print("TRAIN exists", os.path.isdir(TRAIN))
want = ["__init__.py", "v3_csa.py", "v3_moe.py", "v3_ops.py", "ckpt.py", "v3_masks.py",
        "v3_train_cli.py", "sched.py", "opt_fp16mv.py", "checkpoints.py", "v3_attnres.py",
        "nn_ops.py", "preset.py", "data.py", "loader.py", "cuda.py"]
for w in want:
    p = os.path.join(TRAIN, w)
    if os.path.exists(p):
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        print("  %-18s %10d %s" % (w, os.path.getsize(p), h))
    else:
        print("  %-18s MISSING" % (w,))

initp = os.path.join(PKG, "__init__.py")
if os.path.exists(initp):
    h = hashlib.md5(open(initp, "rb").read()).hexdigest()
    print("  tgai_go/__init__.py %10d %s" % (os.path.getsize(initp), h))

import tgai_go
print("version", getattr(tgai_go, "__version__", "?"), "file", tgai_go.__file__)
import numpy
print("numpy", numpy.__version__)
import tgai_go.train as T
print("copy_h2d_into", hasattr(T._cuda(), "copy_h2d_into"))
print("linear_ce", hasattr(T.functional, "linear_cross_entropy"))
print("clamp", hasattr(T, "clamp"))
from tgai_go.train import v3_masks
print("v3_masks ok", hasattr(v3_masks, "seg_additive"))
from tgai_go.train import v3_train_cli
print("v3_train_cli build", hasattr(v3_train_cli, "build"))
print("torch?", end=" ")
try:
    import torch
    print(torch.__version__, torch.cuda.is_available())
except Exception as e:
    print("no torch", e)

print("--- disk ---")
for d in ("D:\\", "C:\\"):
    try:
        out = subprocess.run(["wmic", "logicaldisk", "get", "caption,freespace,size"],
                             capture_output=True, text=True).stdout
        print(out.strip()[:600])
        break
    except Exception as e:
        print("disk probe fail", e)
print("--- gpu ---")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
print("--- dirs ---")
for d in (r"D:\orv_runs", r"D:\orv_src\tools"):
    print(d, os.path.isdir(d))
if os.path.isdir(r"D:\orv_src\tools"):
    print("  tools n files", len(os.listdir(r"D:\orv_src\tools")))
