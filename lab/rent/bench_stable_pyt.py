# -*- coding: utf-8 -*-
"""PyTorch 侧的同协议稳定基准（预热 1 次 + 3 次，取中位数）。"""
import json
import os
import re
import statistics
import subprocess
import sys

PY = r"C:\Python311\python.exe"
ROOT = r"D:\orv_bench"


def one(seq, layers=12, steps=20):
    cmd = [PY, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
           "--layers", str(layers), "--E", "4", "--vocab", "96000",
           "--d_ff", "3328", "--n_heads", "16", "--top_k", "2",
           "--seq", str(seq), "--steps", str(steps), "--ckpt", "0", "--amp"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r'"step_s":\s*([0-9.]+)', out)
    return float(m.group(1)) if m else None


for seq in ([int(x) for x in sys.argv[1:]] or [2048, 512]):
    print("===== pytorch seq%d =====" % seq, flush=True)
    w = one(seq)
    if w is None:
        print("  预热失败")
        continue
    print("  [warmup] %.3f s" % w, flush=True)
    vals = []
    for i in range(3):
        v = one(seq)
        if v is None:
            continue
        vals.append(v)
        print("  [run %d] %.3f s" % (i + 1, v), flush=True)
    if vals:
        med = statistics.median(vals)
        print("[pyt] " + json.dumps({
            "seq": seq, "median": round(med, 3), "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "spread_pct": round(100.0 * (max(vals) - min(vals)) / med, 1)}))
