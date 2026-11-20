# -*- coding: utf-8 -*-
"""PyTorch 侧基准（Linux/4090 版）。协议与 V100 上的 bench_stable_pyt.py 一致：
预热 1 次 + 3 次，取 v3_pyt_train.py 报的 step_s 中位数。

用法： python bench_pyt_4090.py 512 2048
"""
import json
import os
import re
import statistics
import subprocess
import sys

PY = sys.executable
ROOT = "/root/bench"


def one(seq, layers=12, steps=20):
    cmd = [PY, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
           "--layers", str(layers), "--E", "4", "--vocab", "96000",
           "--d_ff", "3328", "--n_heads", "16", "--top_k", "2",
           "--seq", str(seq), "--steps", str(steps), "--ckpt", "0", "--amp"]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env, timeout=3600)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r'"step_s":\s*([0-9.]+)', out)
    if m is None:
        tail = "\n".join(out.strip().splitlines()[-4:])
        print("  失败，末尾输出:\n" + tail, flush=True)
        return None
    return float(m.group(1))


res = {}
for seq in ([int(x) for x in sys.argv[1:]] or [512, 2048]):
    print("===== pytorch seq%d =====" % seq, flush=True)
    w = one(seq)
    if w is None:
        print("  预热失败", flush=True)
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
        res[str(seq)] = {
            "median": round(med, 3), "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "spread_pct": round(100.0 * (max(vals) - min(vals)) / med, 1)}
print("\n[pyt] " + json.dumps(res, ensure_ascii=False))
