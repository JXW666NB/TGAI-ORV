# -*- coding: utf-8 -*-
"""ORV 侧基准（Linux/4090 版）。协议与 V100 上的 bench_stable.py 完全一致：
预热 1 次 + 3 次，取 step_s_steady 的中位数，报极差。

用法： python bench_orv_4090.py 512 2048
"""
import json
import os
import re
import statistics
import subprocess
import sys

PY = sys.executable
ROOT = "/root/bench"
WARMUP = 1
RUNS = 3


def one(seq, layers=12, steps=20):
    cmd = [PY, "-m", "tgai_go.train.v3_pretrain",
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", str(layers), "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
           "--steps", str(steps), "--log-every", "99", "--no-final-save"]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env, timeout=3600)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r'"step_s_steady":\s*([0-9.]+)', out)
    if m is None:
        tail = "\n".join(out.strip().splitlines()[-4:])
        print("  失败，末尾输出:\n" + tail, flush=True)
        return None, None
    pk = re.search(r'"peak_gib":\s*([0-9.]+)', out)
    return float(m.group(1)), (float(pk.group(1)) if pk else None)


res = {}
for seq in ([int(x) for x in sys.argv[1:]] or [512, 2048]):
    print("===== orv seq%d =====" % seq, flush=True)
    w, pk = one(seq)
    if w is None:
        print("  预热失败", flush=True)
        continue
    print("  [warmup] %.3f s" % w, flush=True)
    vals, peaks = [], []
    for i in range(RUNS):
        v, pk = one(seq)
        if v is None:
            continue
        vals.append(v)
        if pk is not None:
            peaks.append(pk)
        print("  [run %d] %.3f s" % (i + 1, v), flush=True)
    if vals:
        med = statistics.median(vals)
        res[str(seq)] = {
            "median": round(med, 3), "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "spread_pct": round(100.0 * (max(vals) - min(vals)) / med, 1),
            "peak_gib": (round(max(peaks), 2) if peaks else None)}
print("\n[orv] " + json.dumps(res, ensure_ascii=False))
