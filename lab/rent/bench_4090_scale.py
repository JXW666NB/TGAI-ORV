# -*- coding: utf-8 -*-
"""规模标度：3/6/12 层（seq512），ORV vs PyT，看倍数是否随规模稳定。"""
import json
import os
import re
import statistics
import subprocess
import sys

PY = sys.executable
ROOT = "/root/bench"
PKG = "/root/patched_pkg"
STEPS = 20
LAYERS = [3, 6, 12]


def run(argv, env_extra):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(env_extra)
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env, timeout=3600)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r'"step_s_steady":\s*([0-9.]+)', out) or \
        re.search(r'"step_s":\s*([0-9.]+)', out)
    pa = re.search(r'"params":\s*([0-9]+)', out)
    if m is None:
        return None, None
    return float(m.group(1)), (int(pa.group(1)) if pa else None)


def orv(layers):
    argv = [PY, "-m", "tgai_go.train.v3_pretrain",
            "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
            "--seq", "512", "--d", "1024", "--layers", str(layers), "--E", "4",
            "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
            "--opt", "int8gpu", "--amp", "off", "--batch", "1",
            "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    return run(argv, {"PYTHONPATH": PKG})


def pyt(layers):
    argv = [PY, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
            "--layers", str(layers), "--E", "4", "--vocab", "96000",
            "--d_ff", "3328", "--n_heads", "16", "--top_k", "2", "--seq", "512",
            "--steps", str(STEPS), "--ckpt", "0"]
    return run(argv, {})


print("=== 规模标度（4090 / seq512 / 20 步 / 2 轮中位）===")
table = {}
for L in LAYERS:
    o, p, params = [], [], None
    for _ in range(2):
        a, pa = orv(L)
        if a:
            o.append(a)
            params = pa or params
        b, _ = pyt(L)
        if b:
            p.append(b)
    if not o or not p:
        print("  L=%-2d 数据不全" % L)
        continue
    mo, mp = statistics.median(o), statistics.median(p)
    table[L] = {"params": params, "orv": round(mo, 4), "pyt": round(mp, 4),
                "speedup": round(mp / mo, 2)}
    print("  L=%-2d params=%-10s ORV %.4fs  PyT %.4fs  => %.2f×"
          % (L, params, mo, mp, mp / mo), flush=True)

print("\n[scale] " + json.dumps(table, ensure_ascii=False))
