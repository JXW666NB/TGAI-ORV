# -*- coding: utf-8 -*-
"""最终权威表：seq512（无 ckpt）与 seq2048（同开 ckpt），两侧交替 3 轮。"""
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


def run(argv, env_extra):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(env_extra)
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env, timeout=3600)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r'"step_s_steady":\s*([0-9.]+)', out) or \
        re.search(r'"step_s":\s*([0-9.]+)', out)
    pk = re.search(r'"pool_peak_live_gib":\s*([0-9.]+)', out)
    return (float(m.group(1)) if m else None,
            float(pk.group(1)) if pk else None)


def orv(seq, ckpt):
    argv = [PY, "-m", "tgai_go.train.v3_pretrain",
            "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
            "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
            "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
            "--opt", "int8gpu", "--amp", "off", "--batch", "1",
            "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    e = {"PYTHONPATH": PKG}
    if ckpt:
        e["TG_GRAD_CKPT"] = "1"
    return run(argv, e)


def pyt(seq, ckpt):
    argv = [PY, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
            "--layers", "12", "--E", "4", "--vocab", "96000", "--d_ff", "3328",
            "--n_heads", "16", "--top_k", "2", "--seq", str(seq),
            "--steps", str(STEPS), "--ckpt", "1" if ckpt else "0"]
    return run(argv, {})


print("=== 最终权威表（4090 24GB，%d 步，交替 3 轮）===" % STEPS)
out = {}
for seq, ckpt in ((512, False), (2048, True)):
    key = "seq%d%s" % (seq, "+ckpt" if ckpt else "")
    o, p = [], []
    print("\n--- %s ---" % key)
    for i in range(3):
        a, pa = orv(seq, ckpt)
        b, pb = pyt(seq, ckpt)
        print("  [%d] ORV %-8s peak=%-6s | PyT %-8s"
              % (i + 1, ("%.4f" % a) if a else "FAIL",
                 ("%.2f" % pa) if pa else "-", ("%.4f" % b) if b else "FAIL"),
              flush=True)
        if a:
            o.append(a)
        if b:
            p.append(b)
    out[key] = {
        "orv": round(statistics.median(o), 4) if o else None,
        "orv_range": [round(min(o), 4), round(max(o), 4)] if o else None,
        "pyt": round(statistics.median(p), 4) if p else None,
        "pyt_range": [round(min(p), 4), round(max(p), 4)] if p else None,
        "peak": round(pa, 2) if pa else None,
    }
    if o and p:
        out[key]["speedup"] = round(statistics.median(p) / statistics.median(o), 2)
        print("  => ORV 中位 %.4f  vs  PyT 中位 %.4f  =>  ORV 快 %.2f×"
              % (statistics.median(o), statistics.median(p),
                 statistics.median(p) / statistics.median(o)))

print("\n[final] " + json.dumps(out, ensure_ascii=False))
