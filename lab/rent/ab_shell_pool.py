# -*- coding: utf-8 -*-
"""B1 安全变体（空壳复用）的 A/B：prev(9a1a946) vs new，交替 3 轮。"""
import json
import os
import re
import statistics
import subprocess
import sys

PY = sys.executable
ROOT = "/root/bench"
NEW = "/root/patched_pkg"
PREV = "/root/prev_pkg"
STEPS = 20


def run(ppath, seq, ckpt):
    argv = [PY, "-m", "tgai_go.train.v3_pretrain",
            "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
            "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
            "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
            "--opt", "int8gpu", "--amp", "off", "--batch", "1",
            "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = ppath
    if ckpt:
        env["TG_GRAD_CKPT"] = "1"
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env, timeout=3600)
    out = (p.stdout or "") + (p.stderr or "")
    s = re.search(r'"step_s_steady":\s*([0-9.]+)', out)
    pk = re.search(r'"pool_peak_live_gib":\s*([0-9.]+)', out)
    lf = re.search(r'"loss_last":\s*([0-9.]+)', out)
    return (float(s.group(1)) if s else None,
            float(pk.group(1)) if pk else None,
            float(lf.group(1)) if lf else None)


print("=== B1 安全变体 A/B（4090，%d 步，交替 3 轮）===" % STEPS)
out = {}
for seq, ckpt in ((512, False), (2048, True)):
    key = "seq%d%s" % (seq, "+ckpt" if ckpt else "")
    a, b = [], []
    print("\n--- %s ---" % key)
    for i in range(3):
        r1 = run(PREV, seq, ckpt)
        r2 = run(NEW, seq, ckpt)
        print("  [%d] prev %-8s peak=%-6s loss=%-9s | new %-8s peak=%-6s loss=%-9s"
              % (i + 1, ("%.4f" % r1[0]) if r1[0] else "FAIL",
                 ("%.2f" % r1[1]) if r1[1] else "-",
                 ("%.7f" % r1[2]) if r1[2] else "-",
                 ("%.4f" % r2[0]) if r2[0] else "FAIL",
                 ("%.2f" % r2[1]) if r2[1] else "-",
                 ("%.7f" % r2[2]) if r2[2] else "-"), flush=True)
        if r1[0]:
            a.append(r1[0])
        if r2[0]:
            b.append(r2[0])
    if a and b:
        ma, mb = statistics.median(a), statistics.median(b)
        out[key] = {"prev": round(ma, 4), "new": round(mb, 4),
                    "prev_range": [round(min(a), 4), round(max(a), 4)],
                    "new_range": [round(min(b), 4), round(max(b), 4)],
                    "delta_pct": round(100.0 * (mb - ma) / ma, 1)}
        print("  => prev 中位 %.4f  new 中位 %.4f  (%+.1f%%)"
              % (ma, mb, out[key]["delta_pct"]))
print("\n[b1] " + json.dumps(out, ensure_ascii=False))
