# -*- coding: utf-8 -*-
"""4090 优化器对比：同一个模型、同一步数，只看整步耗时与峰值显存。

`v3_pretrain --opt` 的六档全部覆盖（adamw / muon / fp16mom / adafactor / int8 / int8gpu）。
注意：这里比的是**整步**（含前反向），所以差异只反映优化器本身 + 状态显存，
不是"纯 opt.step 微秒"；lr 相同、都是 20 步、`--amp off`、seq512、12 层。
"""
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
ROUNDS = 2
OPTS = ["int8gpu", "adamw", "fp16mom", "int8", "muon", "adafactor"]


def run(opt):
    argv = [PY, "-m", "tgai_go.train.v3_pretrain",
            "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
            "--seq", "512", "--d", "1024", "--layers", "12", "--E", "4",
            "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
            "--opt", opt, "--amp", "off", "--batch", "1",
            "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = PKG
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env, timeout=3600)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r'"step_s_steady":\s*([0-9.]+)', out)
    pk = re.search(r'"pool_peak_live_gib":\s*([0-9.]+)', out)
    lf = re.search(r'"loss_last":\s*([0-9.]+)', out)
    if m is None:
        low = out.lower()
        return None, None, None, ("OOM" if ("out of memory" in low or "显存不足" in out) else "FAIL")
    return (float(m.group(1)), float(pk.group(1)) if pk else None,
            float(lf.group(1)) if lf else None, "ok")


print("=== 4090 优化器对比（seq512 / 12 层 / 20 步 / amp off / %d 轮）===" % ROUNDS)
res = {}
for opt in OPTS:
    vals, peaks, losses, err = [], [], [], "ok"
    for _ in range(ROUNDS):
        s, pk, lf, e = run(opt)
        err = e
        if s:
            vals.append(s)
            if pk:
                peaks.append(pk)
            if lf:
                losses.append(lf)
    if vals:
        res[opt] = {"step_s": round(statistics.median(vals), 4),
                    "range": [round(min(vals), 4), round(max(vals), 4)],
                    "peak_gib": round(max(peaks), 2) if peaks else None,
                    "loss_last": round(max(losses), 5) if losses else None}
        print("  %-10s step %.4f s  峰值 %s GiB  loss_last %s"
              % (opt, res[opt]["step_s"], res[opt]["peak_gib"],
                 res[opt]["loss_last"]), flush=True)
    else:
        res[opt] = {"step_s": err}
        print("  %-10s %s" % (opt, err), flush=True)

print("\n[opts] " + json.dumps(res, ensure_ascii=False))
