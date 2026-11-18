# -*- coding: utf-8 -*-
"""batch>1 修好之后的吞吐收益（这就是之前一直拿不到的那块性能）。

测 Mos 12L / seq512，两侧同口径（ckpt=0 / AMP 开），batch = 1 / 2 / 4。
ORV 与 PyTorch 都测，避免"只有我们能用 batch"的错觉。
"""
import json
import os
import re
import subprocess

ROOT = r"D:\orv_bench"
ORV = r"D:\orv_pip_0962\Scripts\python.exe"
PYT = r"C:\Python311\python.exe"
STEPS = 20


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    return (p.stdout or "") + (p.stderr or "")


def orv(batch):
    cmd = [ORV, os.path.join(ROOT, "split_orv.py"),
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", "512", "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--amp", "gemm", "--batch", str(batch),
           "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    out = run(cmd)
    m = re.search(r"^RESULT (\{.*\})$", out, re.M)
    if not m:
        why = "OOM" if ("显存不足" in out or "out of memory" in out.lower()) else "FAIL"
        return None, why
    d = json.loads(m.group(1))
    return {"step": d["step_s_steady"], "mem": d["pool_peak_live_gib"],
            "tok": d["tok_s_steady"]}, None


print("===== ORV：batch 对吞吐的影响（MoE 12L / seq512 / 20 步）=====")
base = None
for b in (1, 2, 4):
    d, why = orv(b)
    if not d:
        print("  batch=%d  FAIL: %s" % (b, why), flush=True)
        continue
    if base is None:
        base = d["tok"]
    print("  batch=%-2d %6.3f s/步  %8.1f tok/s  %5.1f GiB   相对 batch=1：%.2fx"
          % (b, d["step"], d["tok"], d["mem"], d["tok"] / base), flush=True)
