# -*- coding: utf-8 -*-
"""AMP 对照：PyTorch(--amp, ckpt=0) vs ORV(--amp gemm/both)。

两侧都测，才谈得上公平。ORV 的 amp 有三档：off / gemm / mlp / both；
PyTorch 侧是 GradScaler + autocast(fp16)。
"""
import json
import os
import re
import subprocess

ROOT = r"D:\orv_bench"
ORV = r"D:\orv_pip_0962\Scripts\python.exe"
PYT = r"C:\Python311\python.exe"
STEPS = 3


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    return (p.stdout or "") + (p.stderr or "")


def orv(seq, amp):
    cmd = [ORV, os.path.join(ROOT, "split_orv.py"),
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--steps", str(STEPS), "--log-every", "99",
           "--no-final-save", "--amp", amp]
    out = run(cmd)
    r = re.search(r"^RESULT (\{.*\})$", out, re.M)
    if not r:
        why = "OOM" if ("显存不足" in out or "out of memory" in out.lower()) else "FAIL"
        return None, why
    d = json.loads(r.group(1))
    return {"step": d["step_s_steady"], "mem": d["pool_peak_live_gib"],
            "loss": d["loss_last"]}, None


def pyt(seq, amp):
    cmd = [PYT, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
           "--layers", "12", "--E", "4", "--vocab", "96000", "--d_ff", "3328",
           "--n_heads", "16", "--top_k", "2", "--seq", str(seq),
           "--steps", str(STEPS), "--ckpt", "0"] + (["--amp"] if amp else [])
    out = run(cmd)
    m = re.search(r"^RESULT (\{.*\})$", out, re.M)
    if not m:
        why = "OOM" if "out of memory" in out.lower() else "FAIL"
        return None, why
    d = json.loads(m.group(1))
    return {"step": d["step_s"], "mem": d["peak_mib"] / 1024.0}, None


print("口径：两侧都关梯度检查点（PyT --ckpt 0）；ORV 用 int8gpu 优化器")
print()
for seq in (512, 1024, 2048):
    print("===== seq%d =====" % seq)
    for amp in ("off", "gemm", "both"):
        d, why = orv(seq, amp)
        if d:
            print("  ORV amp=%-5s %6.3f s  %8.1f tok/s  %5.1f GiB"
                  % (amp, d["step"], seq / d["step"], d["mem"]), flush=True)
        else:
            print("  ORV amp=%-5s FAIL: %s" % (amp, why), flush=True)
    for amp in (False, True):
        d, why = pyt(seq, amp)
        tag = "amp" if amp else "fp32"
        if d:
            print("  PyT %-9s %6.3f s  %8.1f tok/s  %5.1f GiB"
                  % (tag, d["step"], seq / d["step"], d["mem"]), flush=True)
        else:
            print("  PyT %-9s FAIL: %s" % (tag, why), flush=True)
    print()
