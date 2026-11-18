# -*- coding: utf-8 -*-
"""方法学修正：用 20 步（而非 3 步）重测两侧，并刻画 fp16mom 的发散。

为什么：PyTorch 前几步包含内核自动调优/预热，3 步测到的不是稳态。
3 步时 PyT seq512 是 0.455 s，20 步时只要 0.314 s —— 差 45%。
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


def orv(seq, opt="int8gpu", steps=STEPS):
    cmd = [ORV, os.path.join(ROOT, "split_orv.py"),
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", opt, "--amp", "gemm", "--steps", str(steps),
           "--log-every", "99", "--no-final-save"]
    out = run(cmd)
    r = re.search(r"^RESULT (\{.*\})$", out, re.M)
    m = re.search(r"^losses \[(.*)\]$", out, re.M)
    if not r:
        return None
    d = json.loads(r.group(1))
    return {"step": d["step_s_steady"], "mem": d["pool_peak_live_gib"],
            "losses": [float(x) for x in m.group(1).split(",")] if m else []}


def pyt(seq, steps=STEPS):
    cmd = [PYT, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
           "--layers", "12", "--E", "4", "--vocab", "96000", "--d_ff", "3328",
           "--n_heads", "16", "--top_k", "2", "--seq", str(seq),
           "--steps", str(steps), "--ckpt", "0", "--amp"]
    out = run(cmd)
    m = re.search(r"^RESULT (\{.*\})$", out, re.M)
    if not m:
        return None
    d = json.loads(m.group(1))
    return {"step": d["step_s"], "mem": d["peak_mib"] / 1024.0}


print("===== 20 步稳态（两侧 ckpt=0 / AMP 开 / 同数据同形状）=====")
for seq in (512, 1024, 2048):
    a = orv(seq)
    b = pyt(seq)
    if a and b:
        print("seq%-5d ORV %6.3f s (%8.1f tok/s, %5.1f GiB)   PyT %6.3f s (%8.1f tok/s, %5.1f GiB)   ORV %4.2fx"
              % (seq, a["step"], seq / a["step"], a["mem"],
                 b["step"], seq / b["step"], b["mem"], b["step"] / a["step"]),
              flush=True)
    else:
        print("seq%-5d ORV=%s PyT=%s" % (seq, bool(a), bool(b)), flush=True)

print()
print("===== fp16mom 的发散过程（seq512，前 20 步）=====")
d = orv(512, opt="fp16mom")
if d and d["losses"]:
    print("  " + " ".join("%.2f" % v for v in d["losses"]))
else:
    print("  取不到 loss 轨迹")
