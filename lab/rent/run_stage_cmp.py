# -*- coding: utf-8 -*-
"""逐阶段对比：ORV vs PyTorch 的 forward / backward / optimizer。

口径：MoE 12 层 / d=1024 / d_ff=3328 / E=4+1 / top_k=2 / vocab 96000，
两侧都关梯度检查点、都开 AMP、batch=1、20 步。
ORV 用差值法：backward = 包装 T.backward 累计；forward = 总 − backward − optimizer；
optimizer 用同一优化器单独一步的实测值（0.034 s，已单独测过）。
"""
import json
import os
import re
import subprocess

ROOT = r"D:\orv_bench"
ORV = r"D:\orv_pip_0962\Scripts\python.exe"
PYT = r"C:\Python311\python.exe"
STEPS = 20
ORV_OPT_S = 0.0338          # 之前单独实测：Int8AdamWGpu 一步（MoE 12L）


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    return (p.stdout or "") + (p.stderr or "")


def orv(seq):
    cmd = [ORV, os.path.join(ROOT, "split_stages_orv.py"),
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
           "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    out = run(cmd)
    r = re.search(r"^RESULT (\{.*\})$", out, re.M)
    m = re.search(r"\[split\] (\{.*\})", out)
    if not r or not m:
        return None
    d = json.loads(r.group(1))
    sp = json.loads(m.group(1))
    step = d["step_s_steady"]
    bwd = sp["bwd"] / max(1, sp["bwd_n"])
    attn = sp["attn"] / max(1, sp["attn_n"])
    opt = ORV_OPT_S
    fwd = max(0.0, step - bwd - opt)
    return {"step": step, "fwd": fwd, "bwd": bwd, "opt": opt, "attn": attn,
            "mem": d["pool_peak_live_gib"]}


def pyt(seq):
    cmd = [PYT, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
           "--layers", "12", "--E", "4", "--vocab", "96000", "--d_ff", "3328",
           "--n_heads", "16", "--top_k", "2", "--seq", str(seq),
           "--steps", str(STEPS), "--ckpt", "0", "--amp"]
    out = run(cmd)
    m = re.search(r"^RESULT (\{.*\})$", out, re.M)
    if not m:
        return None
    d = json.loads(m.group(1))
    if "fwd_s" not in d:
        return None
    return {"step": d["step_s"], "fwd": d["fwd_s"], "bwd": d["bwd_s"],
            "opt": d["opt_s"], "mem": d["peak_mib"] / 1024.0}


print("===== 逐阶段对比（MoE 12L，20 步，ckpt=0，AMP 开，batch=1）=====")
for seq in (512, 2048):
    a = orv(seq)
    b = pyt(seq)
    print("--- seq%d ---" % seq)
    if a:
        print("  ORV  总 %6.3f s = fwd %6.3f + bwd %6.3f + opt %6.4f   (注意力 %6.3f)  %5.1f GiB"
              % (a["step"], a["fwd"], a["bwd"], a["opt"], a["attn"], a["mem"]), flush=True)
    if b:
        print("  PyT  总 %6.3f s = fwd %6.3f + bwd %6.3f + opt %6.4f %22s  %5.1f GiB"
              % (b["step"], b["fwd"], b["bwd"], b["opt"], "", b["mem"]), flush=True)
    if a and b:
        print("  比值（PyT/ORV）：总 %.2fx  forward %.2fx  backward %.2fx  optimizer %.2fx"
              % (b["step"] / a["step"], b["fwd"] / a["fwd"],
                 b["bwd"] / a["bwd"], b["opt"] / a["opt"]), flush=True)
    print()
