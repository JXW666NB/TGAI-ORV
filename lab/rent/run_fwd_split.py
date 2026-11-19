# -*- coding: utf-8 -*-
"""ORV forward 的三路拆解：注意力 / MoE / 其余（20 步，seq 512 vs 2048）。

目的：定位"长序列 forward 涨 4 倍"到底涨在哪一路。
"""
import json
import os
import re
import subprocess

ROOT = r"D:\orv_bench"
ORV = r"D:\orv_pip_0962\Scripts\python.exe"
STEPS = 20
ORV_OPT_S = 0.0338


def run(seq, layers=12, topk=2):
    cmd = [ORV, os.path.join(ROOT, "split_stages_orv.py"),
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", str(layers), "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", str(topk), "--csa", "4",
           "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
           "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    out = (p.stdout or "") + (p.stderr or "")
    r = re.search(r"^RESULT (\{.*\})$", out, re.M)
    m = re.search(r"\[split\] (\{.*\})", out)
    if not r or not m:
        return None
    d = json.loads(r.group(1))
    sp = json.loads(m.group(1))
    step = d["step_s_steady"]
    bwd = sp["bwd"] / max(1, sp["bwd_n"])
    fwd = max(0.0, step - bwd - ORV_OPT_S)
    attn = sp["attn"] / float(STEPS)
    moe = sp["moe"] / float(STEPS)
    return {"step": step, "fwd": fwd, "bwd": bwd, "attn": attn, "moe": moe,
            "rest": max(0.0, fwd - attn - moe), "mem": d["pool_peak_live_gib"]}


print("===== ORV forward 三路拆解（MoE 12L / 20 步 / AMP）=====")
res = {}
for seq in (512, 1024, 2048):
    d = run(seq)
    res[seq] = d
    if not d:
        print("  seq%-5d FAIL" % seq)
        continue
    print("  seq%-5d 总 %6.3f = fwd %6.3f (attn %5.3f + moe %5.3f + rest %5.3f) "
          "+ bwd %6.3f + opt %5.4f   %5.1f GiB"
          % (seq, d["step"], d["fwd"], d["attn"], d["moe"], d["rest"],
             d["bwd"], ORV_OPT_S, d["mem"]), flush=True)

print()
a, b = res.get(512), res.get(2048)
if a and b:
    print("4× token（512→2048）的增长倍率：")
    for k, name in (("fwd", "forward"), ("attn", "注意力"), ("moe", "MoE"),
                    ("rest", "其余"), ("bwd", "backward")):
        print("  %-9s %.3f → %.3f s   %.1f×" % (name, a[k], b[k], b[k] / max(1e-9, a[k])))

print()
print("===== seq2048：top_k 对 forward 三路的影响（E 不变，只改每 token 激活的专家数）=====")
for tk in (1, 2):
    d = run(2048, topk=tk)
    if not d:
        print("  top_k=%d FAIL" % tk)
        continue
    print("  top_k=%d  总 %6.3f = fwd %6.3f (attn %5.3f + rest %5.3f) + bwd %6.3f   %5.1f GiB"
          % (tk, d["step"], d["fwd"], d["attn"], d["rest"], d["bwd"], d["mem"]), flush=True)