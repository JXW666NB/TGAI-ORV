# -*- coding: utf-8 -*-
"""决定性消融：E=1（无路由/无专家分发）vs E=4（top_k=1/2），seq2048。

逻辑：
  只减专家计算量（E=4, top_k=1）之前**没有变快**，说明瓶颈不是专家 GEMM。
  那就把**整个路由/分发机制**去掉（E=1），若 "其余" 大幅下降 ⇒ 分发是元凶。
  为把"参数量变化"的影响也标出来，同时打印 params 与显存。
"""
import json
import os
import re
import subprocess

ROOT = r"D:\orv_bench"
ORV = r"D:\orv_pip_0962\Scripts\python.exe"
STEPS = 20
OPT_S = 0.0338


def run(seq, E, topk, layers=12, dff=3328, nq=16):
    cmd = [ORV, os.path.join(ROOT, "split_stages_orv.py"),
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", str(layers), "--E", str(E),
           "--n_q", str(nq), "--d_ff", str(dff), "--top_k", str(topk),
           "--csa", "4", "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
           "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    if E == 1:
        cmd.append("--no-shared")
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
    attn = sp["attn"] / float(STEPS)
    fwd = max(0.0, step - bwd - OPT_S)
    return {"step": step, "fwd": fwd, "attn": attn, "bwd": bwd,
            "rest": max(0.0, fwd - attn), "params": d["params"],
            "mem": d["pool_peak_live_gib"]}


print("===== seq2048 消融：路由/分发是否为主因 =====")
cases = [(4, 2), (4, 1), (1, 1)]
for E, tk in cases:
    d = run(2048, E, tk)
    if not d:
        print("  E=%d top_k=%d FAIL" % (E, tk))
        continue
    print("  E=%d top_k=%d  步时 %6.3f = fwd %6.3f (attn %5.3f + rest %5.3f) "
          "+ bwd %6.3f | 参数 %.0fM | %5.1f GiB"
          % (E, tk, d["step"], d["fwd"], d["attn"], d["rest"], d["bwd"],
             d["params"] / 1e6, d["mem"]), flush=True)

print()
print("===== seq512 同三组（看固定部分是否也受影响）=====")
for E, tk in cases:
    d = run(512, E, tk)
    if not d:
        print("  E=%d top_k=%d FAIL" % (E, tk))
        continue
    print("  E=%d top_k=%d  步时 %6.3f = fwd %6.3f (attn %5.3f + rest %5.3f) "
          "+ bwd %6.3f | 参数 %.0fM | %5.1f GiB"
          % (E, tk, d["step"], d["fwd"], d["attn"], d["rest"], d["bwd"],
             d["params"] / 1e6, d["mem"]), flush=True)
