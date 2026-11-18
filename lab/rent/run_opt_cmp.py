# -*- coding: utf-8 -*-
"""优化器对照：`int8gpu`(m8v16) / `fp16mom` 是否相对 fp32 `adamw` 有"暗伤"。

判据（同一 seed、同一数据、20 步）：
  * 步时与显存（性能维度）
  * 逐步 loss 与 fp32 adamw 的偏差（正确性维度）—— 这才是"暗伤"的检验
PyTorch 侧只有 fp32 AdamW，所以只作速度/显存对照，不参与曲线比较
（两个框架的权重初始化不同，loss 曲线不能逐点对比）。
"""
import json
import os
import re
import subprocess

ROOT = r"D:\orv_bench"
ORV = r"D:\orv_pip_0962\Scripts\python.exe"
PYT = r"C:\Python311\python.exe"
STEPS = 20
SEQ = 512


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    return (p.stdout or "") + (p.stderr or "")


def orv(opt):
    cmd = [ORV, os.path.join(ROOT, "split_orv.py"),
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(SEQ), "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", opt, "--amp", "gemm", "--steps", str(STEPS),
           "--log-every", "99", "--no-final-save"]
    out = run(cmd)
    r = re.search(r"^RESULT (\{.*\})$", out, re.M)
    m = re.search(r"^losses \[(.*)\]$", out, re.M)
    if not r:
        return None
    d = json.loads(r.group(1))
    losses = [float(x) for x in m.group(1).split(",")] if m else []
    return {"step": d["step_s_steady"], "mem": d["pool_peak_live_gib"],
            "losses": losses}


def pyt():
    cmd = [PYT, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
           "--layers", "12", "--E", "4", "--vocab", "96000", "--d_ff", "3328",
           "--n_heads", "16", "--top_k", "2", "--seq", str(SEQ),
           "--steps", str(STEPS), "--ckpt", "0", "--amp"]
    out = run(cmd)
    m = re.search(r"^RESULT (\{.*\})$", out, re.M)
    if not m:
        return None
    d = json.loads(m.group(1))
    return {"step": d["step_s"], "mem": d["peak_mib"] / 1024.0, "losses": []}


res = {}
for opt in ("adamw", "fp16mom", "int8gpu"):
    d = orv(opt)
    res[opt] = d
    if d:
        print("ORV  --opt %-8s %6.3f s  %5.1f GiB  首/末 loss %.6f / %.6f"
              % (opt, d["step"], d["mem"],
                 d["losses"][0] if d["losses"] else float("nan"),
                 d["losses"][-1] if d["losses"] else float("nan")), flush=True)
    else:
        print("ORV  --opt %-8s FAIL" % opt, flush=True)

p = pyt()
if p:
    print("PyT  AdamW(fp32)  %6.3f s  %5.1f GiB" % (p["step"], p["mem"]))

print()
base = res.get("adamw")
if base and base["losses"]:
    for opt in ("fp16mom", "int8gpu"):
        d = res.get(opt)
        if not d or not d["losses"]:
            print("%s 与 fp32 adamw 比较：数据缺失" % opt)
            continue
        n = min(len(base["losses"]), len(d["losses"]))
        dev = [abs(base["losses"][i] - d["losses"][i]) for i in range(n)]
        rel = [dev[i] / max(1e-9, abs(base["losses"][i])) for i in range(n)]
        print("%s vs fp32 adamw：%d 步，最大绝对偏差 %.3e，最大相对偏差 %.3e"
              % (opt, n, max(dev), max(rel)))
