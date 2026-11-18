# -*- coding: utf-8 -*-
"""注意力配置 A/B + 测量稳定性（目标第 2、4 项）。

口径统一：两侧都关梯度检查点（PyT --ckpt 0）、都开 AMP
（PyT --amp，ORV --amp gemm）——这是目前两边各自最好且可比的设置。

变量：
  n_q / n_heads: 16 vs 8
  csa 压缩比:    4 vs 8（只有 ORV 暴露；PyT 参考模型未提供该参数 ⇒ 不可比，只报趋势）
稳定性：baseline 配置各重复 3 次，报中位数与极差。
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


def orv(seq, nq, csa):
    cmd = [ORV, os.path.join(ROOT, "split_orv.py"),
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", str(nq), "--d_ff", "3328", "--top_k", "2", "--csa", str(csa),
           "--opt", "int8gpu", "--amp", "gemm", "--steps", str(STEPS),
           "--log-every", "99", "--no-final-save"]
    out = run(cmd)
    r = re.search(r"^RESULT (\{.*\})$", out, re.M)
    if not r:
        why = "OOM" if ("显存不足" in out or "out of memory" in out.lower()) else "FAIL"
        return None, why
    d = json.loads(r.group(1))
    return {"step": d["step_s_steady"], "mem": d["pool_peak_live_gib"]}, None


def pyt(seq, nh):
    cmd = [PYT, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
           "--layers", "12", "--E", "4", "--vocab", "96000", "--d_ff", "3328",
           "--n_heads", str(nh), "--top_k", "2", "--seq", str(seq),
           "--steps", str(STEPS), "--ckpt", "0", "--amp"]
    out = run(cmd)
    m = re.search(r"^RESULT (\{.*\})$", out, re.M)
    if not m:
        why = "OOM" if "out of memory" in out.lower() else "FAIL"
        return None, why
    d = json.loads(m.group(1))
    return {"step": d["step_s"], "mem": d["peak_mib"] / 1024.0}, None


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


print("===== A. 注意力配置（两侧都 ckpt=0 / AMP 开）=====")
for seq in (512, 2048):
    print("--- seq%d ---" % seq)
    for nq in (16, 8):
        d, why = orv(seq, nq, 4)
        print("  ORV  n_q=%-2d csa=4  " % nq +
              ("%6.3f s  %5.1f GiB" % (d["step"], d["mem"]) if d else "FAIL:%s" % why),
              flush=True)
    for nq in (16, 8):
        d, why = orv(seq, nq, 8)
        print("  ORV  n_q=%-2d csa=8  " % nq +
              ("%6.3f s  %5.1f GiB" % (d["step"], d["mem"]) if d else "FAIL:%s" % why),
              flush=True)
    for nh in (16, 8):
        d, why = pyt(seq, nh)
        print("  PyT  heads=%-2d       " % nh +
              ("%6.3f s  %5.1f GiB" % (d["step"], d["mem"]) if d else "FAIL:%s" % why),
              flush=True)
    print()

print("===== B. 测量稳定性（baseline：ORV n_q16/csa4，PyT heads16，各 3 次）=====")
for seq in (512, 2048):
    o = [orv(seq, 16, 4)[0] for _ in range(3)]
    p = [pyt(seq, 16)[0] for _ in range(3)]
    for tag, xs in (("ORV", o), ("PyT", p)):
        xs = [x["step"] for x in xs if x]
        if len(xs) == 3:
            rng = (max(xs) - min(xs)) / med(xs) * 100
            print("  seq%-5d %s 中位数 %.3f s  极差 %.1f%%  (%s)"
                  % (seq, tag, med(xs), rng, ", ".join("%.3f" % v for v in xs)), flush=True)
        else:
            print("  seq%-5d %s 数据不全: %s" % (seq, tag, xs), flush=True)
