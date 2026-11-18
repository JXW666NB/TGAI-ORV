# -*- coding: utf-8 -*-
"""batch=2 的数值对拍（修完广播反向后必须验，否则只是"能跑"）。

判据：`lr=0`（权重不变）时
    loss(batch=2, 1 步)  ≈  (loss(batch=1, 第0步) + loss(batch=1, 第1步)) / 2
因为 batch=2 吃的是前两个窗口，batch=1 的前两步吃的也是这两个窗口。
"""
import json
import re
import subprocess

PY = r"D:\orv_pip_0962\Scripts\python.exe"
ROOT = r"D:\orv_bench"


def run(batch, steps):
    cmd = [PY, "-m", "tgai_go.train.v3_pretrain",
           "--data", ROOT + r"\data2048_v96000", "--vocab", "96000", "--seq", "64",
           "--d", "128", "--layers", "2", "--E", "2", "--n_q", "4", "--d_ff", "256",
           "--top_k", "2", "--csa", "4", "--opt", "int8gpu", "--batch", str(batch),
           "--lr", "0", "--steps", str(steps), "--log-every", "99", "--no-final-save"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r"^losses \[(.*)\]$", out, re.M)
    if not m:
        print("  失败输出片段：\n" + "\n".join(out.splitlines()[-6:]))
        return None
    return [float(x) for x in m.group(1).split(",")]


print("batch=1，2 步：")
l1 = run(1, 2)
print("  ", l1)
print("batch=2，1 步：")
l2 = run(2, 1)
print("  ", l2)

if l1 and l2 and len(l1) >= 2 and l2:
    expect = (l1[0] + l1[1]) / 2.0
    got = l2[0]
    rel = abs(got - expect) / max(1e-9, abs(expect))
    print()
    print("期望（两样本均值）%.8f   实测 %.8f   相对差 %.3e"
          % (expect, got, rel))
    print("判定：%s" % ("PASS" if rel < 1e-5 else "FAIL（偏差过大，归约可能不对）"))
