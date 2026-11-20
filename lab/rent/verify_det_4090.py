# -*- coding: utf-8 -*-
"""确定性 + 跨版本数值一致性：两个独立进程各跑 12 步，逐位比对 loss/gnorm。

用法：
    python verify_det_4090.py            # 只比当前包（两次独立进程）
    python verify_det_4090.py base       # 额外与已安装 1.0.8 比
"""
import json
import os
import re
import subprocess
import sys

PY = sys.executable
ROOT = "/root/bench"
STEPS = 12

ARGV = [PY, "-m", "tgai_go.train.v3_pretrain",
        "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
        "--seq", "512", "--d", "1024", "--layers", "12", "--E", "4",
        "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
        "--opt", "int8gpu", "--amp", "off", "--batch", "1",
        "--steps", str(STEPS), "--log-every", "1", "--no-final-save"]


def run(ppath, deterministic=True):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if deterministic:
        env["TG_DETERMINISTIC"] = "1"
    if ppath:
        env["PYTHONPATH"] = ppath
    p = subprocess.run(ARGV, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env, timeout=3600)
    out = (p.stdout or "") + (p.stderr or "")
    pairs = re.findall(r"^step\s+\d+/\d+\s+loss=([0-9.]+).*?gnorm=\s*([0-9.]+)",
                       out, re.M)
    if not pairs:
        return None, out.strip().splitlines()[-1][:150] if out.strip() else "no output"
    return [float(a) for a, _ in pairs], [float(b) for _, b in pairs]


def cmp(tag, a, b):
    if a is None or b is None:
        print("  %-28s 无法比较: %s / %s" % (tag, a, b))
        return
    dl = max(abs(x - y) for x, y in zip(a[0], b[0]))
    dg = max(abs(x - y) for x, y in zip(a[1], b[1]))
    same = all(x == y for x, y in zip(a[0], b[0])) and \
        all(x == y for x, y in zip(a[1], b[1]))
    print("  %-28s 逐位相同=%-5s  max|Δloss|=%.3e  max|Δgnorm|=%.3e"
          % (tag, "是" if same else "否", dl, dg))


print("=== 确定性验证（4090, seq512, %d 步, TG_DETERMINISTIC=1）===" % STEPS)
a = run("/root/patched_pkg", True)
b = run("/root/patched_pkg", True)
print("  run1 loss[0:3] = %s" % (a[0][:3] if a[0] else a[1],))
cmp("HEAD 两次独立进程", a, b)

if len(sys.argv) > 1 and sys.argv[1] == "base":
    c = run(None, True)
    cmp("HEAD vs 1.0.8（同开关）", a, c)
    d = run(None, False)
    cmp("1.0.8 关闭开关 两次", c, d)
