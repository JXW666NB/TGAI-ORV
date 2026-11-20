# -*- coding: utf-8 -*-
"""Muon NS 的 GEMM 模式 A/B：fp32(PEDANTIC，默认) vs tf32(张量核)。

同时给出精度代价：8 步的 loss / gnorm 逐步对比。
"""
import json
import os
import re
import subprocess
import sys

PY = sys.executable
ROOT = "/root/bench"
PKG = "/root/patched_pkg"
STEPS = 8

BASE = [PY, "-m", "tgai_go.train.v3_pretrain",
        "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
        "--seq", "512", "--d", "1024", "--layers", "12", "--E", "4",
        "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
        "--opt", "muon", "--amp", "off", "--batch", "1",
        "--steps", str(STEPS), "--log-every", "1", "--no-final-save"]


def run(mode):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = PKG
    env.pop("TG_MUON_NS_GEMM", None)
    if mode == "tf32":
        env["TG_MUON_NS_GEMM"] = "tf32"
    p = subprocess.run(BASE, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env, timeout=3600)
    out = (p.stdout or "") + (p.stderr or "")
    s = re.search(r'"step_s_steady":\s*([0-9.]+)', out)
    pk = re.search(r'"pool_peak_live_gib":\s*([0-9.]+)', out)
    pairs = re.findall(r"^step\s+\d+/\d+\s+loss=([0-9.]+).*?gnorm=\s*([0-9.]+)",
                       out, re.M)
    return (float(s.group(1)) if s else None,
            float(pk.group(1)) if pk else None,
            [float(a) for a, _ in pairs], [float(b) for _, b in pairs])


print("=== Muon NS GEMM 模式 A/B（4090 / seq512 / 12 层 / %d 步）===" % STEPS)
a = run("fp32")
b = run("tf32")
print("  fp32 : step=%s  peak=%s GiB" % (a[0], a[1]))
print("  tf32 : step=%s  peak=%s GiB" % (b[0], b[1]))
res = {"fp32": a[0], "tf32": b[0]}
if a[0] and b[0]:
    res["speedup"] = round(a[0] / b[0], 2)
    print("  ⇒ TF32 快 %.2f×" % (a[0] / b[0]))
if a[2] and b[2]:
    dl = max(abs(x - y) for x, y in zip(a[2], b[2]))
    dg = max(abs(x - y) for x, y in zip(a[3], b[3]))
    res["max_dloss"] = dl
    res["max_dgnorm"] = dg
    res["rel_dgnorm"] = round(dg / max(a[3]), 4) if a[3] else None
    print("  精度：max|Δloss|=%.3e  max|Δgnorm|=%.3e（相对 %.3f%%）"
          % (dl, dg, 100 * dg / max(a[3])))
print("\n[ns-gemm] " + json.dumps(res, ensure_ascii=False))
