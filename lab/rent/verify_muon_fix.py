# -*- coding: utf-8 -*-
"""Muon 修复验证（最小卡时）：fast(默认) vs parity(慢) vs adamw，各 4 步。"""
import json
import os
import re
import subprocess
import sys

PY = sys.executable
ROOT = "/root/bench"
PKG = "/root/patched_pkg"
BASE = [PY, "-m", "tgai_go.train.v3_pretrain",
        "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
        "--seq", "512", "--d", "1024", "--layers", "12", "--E", "4",
        "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
        "--amp", "off", "--batch", "1", "--steps", "4",
        "--log-every", "99", "--no-final-save"]


def run(tag, opt, extra_env=None):
    argv = BASE + ["--opt", opt]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = PKG
    if extra_env:
        env.update(extra_env)
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env, timeout=1800)
    out = (p.stdout or "") + (p.stderr or "")
    s = re.search(r'"step_s_steady":\s*([0-9.]+)', out)
    pk = re.search(r'"pool_peak_live_gib":\s*([0-9.]+)', out)
    if s is None:
        tail = " / ".join(out.strip().splitlines()[-2:])[:200]
        print("  %-22s FAIL  %s" % (tag, tail))
        return None
    v = float(s.group(1))
    print("  %-22s %8.4f s   峰值 %s GiB" % (tag, v, pk.group(1) if pk else "?"))
    return v


print("=== Muon 修复验证（4090 / seq512 / 12 层 / 4 步 / amp off）===")
f = run("muon fast (新默认)", "muon")
s = run("muon parity (旧默认)", "muon", {"TG_MUON_NS_PARITY": "1"})
a = run("adamw（参照）", "adamw")

out = {"muon_fast": f, "muon_parity": s, "adamw": a}
if f and s:
    out["speedup_vs_parity"] = round(s / f, 1)
    print("\n  ⇒ 快路径比原来的 parity 路径快 **%.1f×**" % (s / f))
if f and a:
    out["muon_vs_adamw"] = round(f / a, 2)
    print("  ⇒ Muon(快) 仍是 AdamW 的 %.2f× 倍耗时" % (f / a))
print("\n[muon] " + json.dumps(out, ensure_ascii=False))
