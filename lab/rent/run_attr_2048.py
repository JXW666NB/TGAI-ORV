# -*- coding: utf-8 -*-
"""seq2048 的同一张归因表（2 个口径：vocab96000 现状 / vocab4096）。"""
import json
import os
import re
import subprocess
import sys

PY = r"D:\orv_pip_0962\Scripts\python.exe"
ROOT = r"D:\orv_bench"
SCRIPT = os.path.join(ROOT, "attr_vocab_event.py")
OUT = os.path.join(ROOT, "res_attr_2048.json")
STEPS = 8
SKIP = 2

MATRIX = [
    ("s2048_v96000", "data2048_v96000", 96000),
    ("s2048_v4096", "data2048_v4096", 4096),
]


def one(tag, data, vocab):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["ATTR_TAG"] = tag
    env["ATTR_OUT"] = os.path.join(ROOT, "attr_%s.json" % tag)
    env["ATTR_SKIP"] = str(SKIP)
    env["ATTR_CAST"] = "0"
    env.pop("ORV_TIE", None)
    cmd = [PY, SCRIPT,
           "--data", os.path.join(ROOT, data), "--vocab", str(vocab),
           "--seq", "2048", "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
           "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r"^\[attr\] (\{.*\})$", out, re.M)
    if not m:
        print(out[-2000:], flush=True)
        return {"err": "no [attr]"}
    r = json.loads(m.group(1))
    r["tag"] = tag
    return r


out = {}
for tag, data, vocab in MATRIX:
    print("===== attr %s (seq2048, vocab=%d) =====" % (tag, vocab), flush=True)
    r = one(tag, data, vocab)
    out[tag] = r
    if "err" in r:
        continue
    print("  wall=%.3f s measured=%d" % (r["step_s_wall_mean"], r["measured_steps"]),
          flush=True)
    for k in sorted(r["sections"]):
        print("    %-10s %8.3f ms/step" % (k, r["sections"][k]["ms_per_step"]), flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
print("[attr2048] wrote %s" % OUT, flush=True)
