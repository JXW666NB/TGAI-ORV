# -*- coding: utf-8 -*-
"""快速探针：tie 关/开 的 `params` 与 tensors（用 --dry-run，不跑 step）。

回答任务第 1 问的一半：ORV 侧两种口径的**实际参数量**，以及 `cfg["tie"]` 是否真的生效。
"""
import json
import os
import re
import subprocess
import sys

PY = r"D:\orv_pip_0962\Scripts\python.exe"
ROOT = r"D:\orv_bench"
WRAP = os.path.join(ROOT, "tie_main.py")


def one(tie, vocab=96000):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if tie:
        env["ORV_TIE"] = "1"
    else:
        env.pop("ORV_TIE", None)
    cmd = [PY, WRAP,
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", str(vocab),
           "--seq", "512", "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
           "--steps", "1", "--log-every", "99", "--no-final-save", "--dry-run"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r"\[model\].*params=([0-9.]+)M.*tensors=(\d+)", out)
    n = re.search(r'"params":\s*(\d+)', out)
    line = next((l for l in out.splitlines() if l.startswith("[model]")), out[-400:])
    if not m:
        return {"err": out[-800:]}
    return {"tie": tie, "params_m": float(m.group(1)), "params": int(n.group(1)) if n else None,
            "tensors": int(m.group(2)), "line": line}


for tie in (False, True):
    r = one(tie)
    print(json.dumps(r, ensure_ascii=False), flush=True)
