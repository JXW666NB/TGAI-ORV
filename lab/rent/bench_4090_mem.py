# -*- coding: utf-8 -*-
"""同口径显存对比：ORV vs PyT，两个指标都对上。

* `peak_mib` / `nvidia_peak_mib`  = nvidia-smi 整进程口径（含 CUDA context、cuBLAS workspace）
* `torch_peak_mib` / `pool_peak_live_gib` = 分配器口径（torch 的 max_memory_allocated vs ORV 的池活张量）
"""
import json
import os
import re
import statistics
import subprocess
import sys

PY = sys.executable
ROOT = "/root/bench"
PKG = "/root/patched_pkg"
STEPS = 20
CELLS = [(512, False), (1024, False), (2048, True)]


def run(argv, env_extra):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(env_extra)
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, env=env, timeout=3600)
    out = (p.stdout or "") + (p.stderr or "")
    if "RESULT" not in out:
        return None
    d = {}
    for k, pat in (("step", r'"step_s_steady":\s*([0-9.]+)|"step_s":\s*([0-9.]+)'),
                   ("nvsmi", r'"nvidia_peak_mib":\s*([0-9.]+)|"peak_mib":\s*([0-9.]+)'),
                   ("alloc", r'"pool_peak_live_gib":\s*([0-9.]+)|"torch_peak_mib":\s*([0-9.]+)')):
        m = re.search(pat, out)
        if m:
            d[k] = float(m.group(1) or m.group(2))
    return d


def orv(seq, ckpt):
    argv = [PY, "-m", "tgai_go.train.v3_pretrain",
            "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
            "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
            "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
            "--opt", "int8gpu", "--amp", "off", "--batch", "1",
            "--steps", str(STEPS), "--log-every", "99", "--no-final-save"]
    e = {"PYTHONPATH": PKG}
    if ckpt:
        e["TG_GRAD_CKPT"] = "1"
    return run(argv, e)


def pyt(seq, ckpt):
    argv = [PY, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
            "--layers", "12", "--E", "4", "--vocab", "96000", "--d_ff", "3328",
            "--n_heads", "16", "--top_k", "2", "--seq", str(seq),
            "--steps", str(STEPS), "--ckpt", "1" if ckpt else "0"]
    return run(argv, {})


print("=== 同口径显存对比（4090，%d 步，2 轮取大者）===" % STEPS)
table = {}
for seq, ckpt in CELLS:
    key = "seq%d%s" % (seq, "+ckpt" if ckpt else "")
    a, b = [], []
    for _ in range(2):
        r = orv(seq, ckpt)
        if r:
            a.append(r)
        r = pyt(seq, ckpt)
        if r:
            b.append(r)
    if not a or not b:
        print("  %-12s 数据不全 (orv=%s pyt=%s)" % (key, bool(a), bool(b)))
        continue
    ao, bo = a[-1], b[-1]
    cell = {"orv_step": round(statistics.median([x["step"] for x in a]), 4),
            "pyt_step": round(statistics.median([x["step"] for x in b]), 4),
            "orv_nvsmi_mib": int(max(x.get("nvsmi", 0) for x in a)),
            "pyt_nvsmi_mib": int(max(x.get("nvsmi", 0) for x in b)),
            "orv_alloc_gib": ao.get("alloc"),
            "pyt_alloc_mib": int(bo.get("alloc", 0)),
            }
    cell["speedup"] = round(cell["pyt_step"] / cell["orv_step"], 2)
    table[key] = cell
    print("  %-12s ORV %.4fs nvsmi %5d MiB alloc %s GiB | PyT %.4fs nvsmi %5d MiB alloc %4d MiB | %.2f×"
          % (key, cell["orv_step"], cell["orv_nvsmi_mib"], cell["orv_alloc_gib"],
             cell["pyt_step"], cell["pyt_nvsmi_mib"], cell["pyt_alloc_mib"],
             cell["speedup"]), flush=True)

print("\n[mem] " + json.dumps(table, ensure_ascii=False))
