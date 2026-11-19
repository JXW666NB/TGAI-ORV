# -*- coding: utf-8 -*-
"""池上限 A/B（seq2048）：cap 19.1（当前自适应）vs 24 vs 26 vs 28。

背景：自适应规则是 0.6×总显存 = 19.12 GiB，但**峰值 live 是 23.71 GiB** ——
cap 低于 live ⇒ 池每步 trim、每步重新 alloc（实测 1225 次 alloc + 1333 次 free）。
这里测更高的 cap 能否把空转吃回来，并盯住是否 OOM。
"""
import json
import os
import re
import statistics
import subprocess
import sys

PY = r"D:\orv_pip_0962\Scripts\python.exe"
ROOT = r"D:\orv_bench"


def one(seq=2048, steps=20):
    cmd = [PY, "-m", "tgai_go.train.v3_pretrain",
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
           "--steps", str(steps), "--log-every", "99", "--no-final-save"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r'"step_s_steady":\s*([0-9.]+)', out)
    pk = re.search(r'"pool_peak_bytes":\s*([0-9]+)', out)
    if not m:
        oom = "显存不足" in out or "out of memory" in out.lower()
        return None, ("OOM" if oom else "|".join(out.splitlines()[-2:]))
    return {"step": float(m.group(1)),
            "peak_gib": round(int(pk.group(1)) / (1024 ** 3), 2) if pk else None}, None


def bench(cap):
    env_note = "自适应(19.1)" if cap is None else "TG_POOL_GB=%d" % cap
    os.environ.pop("TG_POOL_GB", None)
    if cap is not None:
        os.environ["TG_POOL_GB"] = str(cap)
    r, err = one()
    if err:
        print("  %-16s 预热失败: %s" % (env_note, err), flush=True)
        return None
    vals, peaks = [], []
    for _ in range(3):
        v, err = one()
        if err:
            print("  %-16s 跑失败: %s" % (env_note, err), flush=True)
            break
        vals.append(v["step"])
        peaks.append(v["peak_gib"])
    if not vals:
        return None
    med = statistics.median(vals)
    out = {"cap": env_note, "median": round(med, 3),
           "min": round(min(vals), 3), "max": round(max(vals), 3),
           "peak_pool_gib": peaks[0]}
    print("  %-16s %.3f s（%.3f–%.3f）pool峰值 %s GiB"
          % (env_note, med, min(vals), max(vals), peaks[0]), flush=True)
    return out


res = []
for cap in (None, 24, 26, 28):
    r = bench(cap)
    if r:
        res.append(r)
print()
print("[pool_ab] " + json.dumps(res, ensure_ascii=False))
