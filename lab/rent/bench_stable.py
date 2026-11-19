# -*- coding: utf-8 -*-
"""稳定基准：同一配置跑 N 次（默认 3），报中位数与极差。

为什么需要它：seq2048 同一配置在不同轮次测到过 0.542 / 0.694 / 0.775 / 0.796 / 0.899 s
（接近 2× 波动），单次数字不能用来判断优化是否有效。这里的做法：
  1) 先跑一次**预热**（不计入统计），把驱动/时钟/页表拉到稳态
  2) 再跑 N 次，取每次的 step_s_steady
  3) 报 中位数 / 最小 / 最大 / 极差百分比
口径固定：20 步、ckpt 关、AMP 开、batch=1、同一份 packed 数据。
"""
import json
import os
import re
import statistics
import subprocess
import sys

PY = r"D:\orv_pip_0962\Scripts\python.exe"
ROOT = r"D:\orv_bench"
WARMUP = 1
RUNS = 3


def one(seq, layers, steps=20):
    cmd = [PY, "-m", "tgai_go.train.v3_pretrain",
           "--data", os.path.join(ROOT, "data2048_v96000"), "--vocab", "96000",
           "--seq", str(seq), "--d", "1024", "--layers", str(layers), "--E", "4",
           "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
           "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
           "--steps", str(steps), "--log-every", "99", "--no-final-save"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    out = (p.stdout or "") + (p.stderr or "")
    m = re.search(r'"step_s_steady":\s*([0-9.]+)', out)
    pk = re.search(r'"pool_peak_live_gib":\s*([0-9.]+)', out)
    if not m:
        return None
    return {"step": float(m.group(1)), "peak": float(pk.group(1)) if pk else None}


def bench(seq, layers=12):
    for i in range(WARMUP):
        r = one(seq, layers)
        if r is None:
            print("  预热失败（seq=%d）" % seq)
            return None
        print("  [warmup %d] %.3f s" % (i + 1, r["step"]), flush=True)
    vals, peaks = [], []
    for i in range(RUNS):
        r = one(seq, layers)
        if r is None:
            print("  第 %d 次失败" % (i + 1))
            continue
        vals.append(r["step"])
        peaks.append(r["peak"])
        print("  [run %d] %.3f s  %.1f GiB" % (i + 1, r["step"], r["peak"] or 0),
              flush=True)
    if not vals:
        return None
    med = statistics.median(vals)
    spread = 100.0 * (max(vals) - min(vals)) / med if med else 0.0
    return {"median": round(med, 3), "min": round(min(vals), 3),
            "max": round(max(vals), 3), "spread_pct": round(spread, 1),
            "peak_gib": peaks[0]}


if __name__ == "__main__":
    seqs = [int(x) for x in sys.argv[1:]] or [2048, 512]
    out = {}
    for s in seqs:
        print("===== seq%d =====" % s, flush=True)
        out[s] = bench(s)
    print("\n[bench] " + json.dumps(out, ensure_ascii=False))
