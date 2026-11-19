# -*- coding: utf-8 -*-
"""A2 迷你 V3 模型：各优化器 × lr 网格（CPU / numpy 后端）。

模型：vocab=4096, d=128, 2 层, 4 头, d_ff=512, seq=64（真 Transformer + 真 CE）。
数据：D:\\orv_bench\\data2048_v4096 的 token（顺序取块，每步一块）。
口径：无裁剪、无调度、无 wd；同一 seed 同一初始化、同一数据顺序。

注：ORV 0.9.97 的 CPU cross_entropy 是"逐元素 python 循环"的 bit-exact 实现，
64×4096 一次要 ~10 秒，所以本脚本用**数学等价的向量化 CE**（已校验：loss 差 8.6e-6，
梯度最大差 1.3e-8 vs 内置实现）。见 p04_ce_check。
"""
import json
import os
import sys

import numpy as np

import pbench_lib as L

L.utf8()
import tgai_go.train as T

assert T.get_device() == "cpu", "本脚本只跑 CPU，当前 device=%s" % T.get_device()

try:
    import polaris  # noqa: F401
    POL_OK = T.plugin.resolve_optimizer("Polaris") is not None
except Exception as e:
    POL_OK = False
    print("POLARIS_IMPORT_FAIL %s: %s" % (type(e).__name__, e))

ORDS = ["AdamW", "Adafactor", "Muon", "LAMB"]
if POL_OK:
    ORDS.append("Polaris")

STEPS = 200
grids = {k: list(v) for k, v in L.GRID.items()}
mode = sys.argv[1] if len(sys.argv) > 1 else ""
if mode == "pilot":
    ORDS = ["AdamW", "Muon", "Polaris"]
    grids = {"AdamW": [1e-3, 1e-2], "Muon": [1e-2], "Polaris": [1e-3, 3e-3]}
    STEPS = 120
SEQ = 64

xs, ys = L.model_blocks(SEQ, STEPS)
print("blocks=%d seq=%d" % (len(xs), SEQ), flush=True)


def fmt(v, p=6):
    return ("%." + str(p) + "g") % v if isinstance(v, (int, float)) else str(v)


recs = []
for name in ORDS:
    for lr in grids[name]:
        try:
            r = L.run_model(T, name, lr, seed=0, steps=STEPS,
                            xs=xs, ys=ys)
            recs.append(r)
            print("%-10s lr=%-8.0e l10=%-10s l50=%-10s l100=%-10s min=%-10s "
                  "ms/step=%-7.1f fin=%s%s"
                  % (name, lr, fmt(r["last10_mean"], 6),
                     fmt(r.get("loss_at_50"), 6), fmt(r.get("loss_at_100"), 6),
                     fmt(r["min"], 6), (r["t_step_s"] or 0) * 1e3, r["all_finite"],
                     (" ns_saved=%s%%" % r["opt_status"]["ns_saved_pct"])
                     if r.get("opt_status") else ""), flush=True)
        except Exception as e:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            recs.append({"task": "mini_v3", "opt": name, "lr": lr, "error":
                         "%s: %s" % (type(e).__name__, e)})
            print("%-10s lr=%-8.0e ERROR %s: %s" % (name, lr, type(e).__name__, e),
                  flush=True)

best = L.table(recs, "last10_mean")
summary = {}
for k, r in best.items():
    summary[k] = {kk: r.get(kk) for kk in
                  ("lr", "last10_mean", "min", "final", "t_step_s", "tokens_per_s",
                   "loss_at_50", "loss_at_100", "opt_status")}
print("\n[best-by-last10]")
for k, v in summary.items():
    print("  %-10s lr=%-8.0e last10=%-10s min=%-10s ms/step=%.1f"
          % (k, v["lr"], fmt(v["last10_mean"], 6), fmt(v["min"], 6),
             (v["t_step_s"] or 0) * 1e3))

L.dump("p11_cpu_model" + ("_pilot" if mode else ""),
       {"records": recs, "best": summary, "steps": STEPS, "seq": SEQ, "ords": ORDS})
print(json.dumps(summary, ensure_ascii=True, indent=1, default=str))
