# -*- coding: utf-8 -*-
"""A1 玩具回归：各优化器 × lr 网格（CPU / numpy 后端）。

任务：W(64×32) 拟合 T = W*X + noise，X(32×256)。有唯一 float64 最小二乘最优点。
口径：无裁剪、无调度、无 wd；每组同一初始化（seed 控制）。
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

ORDS = ["AdamW", "Adafactor", "Muon", "LAMB", "Lion"]
if POL_OK:
    ORDS.append("Polaris")

STEPS = 1200
grids = L.GRID
if len(sys.argv) > 1 and sys.argv[1] == "pilot":
    ORDS = ["AdamW", "Polaris"]
    grids = {k: [1e-2] for k in ORDS}
    STEPS = 200


def fmt(v, p=6):
    return ("%." + str(p) + "g") % v if isinstance(v, (int, float)) else str(v)


recs = []
for name in ORDS:
    for lr in grids[name]:
        try:
            r = L.run_toy(T, name, lr, seed=0, steps=STEPS)
            recs.append(r)
            print("%-10s lr=%-8.0e last10=%-11s excess=%-11s dist=%-10s "
                  "st<=1e-3:%-6s t=%.4fs fin=%s"
                  % (name, lr, fmt(r["last10_mean"], 7),
                     fmt(r["excess_last10"], 4), fmt(r["rel_dist_to_Wstar"], 4),
                     fmt(r["steps_to_excess_1e-3"], 6), r["t_step_s"] or -1,
                     r["all_finite"]), flush=True)
        except Exception as e:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            recs.append({"task": "toy", "opt": name, "lr": lr, "error":
                         "%s: %s" % (type(e).__name__, e)})
            print("%-10s lr=%-8.0e ERROR %s: %s" % (name, lr, type(e).__name__, e),
                  flush=True)

best = L.table(recs, "last10_mean")
summary = {}
for k, r in best.items():
    summary[k] = {"lr": r["lr"], "last10_mean": r["last10_mean"], "min": r["min"],
                  "final": r["final"], "rel_dist": r["rel_dist_to_Wstar"],
                  "excess_last10": r["excess_last10"],
                  "steps_to_1e-3": r["steps_to_excess_1e-3"],
                  "steps_to_1e-5": r["steps_to_excess_1e-5"],
                  "t_step_s": r["t_step_s"], "t_total_s": r["t_total_s"]}
print("\n[best-by-last10]")
for k, v in summary.items():
    print("  %-10s lr=%-8.0e last10=%-11s excess=%-11s dist=%-9s st<=1e-3=%s"
          % (k, v["lr"], fmt(v["last10_mean"], 7), fmt(v["excess_last10"], 4),
             fmt(v["rel_dist"], 4), fmt(v["steps_to_1e-3"], 6)))

L.dump("p10_cpu_toy", {"records": recs, "best": summary,
                       "loss_star_f64": L.toy_problem()[5],
                       "steps": STEPS, "ords": ORDS})
print(json.dumps(summary, ensure_ascii=True, indent=1, default=str))
