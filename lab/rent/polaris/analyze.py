# -*- coding: utf-8 -*-
"""本地分析：把 remote_out/*.json 汇总成 markdown 表。"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REM = os.path.join(HERE, "remote_out")


def load(name):
    p = os.path.join(REM, name if name.endswith(".json") else name + ".json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def toy_table(doc):
    recs = [r for r in doc["records"] if "error" not in r]
    ords = []
    for r in recs:
        if r["opt"] not in ords:
            ords.append(r["opt"])
    lines = []
    lines.append("| 优化器 | lr | 末10均值 | min | final | 超额(final-l*) | ||W-W*||/||W*|| | 步时 ms |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for o in ords:
        rs = sorted([r for r in recs if r["opt"] == o], key=lambda r: r["lr"])
        for r in rs:
            lines.append("| %s | %.0e | %.6g | %.6g | %.6g | %.3g | %.3g | %.3f |"
                         % (o, r["lr"], r["last10_mean"], r["min"], r["final"],
                            r["excess_last10"], r["rel_dist_to_Wstar"],
                            r["t_step_s"] * 1e3))
    return "\n".join(lines), ords


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    d = load("p10_cpu_toy")
    if d:
        t, ords = toy_table(d)
        print("## toy (CPU, %d steps)" % d["steps"])
        print(t)
        print()
        print("loss* (float64) = %.10g" % d["loss_star_f64"])
