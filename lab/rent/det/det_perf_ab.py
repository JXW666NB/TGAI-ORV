# -*- coding: utf-8 -*-
"""交替 A/B（off,on,on,off）：抵消同一张卡上"不同时间段整体快慢"的漂移。

跑法：D:\\orv_bench\\remote_py.bat D:\\orv_bench\\det_perf_ab.py
"""
import json
import os
import sys

sys.path.insert(0, r"D:\orv_bench")

import bench_stable as B                                     # noqa: E402

OUT = r"D:\orv_bench\det_perf_ab.json"


def main():
    seq = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    plan = ["off", "on", "on", "off"]
    res = {"seq": seq, "order": plan, "runs": []}
    for mode in plan:
        if mode == "on":
            os.environ["TG_DETERMINISTIC"] = "1"
        else:
            os.environ.pop("TG_DETERMINISTIC", None)
        print("===== %s =====" % mode, flush=True)
        r = B.bench(seq)
        print("  -> %s" % (r,), flush=True)
        res["runs"].append({"mode": mode, "res": r})
    offs = [x["res"]["median"] for x in res["runs"] if x["mode"] == "off" and x["res"]]
    ons = [x["res"]["median"] for x in res["runs"] if x["mode"] == "on" and x["res"]]
    res["off_medians"] = offs
    res["on_medians"] = ons
    if offs and ons:
        res["off_mean"] = sum(offs) / len(offs)
        res["on_mean"] = sum(ons) / len(ons)
        res["delta_pct"] = 100.0 * (res["on_mean"] - res["off_mean"]) / res["off_mean"]
        print("\n[AB seq%d] off %s (mean %.3f)  on %s (mean %.3f)  Δ %+.1f%%"
              % (seq, [round(x, 3) for x in offs], res["off_mean"],
                 [round(x, 3) for x in ons], res["on_mean"], res["delta_pct"]))
    with open(OUT.replace(".json", "_%d.json" % seq), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())
