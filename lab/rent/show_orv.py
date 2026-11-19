# -*- coding: utf-8 -*-
"""单独看 ORV 的 kernel 榜 / 分段 / 主机侧调用（不依赖 PyT 结果）。

用法： python lab/rent/show_orv.py [prof.json]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "prof_2048_new.json")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    wall = d["step_s_wall_mean"] * 1000.0
    leaf = d["sum_leaf_ms_per_step"]
    print("file=%s  tag=%s  steps=%d  wall=%.1f ms  leaf=%.1f ms  gap=%.1f ms  leaf/step=%.0f"
          % (os.path.basename(path), d.get("tag"), d["measured_steps"], wall, leaf,
             d["sum_gap_ms_per_step"], d["n_leaf_per_step"]))
    per = {}
    for r in d["ops"]:
        a = per.setdefault(r["label"], {"ms": 0.0, "n": 0.0, "ph": {}})
        a["ms"] += r["ms_per_step"]
        a["n"] += r["n_per_step"]
        for k, v in r["phase"].items():
            a["ph"][k] = a["ph"].get(k, 0.0) + v
    tot = sum(a["ms"] for a in per.values())
    print()
    print("%-3s %-28s %9s %7s %7s %8s %8s %7s"
          % ("#", "leaf", "ms/step", "%wall", "%leaf", "fwd", "bwd", "n/step"))
    print("-" * 88)
    for i, (lbl, a) in enumerate(sorted(per.items(), key=lambda kv: -kv[1]["ms"])[:25], 1):
        ph = a["ph"]
        print("%-3d %-28s %9.2f %6.1f%% %6.1f%% %8.1f %8.1f %7.0f"
              % (i, lbl[:28], a["ms"], 100 * a["ms"] / wall, 100 * a["ms"] / tot,
                 ph.get("forward", 0.0), ph.get("backward", 0.0), a["n"]))
    print("-" * 88)
    rest = sum(a["ms"] for a in sorted(per.values(), key=lambda x: -x["ms"])[25:])
    print("    其余 %d 个 leaf 合计 %.2f ms（%.1f%% wall）"
          % (max(0, len(per) - 25), rest, 100 * rest / wall))
    print("    合计 %d 个 leaf  %.2f ms" % (len(per), tot))
    print()
    print("四段（clean phase-only 模式才有意义）：")
    for k, v in d["phases"].items():
        print("  %-10s %8.2f ms  %5.1f%%" % (k, v["ms_per_step"],
                                             100 * v["ms_per_step"] / wall))
    print()
    print("主机侧（同步/分配类，n=次/步, ms=主机耗时/步）：")
    for k, v in sorted(d["host"].items(), key=lambda kv: -kv[1]["ms_per_step"]):
        print("  %-22s %8.1f %9.2f ms   %s" % (k, v["n_per_step"], v["ms_per_step"],
                                               {kk: round(vv, 2) for kk, vv in v["phase"].items()}))
    print()
    print("池：", json.dumps(d["pool_stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
