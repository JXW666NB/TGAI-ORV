# -*- coding: utf-8 -*-
"""ORV 分相 kernel 合计：把 prof_2048_new.json 里每个 leaf 的 phase 分摊按段求和。

段落 inclusive 时间来自 clean 模式（prof_2048_phases.json，只有 8 个 event/步）。
   段内空转 = 段 inclusive − 段内 kernel 合计
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "prof_2048_new.json")
    b = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "prof_2048_phases.json")
    with open(a, encoding="utf-8") as f:
        leaf = json.load(f)
    with open(b, encoding="utf-8") as f:
        clean = json.load(f)

    ph_k = {}
    ph_n = {}
    for r in leaf["ops"]:
        for k, v in r["phase"].items():
            ph_k[k] = ph_k.get(k, 0.0) + v
            ph_n[k] = ph_n.get(k, 0.0) + r["n_per_step"]
    wall_clean = clean["step_s_wall_mean"] * 1000.0
    print("ORV 分相：clean 墙钟 %.1f ms（PROF_LEAVES=0，8 event/步）" % wall_clean)
    print("%-10s %10s %10s %10s %8s" % ("阶段", "clean 段", "段内 kernel", "段内空转", "空转%"))
    tot_inc = tot_k = 0.0
    for k in ("forward", "backward", "clip", "optimizer"):
        inc = (clean["phases"].get(k) or {}).get("ms_per_step", 0.0)
        ker = ph_k.get(k, 0.0)
        tot_inc += inc
        tot_k += ker
        print("%-10s %10.2f %10.2f %10.2f %7.1f%%"
              % (k, inc, ker, inc - ker, 100.0 * (inc - ker) / inc if inc else 0.0))
    print("%-10s %10.2f %10.2f %10.2f %7.1f%%"
          % ("四段合计", tot_inc, tot_k, tot_inc - tot_k,
             100.0 * (tot_inc - tot_k) / tot_inc))
    print("%-10s %10.2f %10s %10.2f"
          % ("段外", wall_clean - tot_inc, "-", wall_clean - tot_inc))
    print("%-10s %10.2f %10.2f %10.2f"
          % ("整步", wall_clean, tot_k, wall_clean - tot_k))
    print()
    print("注：段内 kernel 来自逐 leaf event 的那次运行（18 个 event/leaf 区间），"
          "每个区间含 1 个 event 记录开销 ≈ 1 µs×(开+关)/2；6690 leaf ⇒ 约 +7 ms。")


if __name__ == "__main__":
    main()
