# -*- coding: utf-8 -*-
"""渲染 sweep_ds_k.json（纯 CPU）。"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "ds_out", "sweep_ds_k.json")
with open(P, encoding="utf-8") as f:
    D = json.load(f)
steps = D["steps"]
print("steps =", list(steps))

print("\n== 1) 输入量化 relFro vs k（真实 dS，global layer3）==")
ks = list(range(0, 49, 2))
print("%-8s" % "step" + "".join("%9d" % k for k in ks))
for st, r in steps.items():
    row = ""
    for k in ks:
        v = r["relfro_input"].get(str(k), {}).get("fp16")
        row += "%9s" % ("inf" if v is None else "%.1e" % v)
    print("s%-7s%s" % (st, row))
print("\n%-8s %10s %10s %10s %10s %10s" % ("step", "k_safe", "k_ideal", "k0", "k@ideal",
                                          "k@20"))
for st, r in steps.items():
    ri = r["relfro_input"]
    print("s%-7s %10d %10d %10.2e %10.2e %10.2e"
          % (st, r["k_safe"], r["k_ideal"], ri["0"]["fp16"],
             ri[str(r["k_ideal"])]["fp16"], ri["20"]["fp16"]))

print("\n== 2) 真实 dq/dk relFro（float64 参考）==")
print("%-8s %-10s %10s %10s" % ("step", "k", "dq", "dk"))
for st, r in steps.items():
    for k, v in r["dq_dk"].items():
        print("s%-7s %-10s %10.3e %10.3e" % (st, k, v["dq_relfro"], v["dk_relfro"]))
print("\n累加误差（fp32 累加 vs fp64）：",
      {st: "%.2e" % r["accum_relfro_fp32_vs_fp64"] for st, r in steps.items()})

print("\n== 3) 误差归因 ==")
for st, r in steps.items():
    for k, v in r["attrib"].items():
        print("s%-5s %-6s full_fp16=%.3e  mantissa_only(11bit)=%.3e  "
              "underflow_only=%.3e  下溢零化能量=%.4f"
              % (st, k, v["relfro_full_fp16"], v["relfro_mantissa_only_11bit"],
                 v["relfro_underflow_only"], v["frac_energy_zeroed_by_underflow"]))

print("\n== 4) k 的可行窗口（relFro<=3.5e-4 且不溢出）==")
for st, r in steps.items():
    w = r["k_window_relfro_le_3.5e-4"]
    ks_ = r["k_window_overflow_free"]
    lo = max(w[0], ks_[0]) if w else None
    hi = min(w[1], ks_[1]) if w else None
    print("s%-5s relFro窗口=%r  溢出上限=%d  ⇒ 实际可用=[%r, %r] 宽 %r bit"
          % (st, w, ks_[1], lo, hi, (hi - lo + 1) if w else None))
