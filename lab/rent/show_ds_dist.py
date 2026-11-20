# -*- coding: utf-8 -*-
"""把 ds_dist.json 渲染成报告用的表（step × 层）。纯 CPU。"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "ds_out", "ds_dist.json")
with open(P, encoding="utf-8") as f:
    D = json.load(f)
rows = D["rows"]
loss = D["loss"]
print("loss =", json.dumps(loss))
print("fix_csa_mask =", D.get("fix_csa_mask"), " target =", D.get("target"))

ORDER = ["ds_sdpa_g3", "ds_sdpa_g11", "ds_csa_l0", "ds_csa_l8",
         "ds_csa_l0_loc", "ds_csa_l8_loc",
         "dp_sdpa_g3", "dp_csa_l0"]
by = {}
for r in rows:
    by.setdefault(r["tag"], {})[r["step"]] = r

hdr = ("%-14s %5s %11s %11s %11s %11s %11s %11s %8s %8s %8s"
       % ("tag", "step", "absmax", "std", "p50", "p99", "p99.99",
          "max", "frac0", "f<sub", "f<nrm"))
print("\n" + hdr)
print("-" * len(hdr))
for tag in ORDER:
    if tag not in by:
        continue
    for st in sorted(by[tag]):
        r = by[tag][st]
        print("%-14s %5d %11.3e %11.3e %11.3e %11.3e %11.3e %11s %8.4f %8.4f %8.4f"
              % (tag, st, r["absmax"], r["std"], r["q_abs_p50"], r["q_abs_p99"],
                 r["q_abs_p9999"], "-", r["frac_exact_zero"],
                 r["frac_lt_fp16_min_sub"], r["frac_lt_fp16_min_normal"]))
    print()

print("\n== 有效（非结构 0）下溢比例：把因果掩码的 0 剔掉 ==")
print("%-14s %5s %10s %12s %12s" % ("tag", "step", "frac_exact0", "f<sub|有效", "f<nrm|有效"))
for tag in ORDER:
    if tag not in by:
        continue
    for st in sorted(by[tag]):
        r = by[tag][st]
        z = r["frac_exact_zero"]
        den = max(1.0 - z, 1e-9)
        print("%-14s %5d %10.4f %12.4f %12.4f"
              % (tag, st, z, max(r["frac_lt_fp16_min_sub"] - z, 0) / den,
                 max(r["frac_lt_fp16_min_normal"] - z, 0) / den))
    print()

print("\n== 缩放系数需求（k_safe: absmax*2^k<65504；k_ideal: 目标 32768）==")
print("%-14s %5s %11s %8s %8s" % ("tag", "step", "absmax", "k_safe", "k_ideal"))
for tag in ORDER:
    if tag not in by:
        continue
    for st in sorted(by[tag]):
        r = by[tag][st]
        print("%-14s %5d %11.3e %8.0f %8.0f"
              % (tag, st, r["absmax"], r["k_safe"], r["k_ideal"]))
    print()

print("\n== absmax 随 step 的倍数（相对 step2，同层）==")
base = {}
for tag in ORDER:
    if tag not in by or 2 not in by[tag]:
        continue
    b = by[tag][2]["absmax"]
    base[tag] = b
    line = "%-14s" % tag
    for st in sorted(by[tag]):
        line += "  s%-3d x%8.2f" % (st, by[tag][st]["absmax"] / b if b else float("nan"))
    print(line)
