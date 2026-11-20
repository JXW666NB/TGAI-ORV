# -*- coding: utf-8 -*-
"""k 的**双侧**约束：dp 的溢出上限 vs dS 的下溢下限（0.382 s 口径下的取舍）。

若把 2^k 折进 dP GEMM 的 alpha（零成本方案），那么同一个 k 同时作用在 dp 和 dS 上：
  * 上界：dp_absmax · 2^k < 65504      （dp 溢出）
  * 下界：dS_absmax · 2^k > 6.1e-5·m   （dS 落到 fp16 最小正规数以下就丢精度）
若改用"单独 scale 内核"，则只有 dS 的下界约束，上界变成 dS_absmax·2^k < 65504。
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "ds_out", "ds_dist.json"), encoding="utf-8") as f:
    D = json.load(f)
by = {}
for r in D["rows"]:
    by.setdefault(r["tag"], {})[r["step"]] = r

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.103515625e-05


def bounds(ds, dp, m=1.0):
    hi = np.floor(np.log2(FP16_MAX / dp)) if dp > 0 else np.inf
    lo = np.ceil(np.log2(FP16_MIN_NORMAL * m / ds)) if ds > 0 else -np.inf
    return lo, hi


print("== 折叠方案（2^k 同时作用在 dp 与 dS）：k 的双侧约束 ==")
print("%-16s %5s %11s %11s %6s %6s %8s" %
      ("层", "step", "dS absmax", "dp absmax", "k≥", "k≤", "宽度"))
pairs = [("ds_sdpa_g3", "dp_sdpa_g3", "global L3"),
         ("ds_csa_l0", "dp_csa_l0", "CSA L0")]
for dstag, dptag, name in pairs:
    for st in sorted(by.get(dstag, {})):
        if dptag not in by or st not in by[dptag]:
            continue
        ds = by[dstag][st]["absmax"]
        dp = by[dptag][st]["absmax"]
        lo, hi = bounds(ds, dp)
        print("%-16s %5d %11.3e %11.3e %6.0f %6.0f %8.0f"
              % (name, st, ds, dp, lo, hi, hi - lo + 1))

print("\n== 不折叠（单独 scale 内核，只受 dS 约束）：k 的双侧约束 ==")
print("%-16s %5s %11s %6s %6s %8s" % ("层", "step", "dS absmax", "k≥", "k≤", "宽度"))
for tag in ("ds_sdpa_g3", "ds_csa_l0", "ds_csa_l8", "ds_csa_l0_loc", "ds_csa_l8_loc"):
    for st in sorted(by.get(tag, {})):
        ds = by[tag][st]["absmax"]
        if ds <= 0:
            continue
        lo, hi = bounds(ds, ds)
        print("%-16s %5d %11.3e %6.0f %6.0f %8.0f"
              % (tag, st, ds, lo, hi, hi - lo + 1))

print("\n== 全局交集 ==")
los, his = [], []
for dstag, dptag, _n in pairs:
    for st in sorted(by.get(dstag, {})):
        if dptag not in by or st not in by[dptag]:
            continue
        lo, hi = bounds(by[dstag][st]["absmax"], by[dptag][st]["absmax"])
        los.append(lo)
        his.append(hi)
print("折叠方案：k ∈ [%.0f, %.0f]（宽 %.0f bit）  ⇒ k=20 的溢出余量 %.1f×，"
      "下溢余量 %.1f×"
      % (max(los), min(his), min(his) - max(los) + 1,
         2.0 ** (min(his) - 20), 2.0 ** (20 - max(los)) if max(los) < np.inf else float("inf")))
