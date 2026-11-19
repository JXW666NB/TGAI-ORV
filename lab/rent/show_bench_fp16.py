# -*- coding: utf-8 -*-
"""汇总 bench_bmm_fp16.json：分组收益 + 误差 + fp16 量化统计。"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"lab\rent\bench_bmm_fp16.json"
res = json.load(open(path, encoding="utf-8"))

hdr = ("%-13s %5s %6s %8s %8s %8s %8s %7s %7s %9s %9s %8s"
       % ("tag", "n/步", "b", "fp32", "fp16", "+cast", "win/步",
          "TF32", "TF16", "maxabs16", "relfro16", "cast_rel"))
print(hdr)
print("-" * len(hdr))
groups = {}
for r in res:
    if "error" in r or r.get("fp32_ms") is None:
        print("%-13s ERROR %s" % (r.get("tag", "?"), r.get("error", "")[:60]))
        continue
    n = r["per_step"]
    win = n * (r["fp32_ms"] - r["fp16_cast_ms"])
    g = r.get("group", "?")
    a = groups.setdefault(g, [0.0, 0.0, 0.0])
    a[0] += n * r["fp32_ms"]
    a[1] += n * r["fp16_ms"]
    a[2] += n * r["fp16_cast_ms"]
    e16 = r["err_fp16"]
    q = r["quant"]
    print("%-13s %5d %6d %8.4f %8.4f %8.4f %+8.3f %7.1f %7.1f %9.2e %9.2e %8.1e"
          % (r["tag"], n, r["b"], r["fp32_ms"], r["fp16_ms"], r["fp16_cast_ms"],
             win, r["tflops_fp32"], r["tflops_fp16"],
             e16["max_abs"], e16["rel_fro"], q["a_cast_rel_fro"]))
print("\n== 分组（ms/步）==")
t32 = t16 = t16c = 0.0
for g, (a, b, c) in sorted(groups.items()):
    t32 += a
    t16 += b
    t16c += c
    print("%-10s fp32 %7.2f  fp16 %6.2f  fp16+cast %6.2f  收益 %+6.2f"
          % (g, a, b, c, a - c))
print("%-10s fp32 %7.2f  fp16 %6.2f  fp16+cast %6.2f  收益 %+6.2f"
      % ("合计", t32, t16, t16c, t32 - t16c))

print("\n== 逐元素误差明细（fp32 基线 vs fp64 参考 / fp16 vs fp64）==")
for r in res:
    if "error" in r or r.get("fp32_ms") is None:
        continue
    e32, e16, q = r["err_fp32"], r["err_fp16"], r["quant"]
    print("%-13s fp32: maxabs %.2e relfro %.2e | fp16: maxabs %.2e relfro %.2e "
          "maxrel %.2e | A absmax %.3g  <minN %.2f%%  <minSub %.2f%%"
          % (r["tag"], e32["max_abs"], e32["rel_fro"],
             e16["max_abs"], e16["rel_fro"], e16["max_rel"],
             q["a_absmax"], 100 * q["a_frac_below_fp16_min_normal"],
             100 * q["a_frac_below_fp16_min_subnormal"]))
