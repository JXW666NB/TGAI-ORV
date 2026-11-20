# -*- coding: utf-8 -*-
"""用本轮实测（bench_ds_fp16.json）算 dS 这一块的三种实现口径的成本账。"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "ds_out", "bench_ds_fp16_new.json")
with open(P, encoding="utf-8") as f:
    D = json.load(f)
C = {r["tag"]: r for r in D["cases"]}
print("kscale =", D["kscale"], " ds_absmax =", D["ds_absmax"])

PAIRS = [("sdpa_dq", "sdpa_dk", "global", 3),
         ("csa_dq", "csa_dk", "CSA 全局", 9),
         ("csa_dq_loc", "csa_dk_loc", "CSA 局部窗", 9)]

# 口径
#  (1) fp32：现状
#  (2) fp16，每次调用都 cast（上一轮的口径）
#  (3) fp16，每层 cast 一次、dq/dk 共用（**推荐**）
#  (4) fp16 + 单独的 scale 内核 + cast（不折 alpha）
#  (5) fp16 GEMM only（理论上限：softmax_bwd 直接产 fp16 dS）
tot = {k: 0.0 for k in ("fp32", "cast_each", "cast_once", "scale_cast", "gemm_only")}
print("\n%-12s %4s %9s %9s %9s %9s %9s" %
      ("族", "n", "fp32", "cast每次", "cast一次", "+scale核", "仅GEMM"))
for a, b, name, n in PAIRS:
    ra, rb = C[a], C[b]
    f32 = ra["fp32_ms"] + rb["fp32_ms"]
    each = (ra["fp16_cast_ms"] + rb["fp16_cast_ms"])
    once = ra["cast_only_ms"] + ra["fp16_ms"] + rb["fp16_ms"]
    sc = (ra["fp16_scalecast_ms"] + rb["fp16_scalecast_ms"])
    gemm = ra["fp16_ms"] + rb["fp16_ms"]
    print("%-12s %4d %9.3f %9.3f %9.3f %9.3f %9.3f"
          % (name, n, n * f32, n * each, n * once, n * sc, n * gemm))
    for k, v in (("fp32", f32), ("cast_each", each), ("cast_once", once),
                 ("scale_cast", sc), ("gemm_only", gemm)):
        tot[k] += n * v
print("-" * 66)
print("%-12s %4s %9.3f %9.3f %9.3f %9.3f %9.3f"
      % ("合计 ms/步", "", tot["fp32"], tot["cast_each"], tot["cast_once"],
         tot["scale_cast"], tot["gemm_only"]))
print()
base = tot["fp32"]
for k in ("cast_each", "cast_once", "scale_cast", "gemm_only"):
    d = tot[k] - base
    print("%-12s %8.3f ms/步  Δ=%+7.3f ms/步  (%.1f%% of 0.382 s)  活体×0.92=%+.2f"
          % (k, tot[k], d, 100.0 * (-d) / 382.0, d * 0.92))

print("\n== 单项成本（268 MB 的 dS）==")
for t in ("sdpa_dq", "csa_dq", "csa_dq_loc"):
    r = C[t]
    print("%-12s cast=%.4f  scale=%.4f  reduce=%.4f  fp16=%.4f  fp32=%.4f  bf16=%s"
          % (t, r["cast_only_ms"], r["scale_only_ms"], r["reduce_ms"] or -1,
             r["fp16_ms"], r["fp32_ms"],
             ("%.4f ms / relFro %.2e" % (r["bf16"]["ms"], r["bf16"]["err"]["rel_fro"]))
             if r["bf16"].get("ok") else r["bf16"].get("err")))
print("\n== 累加误差（cuBLAS fp32 累加 vs float64）==")
for t, r in C.items():
    print("  %-12s %.2e" % (t, r["err_fp32"]["rel_fro"]))
print("\n== 不缩放 fp16 的误差（合成的 dS，absmax %.2e）==" % D["ds_absmax"])
for t, r in C.items():
    print("  %-12s %.3e" % (t, r["err_fp16"]["rel_fro"]))
print("\n== 极差（3 轮中位之间，%）==")
for t, r in C.items():
    print("  %-12s fp32 %.1f%%  fp16 %.1f%%" % (t, r["fp32_spread"], r["fp16_spread"]))
