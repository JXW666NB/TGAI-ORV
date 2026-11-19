# -*- coding: utf-8 -*-
"""把「每步调用点清单」(probe) 与「逐形状微基准」(bench) 对齐成报告用的族表。"""
import json
import sys

SITES = sys.argv[1] if len(sys.argv) > 1 else r"lab\rent\bmm_sites.json"
BENCH = sys.argv[2] if len(sys.argv) > 2 else r"lab\rent\bench_bmm_fp16.json"

sites = json.load(open(SITES, encoding="utf-8"))
bench = {r["tag"]: r for r in json.load(open(BENCH, encoding="utf-8"))}
steps = max(sites["steps"])   # 计数覆盖 step 0..N-1 共 N 步
per = sites["per_step"][str(sites["steps"][-1])]

MAP = {
    "matmul_general | mmgen lead=1 M=2048 N=1024 K=1024": ("A 投影 fwd", "proj_fwd_x_wq"),
    "matmul_general | mmgen lead=1 M=2048 N=256 K=1024": ("A 投影 fwd", "proj_wkv"),
    "matmul_general | mmgen lead=1 M=512 N=1024 K=256": ("A 投影 fwd", "proj_kv_tc"),
    "matmul_general | mmgen lead=1 M=2048 N=1024 K=256": ("A 投影 bwd(da)", "proj_wo"),
    "matmul_general | mmgen lead=1 M=512 N=256 K=1024": ("A 投影 bwd(da)", "dW_kv2"),
    "sgemm_tn_strided_batched | ssbTN b=1 M=1024 N=1024 K=2048": ("B dW(TN)", "dW_main"),
    "sgemm_tn_strided_batched | ssbTN b=1 M=1024 N=256 K=2048": ("B dW(TN)", "dW_kv"),
    "sgemm_tn_strided_batched | ssbTN b=1 M=256 N=1024 K=512": ("B dW(TN)", "dW_kv2"),
    "sgemm_nt_strided_batched | ssbNT b=16 M=2048 N=2048 K=64": ("C SDPA scores", "sdpa_scores"),
    "bmm | bmm b=16 M=2048 N=2048 K=64": ("C SDPA dP", "sdpa_dp"),
    "bmm | bmm b=16 M=2048 N=64 K=2048": ("C SDPA pv+dv", "sdpa_pv"),
    "sgemm_strided_batched | ssb b=16 M=2048 N=64 K=2048": ("C SDPA dq", "sdpa_dq"),
    "sgemm_tn_strided_batched | ssbTN b=16 M=2048 N=64 K=2048": ("C SDPA dk", "sdpa_dk"),
    "matmul_general | mmgen lead=1x16 M=2048 N=512 K=64": ("D CSA scores", "csa_scores"),
    "matmul_general | mmgen lead=1x16 M=2048 N=64 K=512": ("D CSA pv", "csa_pv"),
    "sgemm_nt_strided_batched | ssbNT b=16 M=2048 N=512 K=64": ("D CSA dP", "csa_dp"),
    "sgemm_nt_strided_batched | ssbNT b=16 M=2048 N=64 K=512": ("D CSA dq", "csa_dq"),
    "sgemm_tn_strided_batched | ssbTN b=16 M=512 N=64 K=2048": ("D CSA dv", "csa_dv"),
    "sgemm_tn_strided_batched | ssbTN b=16 M=64 N=512 K=2048": ("D CSA dk", "csa_dk"),
    "matmul_general | mmgen lead=1x16 M=2048 N=32 K=64": ("E CSA 局部窗", None),
    "matmul_general | mmgen lead=1x16 M=2048 N=64 K=32": ("E CSA 局部窗", None),
    "sgemm_nt_strided_batched | ssbNT b=16 M=2048 N=32 K=64": ("E CSA 局部窗", None),
    "sgemm_nt_strided_batched | ssbNT b=16 M=2048 N=64 K=32": ("E CSA 局部窗", None),
    "sgemm_tn_strided_batched | ssbTN b=16 M=32 N=64 K=2048": ("E CSA 局部窗", None),
    "sgemm_tn_strided_batched | ssbTN b=16 M=64 N=32 K=2048": ("E CSA 局部窗", None),
    "matmul_general | mmgen lead=1x16 M=2048 N=64 K=64": ("F RoPE", "rope"),
    "matmul_general | mmgen lead=1x16 M=512 N=64 K=64": ("F RoPE", None),
    "matmul_general | mmgen lead=1x16 M=32 N=64 K=64": ("F RoPE", None),
}

fam = {}
unmapped = []
for r in sites["rows"]:
    k = "%s | %s" % (r["entry"], r["geom"])
    n = per.get(k, 0)
    if not n:
        continue
    if k not in MAP:
        unmapped.append((k, n))
        continue
    f, tag = MAP[k]
    a = fam.setdefault(f, {"n": 0, "flop": 0.0, "ms32": 0.0, "ms16": 0.0,
                           "ms16c": 0.0, "cases": []})
    a["n"] += n
    a["flop"] += r["flops_total"] / steps
    if tag and tag in bench:
        b = bench[tag]
        a["ms32"] += n * b["fp32_ms"]
        a["ms16"] += n * b["fp16_ms"]
        a["ms16c"] += n * b["fp16_cast_ms"]
        a["cases"].append((tag, n, b["fp32_ms"]))

print("%-16s %6s %10s %9s %9s %9s %9s" %
      ("族", "次/步", "GFLOP/步", "fp32 ms", "fp16 ms", "+cast ms", "收益"))
tot = {"n": 0, "flop": 0.0, "ms32": 0.0, "ms16": 0.0, "ms16c": 0.0}
for f in sorted(fam):
    a = fam[f]
    print("%-16s %6d %10.1f %9.2f %9.2f %9.2f %+9.2f"
          % (f, a["n"], a["flop"] / 1e9, a["ms32"], a["ms16"], a["ms16c"],
             a["ms32"] - a["ms16c"]))
    for k in tot:
        tot[k] += a[k]
print("%-16s %6d %10.1f %9.2f %9.2f %9.2f %+9.2f"
      % ("合计", tot["n"], tot["flop"] / 1e9, tot["ms32"], tot["ms16"],
         tot["ms16c"], tot["ms32"] - tot["ms16c"]))
if unmapped:
    print("\n未映射（占位）:")
    for k, n in unmapped:
        print("   %-70s %d" % (k, n))
print("\n== 选择性（只转 cast 便宜的算子）==")
sel = 0.0
for f in sorted(fam):
    a = fam[f]
    keep = 0.0
    for tag, n, ms in a["cases"]:
        b = bench[tag]
        # 只有 fp16+cast 比 fp32 更快的才转
        if b["fp16_cast_ms"] < b["fp32_ms"]:
            keep += n * b["fp16_cast_ms"]
        else:
            keep += n * b["fp32_ms"]
    if a["ms32"]:
        print("%-16s fp32 %7.2f -> 选择性 %7.2f  (%+7.2f)" %
              (f, a["ms32"], keep, keep - a["ms32"]))
