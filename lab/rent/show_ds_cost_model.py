# -*- coding: utf-8 -*-
"""从上一轮的 bench_bmm_fp16.json 里取出 dS 相关 6 个形状的现状数（fp32/fp16/cast）。

用途：在 GPU 被别的任务占用时，先按「已实测的形状级数字」把方案 A/B/C 的成本账搭起来，
标好哪些格是实测、哪些是待本轮的 `bench_ds_fp16.py` 复核。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "bench_bmm_fp16.json")
with open(P, encoding="utf-8") as f:
    rows = json.load(f)
by = {r["tag"]: r for r in rows}
TAGS = ["sdpa_dq", "sdpa_dk", "csa_dq", "csa_dk", "csa_dq_loc", "csa_dk_loc"]
print("%-12s %6s %8s %8s %8s %8s %9s %9s" %
      ("tag", "n/步", "fp32 ms", "fp16 ms", "+cast", "cast", "TF32", "TF16"))
t32 = t16 = t16c = 0.0
for t in TAGS:
    r = by.get(t)
    if r is None:
        print("%-12s (not in json)" % t)
        continue
    if "fp32_ms" not in r:
        print("%-12s ERROR %r" % (t, r.get("error")))
        continue
    n = r["per_step"]
    t32 += n * r["fp32_ms"]
    t16 += n * r["fp16_ms"]
    t16c += n * r["fp16_cast_ms"]
    print("%-12s %6d %8.4f %8.4f %8.4f %8.4f %9.1f %9.1f"
          % (t, n, r["fp32_ms"], r["fp16_ms"], r["fp16_cast_ms"],
             r.get("cast_only_ms", -1), r["tflops_fp32"], r["tflops_fp16"]))
print("-" * 92)
print("%-12s %6s %8.3f %8.3f %8.3f   ms/步" % ("合计", "", t32, t16, t16c))
print("\n注意：`+cast` 含 cast 两个操作数（大张量=dS）。上一轮的 cast 是 2 次\n"
      "`cast_f32_to_f16_into` 并列，没有把 2^k 折进 alpha，也没有折进 dP 的 alpha。")
