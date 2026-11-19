# -*- coding: utf-8 -*-
"""深挖三项，供 (a)/(b)/(c) 判据用：

1. ORV GEMM 按精度路径拆 FLOPs / TFLOPS（fp16 cublasGemmEx vs fp32 SgemmStridedBatched）
2. ORV cast_f32_to_f16 的总字节与有效带宽（每个 cast 的 n 来自 profile 记录的 int 参数）
3. 两侧关键 op 的每步次数对照表（用于判 (a)/(c)）

用法： python lab/rent/deep_dive.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    o = load("prof_2048_new.json")
    p = load("prof_pyt_2048.json")

    print("=" * 100)
    print("1. ORV GEMM：按精度路径拆 FLOPs / 有效 TFLOPS")
    print("=" * 100)
    agg = {}
    for r in o["ops"]:
        if not r["label"].startswith("cublas"):
            continue
        aux = r["aux"]
        m, n, k = (aux[0], aux[1], aux[2]) if len(aux) >= 3 else (0, 0, 0)
        b = aux[3] if len(aux) == 4 else 1
        if min(m, n, k) <= 0:
            continue
        fl = 2.0 * m * n * k * max(1, b) * r["n_per_step"]
        a = agg.setdefault(r["label"], {"flop": 0.0, "ms": 0.0, "n": 0.0})
        a["flop"] += fl
        a["ms"] += r["ms_per_step"]
        a["n"] += r["n_per_step"]
    tot_f = sum(a["flop"] for a in agg.values())
    tot_ms = sum(a["ms"] for a in agg.values())
    print("%-30s %8s %10s %10s %9s" % ("cublas 入口", "次/步", "ms/步", "TFLOP/步", "TFLOPS"))
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["ms"]):
        print("%-30s %8.0f %10.2f %10.3f %9.1f"
              % (k, a["n"], a["ms"], a["flop"] / 1e12,
                 a["flop"] / (a["ms"] * 1e-3) / 1e12))
    print("%-30s %8.0f %10.2f %10.3f %9.1f"
          % ("合计", sum(a["n"] for a in agg.values()), tot_ms, tot_f / 1e12,
             tot_f / (tot_ms * 1e-3) / 1e12))
    print()
    print("  PyT 同数学 FLOPs（假设同构）用 94.01 ms ⇒ %.1f TFLOPS"
          % (tot_f / (94.01e-3) / 1e12))
    print("  V100：fp32 峰值 15.7 TFLOPS；fp16 TensorCore 峰值 125 TFLOPS")

    print()
    print("=" * 100)
    print("2. ORV cast_f32_to_f16：字节数与有效带宽")
    print("=" * 100)
    for r in o["ops"]:
        if r["label"] != "cast_f32_to_f16":
            continue
        n_elem = r["aux"][0] if r["aux"] else 0
        # 每个 cast 的 aux 是首个 int 参数（元素数）；f32→f16 读写 = 4+2 = 6 B/元素
        by = n_elem * 6.0 * r["n_per_step"]
        print("  n=%-9d 次/步 %5.0f  ms/步 %6.2f  字节/步 %8.1f MB  有效 %5.1f GB/s"
              % (n_elem, r["n_per_step"], r["ms_per_step"], by / 1e6,
                 by / (r["ms_per_step"] * 1e-3) / 1e9))
    print("  （V100 HBM2 实测 STREAM 约 830 GB/s）")

    print()
    print("=" * 100)
    print("3. 关键 op 次数对照（次/步）")
    print("=" * 100)
    o_pat = [("cast_f32_to_f16", "cast_f32_to_f16"), ("mem:cuMemcpyDtoHAsync",
                                                     "cuMemcpyDtoHAsync"),
             ("mem:cuMemcpyHtoDAsync", "cuMemcpyHtoDAsync"),
             ("mem:cuMemsetD32Async", "cuMemsetD32Async"),
             ("sumsq_partial", "sumsq_partial"), ("sumsq_final", "sumsq_final"),
             ("adamw_multi_m8v16", "adamw_multi_m8v16"),
             ("cublasGemmEx_fp16", "cublasGemmEx_fp16"),
             ("cublasSgemmStridedBatched", "cublasSgemmStridedBatched"),
             ("softmax_lastdim", "softmax_lastdim"),
             ("moe_scatter_pairs", "moe_scatter_pairs"),
             ("moe_topk", "moe_topk")]
    o_cnt = {}
    for r in o["ops"]:
        o_cnt[r["label"]] = o_cnt.get(r["label"], 0.0) + r["n_per_step"]
    p_cnt = {}
    for r in p["kernels"]:
        p_cnt[r["name"]] = r["count"] / float(p["prof_steps"])
    print("%-34s %10s %10s" % ("ORV leaf", "次/步", "ms/步"))
    o_ms = {}
    for r in o["ops"]:
        o_ms[r["label"]] = o_ms.get(r["label"], 0.0) + r["ms_per_step"]
    for k, _ in o_pat:
        if k in o_cnt:
            print("%-34s %10.0f %10.2f" % (k, o_cnt[k], o_ms[k]))
    print()
    print("%-34s %10s %10s" % ("PyT leaf op", "次/步", "ms/步"))
    p_ms = {r["name"]: r["us_per_step"] / 1000.0 for r in p["kernels"]}
    for k in ("aten::mm", "aten::copy_", "aten::_to_copy", "aten::mul", "aten::add_",
              "aten::nonzero", "aten::item", "aten::_local_scalar_dense",
              "aten::fill_", "aten::cat", "aten::_foreach_lerp_",
              "aten::_amp_foreach_non_finite_check_and_unscale_", "aten::as_strided",
              "aten::_efficient_attention_forward",
              "aten::_efficient_attention_backward", "aten::result_type"):
        if k in p_cnt:
            print("%-34s %10.0f %10.2f" % (k, p_cnt[k], p_ms[k]))
    print()
    print("  ORV 总 leaf 次数/步 = %.0f ；PyT 总 kernel 次数/步 = %.0f"
          % (o["n_leaf_per_step"], p["kernels_per_step"]))
    print("  ORV cuMemAlloc/Free 次数/步（主机侧）见 prof json 的 host 段；"
          "pool misses=%d、trims=%d（整个 20 步进程累计）"
          % (o["pool_stats"]["misses"], o["pool_stats"]["trims"]))


if __name__ == "__main__":
    main()
