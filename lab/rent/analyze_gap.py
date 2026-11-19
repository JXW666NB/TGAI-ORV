# -*- coding: utf-8 -*-
"""ORV vs PyTorch 同类 kernel 分组对照（读两份 profiler JSON，只做聚合，不重跑）。

输入：
  * ORV : lab/rent/prof_2048_new.json     （prof_event_kernels.py，逐 leaf 打点）
  * PyT : lab/rent/prof_pyt_2048.json     （prof_pyt.py，torch.profiler **叶子事件**）

为什么 PyT 侧必须取叶子事件：torch 的 range 归因会把同一段 kernel 时间同时记到
父 range 与子 op 上（实测全事件求和 665.9 ms vs 叶子 366.0 ms，重复 1.8×）。

分类规则在 ORV_RULES / PYT_RULES 里，**按顺序取第一个命中的正则**，规则会打印出来。

用法： python lab/rent/analyze_gap.py [orv.json] [pyt.json]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ORV_RULES = [
    ("1_GEMM",        r"^cublas"),
    ("2_ATTN",        r"^(attn_|softmax_lastdim|log_softmax|attn_mask_softmax)"),
    ("5_CE_CLIP_OPT", r"^(ce_|sumsq_|adamw_|muon_momentum|scale_multi)"),
    ("3_EW_NORM",     r"^(ewise_|rmsnorm_|rms_|silu_|situ_glu_|fused_add_ip|bcast_add4|"
                      r"add_bcast8|mul_bcast8|broadcast_to8|reduce_sum|sum_axis|rowwise_max|"
                      r"moe_sigmoid|moe_topk|fused_rope|rope_half)"),
    ("4_MOVE_MOE",    r"^(cast_|mem:|transpose2d|permute8|swap_last2|cat_|gather_|index_|"
                      r"scatter_|repeat_reduce|moe_(compact|gather|scatter|permute|pad|row_dot|"
                      r"row_scale|unpermute))"),
]
PYT_RULES = [
    ("1_GEMM",        r"^(aten::(mm|bmm|matmul|linear|addmm|einsum|mv)$)"),
    ("2_ATTN",        r"(_efficient_attention|flash|scaled_dot_product|attention)"),
    ("5_CE_CLIP_OPT", r"(_foreach_|_amp_foreach_|nll_loss|cross_entropy|_log_softmax)"),
    ("3_EW_NORM",     r"^aten::(mul|add|add_|sub|div|tanh|tanh_backward|sigmoid|"
                      r"sigmoid_backward|exp|pow|sqrt|rsqrt|sum|mean|linalg_vector_norm|"
                      r"clamp|relu|gelu)$"),
    ("4_MOVE_MOE",    r"^aten::(copy_|_to_copy|as_strided|fill_|empty|empty_strided|cat|"
                      r"view|reshape|contiguous|index|index_select|index_put|_index_put_impl_|"
                      r"gather|scatter|nonzero|stack|slice|select|t|permute|transpose|expand|"
                      r"repeat_interleave|bincount|one_hot|zeros|zeros_like|result_type)"),
]


def classify(name, rules):
    for cat, pat in rules:
        if re.search(pat, name):
            return cat
    return "9_OTHER"


def bar(frac, width=22):
    n = int(round(frac * width))
    return "#" * n + "." * (width - n)


def main():
    o_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "prof_2048_new.json")
    p_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "prof_pyt_2048.json")
    with open(o_path, encoding="utf-8") as f:
        o = json.load(f)
    with open(p_path, encoding="utf-8") as f:
        p = json.load(f)

    # ── ORV 聚合 ──────────────────────────────────────────────────────────
    o_per = {}
    for r in o["ops"]:
        a = o_per.setdefault(r["label"], {"ms": 0.0, "n": 0.0, "ph": {}})
        a["ms"] += r["ms_per_step"]
        a["n"] += r["n_per_step"]
        for k, v in r["phase"].items():
            a["ph"][k] = a["ph"].get(k, 0.0) + v
    o_cat = {}
    for lbl, a in o_per.items():
        c = classify(lbl, ORV_RULES)
        s = o_cat.setdefault(c, {"ms": 0.0, "n": 0.0, "ph": {}, "mem": []})
        s["ms"] += a["ms"]
        s["n"] += a["n"]
        for k, v in a["ph"].items():
            s["ph"][k] = s["ph"].get(k, 0.0) + v
        s["mem"].append((a["ms"], lbl))

    # ── PyT 聚合（叶子事件，µs → ms）──────────────────────────────────────
    p_per = {}
    for r in p["kernels"]:
        a = p_per.setdefault(r["name"], {"ms": 0.0, "n": 0.0})
        a["ms"] += r["us_per_step"] / 1000.0
        a["n"] += r["count"] / float(p["prof_steps"])
    p_cat = {}
    for nm, a in p_per.items():
        c = classify(nm, PYT_RULES)
        s = p_cat.setdefault(c, {"ms": 0.0, "n": 0.0, "mem": []})
        s["ms"] += a["ms"]
        s["n"] += a["n"]
        s["mem"].append((a["ms"], nm))

    o_wall = o["step_s_wall_mean"] * 1000.0
    o_leaf = o["sum_leaf_ms_per_step"]
    p_wall = p["wall_median_s"] * 1000.0
    p_cuda = p["cuda_ms_per_step"]

    print("=" * 112)
    print("A. ORV kernel 榜 top 25 —— seq2048 / B=1 / 12层 MoE(d=1024,d_ff=3328,E=4+1,top2,csa=4) / vocab96000")
    print("   AMP=gemm(int8gpu opt) / 20 步跑、前 3 步不计 ⇒ 17 步稳态 / 墙钟 %.1f ms（含打点开销 %.0f ms）"
          % (o_wall, o_wall - 490.0))
    print("   叶子(逐 kernel event)GPU 合计 %.1f ms ｜ leaf 调用 %.0f 次/步 ｜ 叶子之间空档 %.1f ms"
          % (o_leaf, o["n_leaf_per_step"], o["sum_gap_ms_per_step"]))
    print("=" * 112)
    print("%-3s %-27s %9s %7s %7s %8s %8s %8s"
          % ("#", "leaf / kernel", "ms/步", "%墙钟", "%叶子", "前向", "反向", "次/步"))
    print("-" * 112)
    for i, (lbl, a) in enumerate(sorted(o_per.items(), key=lambda kv: -kv[1]["ms"])[:25], 1):
        ph = a["ph"]
        print("%-3d %-27s %9.2f %6.1f%% %6.1f%% %8.1f %8.1f %8.0f"
              % (i, lbl[:27], a["ms"], 100 * a["ms"] / o_wall, 100 * a["ms"] / o_leaf,
                 ph.get("forward", 0.0), ph.get("backward", 0.0), a["n"]))
    print("-" * 112)
    print("    其余 %d 个 leaf 合计 %.2f ms（%.1f%% 叶子）"
          % (max(0, len(o_per) - 25),
             sum(a["ms"] for a in sorted(o_per.values(), key=lambda x: -x["ms"])[25:]),
             100 * sum(a["ms"] for a in sorted(o_per.values(), key=lambda x: -x["ms"])[25:]) / o_leaf))

    print()
    print("=" * 112)
    print("B. ORV 四段（event 分段，段内不 sync；clean 模式 PROF_LEAVES=0，墙钟 497.2 ms）")
    print("=" * 112)
    print("%-10s %10s %8s" % ("阶段", "ms/步", "%墙钟"))
    for k in ("forward", "backward", "clip", "optimizer"):
        st = o["phases"].get(k)
        if st:
            print("%-10s %10.2f %7.1f%%" % (k, st["ms_per_step"],
                                            100 * st["ms_per_step"] / o_wall))

    print()
    print("=" * 112)
    print("C. PyTorch 类别表 —— 同配置（740.769M 参数，tie_embeddings=True）/ 单次 sync 墙钟 %.1f ms"
          % p_wall)
    print("   CUDA kernel（叶子）合计 %.1f ms/步 ｜ CUDA/墙钟 = %.3f ｜ kernel 次数 %.0f/步 ｜ CPU aten 事件 %.0f/步"
          % (p_cuda, p_cuda / p_wall, p["kernels_per_step"], p["cpu_ops_per_step"]))
    print("   事件分相：fwd %.1f / bwd %.1f / opt %.1f ms（三段和 %.1f ≈ 墙钟）"
          % (p["phase_ms"]["forward"], p["phase_ms"]["backward"],
             p["phase_ms"]["optimizer"],
             p["phase_ms"]["forward"] + p["phase_ms"]["backward"] + p["phase_ms"]["optimizer"]))
    print("=" * 112)
    print("%-14s %9s %7s %8s %10s" % ("类别", "ms/步", "%CUDA", "%墙钟", "次数/步"))
    print("-" * 112)
    for c in sorted(p_cat):
        s = p_cat[c]
        print("%-14s %9.2f %6.1f%% %7.1f%% %10.0f"
              % (c, s["ms"], 100 * s["ms"] / p_cuda, 100 * s["ms"] / p_wall, s["n"]))

    print()
    print("=" * 112)
    print("D. 分组对照（同一套类别；ORV=逐 leaf event，PyT=叶子事件 self CUDA）")
    print("=" * 112)
    print("%-16s %10s %10s %12s %9s" % ("类别", "ORV ms", "PyT ms", "差(ORV−PyT)", "倍数"))
    print("-" * 112)
    rows = []
    for c in sorted(set(o_cat) | set(p_cat)):
        a = (o_cat.get(c) or {}).get("ms", 0.0)
        b = (p_cat.get(c) or {}).get("ms", 0.0)
        rows.append((c, a, b, a - b))
    for c, a, b, d in rows:
        print("%-16s %10.2f %10.2f %+12.2f %9s"
              % (c, a, b, d, ("%.2fx" % (a / b)) if b > 0.05 else "n/a"))
    print("-" * 112)
    print("%-16s %10.2f %10.2f %+12.2f"
          % ("kernel 合计", sum(r[1] for r in rows), sum(r[2] for r in rows),
             sum(r[3] for r in rows)))
    print("%-16s %10.2f %10.2f %+12.2f   ← 墙钟差（官方协议 490 vs 412 = 78 ms）"
          % ("墙钟", o_wall, p_wall, o_wall - p_wall))
    print("%-16s %10.2f %10.2f %+12.2f   ← 非 kernel（墙钟−kernel）"
          % ("非 kernel", o_wall - sum(r[1] for r in rows),
             p_wall - sum(r[2] for r in rows),
             (o_wall - sum(r[1] for r in rows)) - (p_wall - sum(r[2] for r in rows))))

    print()
    print("  按 |差| 排前三（kernel 侧）:")
    for c, a, b, d in sorted(rows, key=lambda x: -abs(x[3]))[:3]:
        print("    %-16s %+9.2f ms  （ORV %.2f vs PyT %.2f）" % (c, d, a, b))

    print()
    print("=" * 112)
    print("E. ORV 各类成员（ms 降序）")
    print("=" * 112)
    for c in sorted(o_cat):
        s = o_cat[c]
        mem = sorted(s["mem"], reverse=True)
        print("%-14s %8.2f ms %6.0f 次/步" % (c, s["ms"], s["n"]))
        for m, nm in mem[:10]:
            print("      %7.2f  %s" % (m, nm))

    print()
    print("=" * 112)
    print("F. PyT 各类成员（ms 降序）")
    print("=" * 112)
    for c in sorted(p_cat):
        s = p_cat[c]
        mem = sorted(s["mem"], reverse=True)
        print("%-14s %8.2f ms %6.0f 次/步" % (c, s["ms"], s["n"]))
        for m, nm in mem[:10]:
            print("      %7.2f  %s" % (m, nm))

    print()
    print("=" * 112)
    print("G. 映射规则（顺序敏感，先命中者胜）")
    print("=" * 112)
    for c, pat in ORV_RULES:
        print("  ORV %-14s %s" % (c, pat))
    for c, pat in PYT_RULES:
        print("  PyT %-14s %s" % (c, pat))

    print()
    print("=" * 112)
    print("H. GEMM 有效算力（FLOPs 由 ORV 实测 cuBLAS 形状 Σ2mnk·batch·次数 推出；两侧数学同构）")
    print("=" * 112)
    flop = 0.0
    n_gemm = 0.0
    for r in o["ops"]:
        if not r["label"].startswith("cublas"):
            continue
        aux = r["aux"]
        if len(aux) == 3:
            m, n, k = aux
            b = 1
        else:
            m, n, k, b = aux
        if m <= 0 or n <= 0 or k <= 0:
            continue
        flop += 2.0 * m * n * k * max(1, b) * r["n_per_step"]
        n_gemm += r["n_per_step"]
    print("  GEMM FLOPs/步 = %.3f TFLOP（%.0f 次 cuBLAS 调用/步）" % (flop / 1e12, n_gemm))
    o_gemm_ms = (o_cat.get("1_GEMM") or {}).get("ms", 0.0)
    p_gemm_ms = (p_cat.get("1_GEMM") or {}).get("ms", 0.0)
    print("  ORV GEMM %.2f ms ⇒ %.1f TFLOPS ；PyT GEMM %.2f ms ⇒ %.1f TFLOPS"
          % (o_gemm_ms, flop / (o_gemm_ms * 1e-3) / 1e12,
             p_gemm_ms, flop / (p_gemm_ms * 1e-3) / 1e12))
    print("  （V100 fp32 峰值 15.7 / fp16-TC 峰值 125 TFLOPS）")


if __name__ == "__main__":
    main()
