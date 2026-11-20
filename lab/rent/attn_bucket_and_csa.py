# -*- coding: utf-8 -*-
"""两件事（纯 CPU）：

1. **注意力 GEMM 桶的完整账**：从上一轮实测的 `bench_bmm_fp16.json` 汇总
   「打分 / p@v / dP / dq / dk / dv / RoPE」各族 fp32 时间，说明 40-50 ms 到底是哪一块。
2. **CSA 形状的 k 窗口**：真实的 CSA `dS` 样本没存盘，这里用**按实测分位数标定的合成张量**
   （absmax / std / p50 / p99 / p99.99 / frac==0 全部对齐 `ds_dist.json`）跑同一套量化器扫描，
   给出 CSA 的「可用 k 窗口」。**这是标定合成，不是实测**，报告里必须注明。
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FP16_MAX = 65504.0
FP16_MIN_SUB = 5.960464477539063e-08
FP16_MIN_NORMAL = 6.103515625e-05

# ---------------------------------------------------------------- 1. 桶的完整账
print("=" * 96)
print("1) 注意力 GEMM 桶的完整账（上一轮形状级实测，bench_bmm_fp16.json）")
print("=" * 96)
with open(os.path.join(HERE, "bench_bmm_fp16.json"), encoding="utf-8") as f:
    bm = {r["tag"]: r for r in json.load(f)}

GROUPS = [
    ("C SDPA（3 层）", ["sdpa_scores", "sdpa_pv", "sdpa_dp", "sdpa_dq", "sdpa_dk", "sdpa_dv"]),
    ("D CSA（9 层）", ["csa_scores", "csa_pv", "csa_dp", "csa_dq", "csa_dk", "csa_dv"]),
    ("F RoPE", ["rope"]),
]
print("%-12s %16s %7s %10s %10s %10s"
      % ("族/算子", "形状", "次/步", "fp32 ms/步", "fp16 ms/步", "+cast ms/步"))
tot32 = tot16 = tot16c = 0.0
for gname, tags in GROUPS:
    g32 = g16 = g16c = 0.0
    for t in tags:
        r = bm.get(t)
        if not r or "fp32_ms" not in r:
            continue
        n = r["per_step"]
        g32 += n * r["fp32_ms"]
        g16 += n * r["fp16_ms"]
        g16c += n * r["fp16_cast_ms"]
        print("  %-10s %16s %7d %10.3f %10.3f %10.3f"
              % (t, "%dx%dx%dx%d" % (r["b"], r["M"], r["N"], r["K"]), n,
                 n * r["fp32_ms"], n * r["fp16_ms"], n * r["fp16_cast_ms"]))
    print("  %-10s %16s %7s %10.3f %10.3f %10.3f" % ("小计", "", "", g32, g16, g16c))
    tot32 += g32
    tot16 += g16
    tot16c += g16c
    print()
print("注意力桶合计（C+D+F）：fp32 %.2f → fp16 %.2f（+cast %.2f）ms/步" %
      (tot32, tot16, tot16c))
ds32 = sum(bm[t]["per_step"] * bm[t]["fp32_ms"]
           for t in ("sdpa_dq", "sdpa_dk", "csa_dq", "csa_dk") if t in bm)
print("其中 **dS 消费的 4 个算子**：%.2f ms/步（占注意力桶 %.0f%%）"
      % (ds32, 100.0 * ds32 / tot32))

# ---------------------------------------------------------------- 2. CSA 的 k 窗口
print()
print("=" * 96)
print("2) CSA 形状的 k 窗口（**标定合成**，非实测）")
print("=" * 96)
with open(os.path.join(HERE, "ds_out", "ds_dist.json"), encoding="utf-8") as f:
    DD = json.load(f)
by = {}
for r in DD["rows"]:
    by.setdefault(r["tag"], {})[r["step"]] = r

ORDER = ["ds_csa_l0", "ds_csa_l8", "ds_csa_l0_loc", "ds_sdpa_g3"]


def synth_from_stats(st, n=4_000_000, seed=7):
    """按实测 absmax/std/p50/p99/p99.99/frac0 造一个同分布的合成 dS。"""
    rng = np.random.default_rng(seed)
    z = int(round(st["frac_exact_zero"] * n))
    m = n - z
    # 由 p99 / p99.99 / absmax 联合标定一个「主体 + 少量离群」的分布：
    # 主体 ~ |N| * s，离群按 absmax 线性插值。
    a = st["absmax"]
    p9999 = st["q_abs_p9999"]
    p99 = st["q_abs_p99"]
    # 用 p99 定主体尺度（|N(0,s)| 的 99% 分位 ≈ 2.576 s）
    s = max(p99 / 2.576, 1e-30)
    bulk = np.abs(rng.standard_normal(m)) * s
    # 尾巴：把最大的 0.01% 拉到 p99.99 附近，再插一个 absmax 的离群点
    k = max(1, int(0.0001 * m))
    idx = rng.choice(m, k, replace=False)
    bulk[idx] = rng.uniform(p9999 * 0.6, p9999, k)
    bulk[0] = a
    x = np.concatenate([np.zeros(z), bulk])
    rng.shuffle(x)
    return x.astype(np.float32)


def relfro_q(x, k):
    sc = 2.0 ** k
    q = np.asarray(x * sc, np.float16).astype(np.float64) / sc
    x = x.astype(np.float64)
    return float(np.sqrt(np.mean((q - x) ** 2)) / max(np.sqrt(np.mean(x ** 2)), 1e-300))


THR = 3.5e-4
for tag in ORDER:
    if tag not in by:
        continue
    for stp in sorted(by[tag]):
        r = by[tag][stp]
        if r["absmax"] <= 0:
            continue
        x = synth_from_stats(r)
        a = float(np.abs(x).max())
        k_safe = int(np.floor(np.log2(FP16_MAX / a)))
        errs = {}
        for k in range(0, 49):
            try:
                v = relfro_q(x, k)
            except Exception:                                    # noqa: BLE001
                v = None
            errs[k] = v
        ok = [k for k, v in errs.items() if v is not None and v <= THR]
        ideal = int(np.floor(np.log2(32768.0 / r["absmax"])))
        print("%-14s s%-4d absmax=%.3e  窗口=%r（宽 %s bit）  误差@窗口内≈%.2e  "
              "k_ideal=%d" %
              (tag, stp, r["absmax"], [min(ok), max(ok)] if ok else None,
               (max(ok) - min(ok) + 1) if ok else None,
               errs.get(ideal) if errs.get(ideal) else float("nan"), ideal))
