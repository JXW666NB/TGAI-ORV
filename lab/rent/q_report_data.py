# -*- coding: utf-8 -*-
"""把 q_out/*.log 汇总成报告用的数字（markdown 片段写到 q_out/report_data.md）。"""
import io
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "q_out")

RUNS = [
    ("orv_16L_80", "orv_16L_80.log"),
    ("orv_16L_200", "orv_16L_200.log"),
    ("orv_12L_200", "orv_12L_200.log"),
    ("pyt16_fp16_s1234_200", "pyt_match_16L_200.log"),
    ("pyt16_fp16_s7_200", "pyt_match_16L_200_s7.log"),
    ("pyt16_fp16_s1234_80", "pyt_match_16L_80.log"),
    ("pyt16_fp16_s7_80", "pyt_match_16L_80_s7.log"),
    ("pyt16_full_s1234_80", "pyt_full_16L_80.log"),
    ("pyt12_fp16_200", "pyt_match_12L_200_fp16.log"),
    ("pyt12_fp32_200", "pyt_match_12L_200_fp32.log"),
]


def rd(p):
    raw = open(p, "rb").read()
    for e in ("utf-8", "utf-16", "gbk"):
        try:
            t = raw.decode(e)
        except Exception:
            continue
        if "losses" in t:
            return t
    raise SystemExit("decode fail: %s" % p)


def parse(p):
    t = rd(p)
    m = None
    for m in re.finditer(r"^losses\s+(\[.*\])\s*$", t, re.M):
        pass
    if m is None:
        return None
    lo = np.asarray(json.loads(m.group(1)), float)
    gn = [float(x) for x in re.findall(r"gnorm=\s*([0-9.]+)", t)]
    res = None
    for mm in re.finditer(r"^RESULT (\{.*\})\s*$", t, re.M):
        res = json.loads(mm.group(1))
    return lo, np.asarray(gn, float), res


def slope(y, lo, hi):
    x = np.arange(lo, min(hi, len(y)), dtype=float)
    yy = y[lo:min(hi, len(y))]
    if len(x) < 2:
        return float("nan")
    A = np.vstack([x, np.ones_like(x)]).T
    return float(np.linalg.lstsq(A, yy, rcond=None)[0][0])


def main():
    D = {}
    for name, fn in RUNS:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            print("[skip] %s" % fn)
            continue
        got = parse(p)
        if got is None:
            print("[skip-nolosses] %s" % fn)
            continue
        D[name] = got
        print("[ok] %-22s n=%d" % (name, len(got[0])))

    buf = io.StringIO()

    def p(s=""):
        print(s)
        buf.write(s + "\n")

    p("# 汇总数字（自动生成）\n")
    p("| run | first | last | min | min@ | last10 | last20 | slope[20:] | slope[last 40] | step_s | peak |")
    p("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, (lo, gn, res) in D.items():
        ss = (res or {}).get("step_s_mean") or (res or {}).get("step_s") or 0
        pk = (res or {}).get("peak_mib") or (res or {}).get("nvidia_peak_mib") or 0
        n = len(lo)
        p("| %s | %.4f | %.4f | %.4f | %d | %.4f | %.4f | %+.2e | %+.2e | %.3f | %d MiB |"
          % (name, lo[0], lo[-1], lo.min(), int(lo.argmin()), np.mean(lo[-10:]),
             np.mean(lo[-20:]), slope(lo, 20, n), slope(lo, max(0, n - 40), n),
             ss, pk))

    def pair(a, b, label):
        A, B = D[a][0], D[b][0]
        n = min(len(A), len(B))
        d = B[:n] - A[:n]
        p("\n## %s（Δ = B − A）" % label)
        p("n=%d  meanΔ=%+.4f  mean|Δ|=%.4f  max|Δ|=%.4f  last10Δ=%+.4f"
          % (n, d.mean(), np.abs(d).mean(), np.abs(d).max(), d[-10:].mean()))
        cuts = [0, 10, 20, 40, 60, 80, 100, 150, n]
        seg = []
        for i in range(len(cuts) - 1):
            lo, hi = cuts[i], min(cuts[i + 1], n)
            if hi > lo:
                seg.append("[%d,%d)=%+.4f" % (lo, hi, d[lo:hi].mean()))
        p("  分段均值: " + "  ".join(seg))
        p("  末 20 步: A=%.4f B=%.4f Δ=%+.4f" % (np.mean(A[n - 20:n]),
                                                np.mean(B[n - 20:n]),
                                                d[n - 20:].mean()))
        p("  corr(level)=%.5f  corr(diff)=%.5f"
          % (np.corrcoef(A[:n], B[:n])[0, 1],
             np.corrcoef(np.diff(A[:n]), np.diff(B[:n]))[0, 1]))

    pair("orv_16L_200", "pyt16_fp16_s1234_200", "16L 200 步：ORV vs PyT-autocast-fp16 (init 1234)")
    pair("orv_16L_200", "pyt16_fp16_s7_200", "16L 200 步：ORV vs PyT-autocast-fp16 (init 7)")
    pair("pyt16_fp16_s1234_200", "pyt16_fp16_s7_200", "16L 200 步：PyT 两个 init 之间（噪声包络）")
    pair("orv_16L_80", "pyt16_fp16_s1234_80", "16L 80 步：ORV vs PyT (init 1234)")
    pair("orv_16L_80", "pyt16_full_s1234_80", "16L 80 步：ORV vs PyT-full（HOPE V3 全默认）")
    pair("orv_12L_200", "pyt12_fp16_200", "12L 200 步：ORV vs PyT-autocast-fp16")
    pair("orv_12L_200", "pyt12_fp32_200", "12L 200 步：ORV vs PyT-fp32（精度消融）")
    pair("pyt12_fp16_200", "pyt12_fp32_200", "12L 200 步：PyT fp16 vs fp32（同一 init）")

    # 逐步表：16L 80 步
    p("\n## 16L / 80 步 逐步 loss（ORV vs PyT-match vs PyT-full）")
    p("| step | ORV | PyT-match | Δ | PyT-s7 | Δ | PyT-full | Δ |")
    p("|---|---|---|---|---|---|---|---|")
    A = D["orv_16L_80"][0]
    B = D["pyt16_fp16_s1234_80"][0]
    C = D["pyt16_fp16_s7_80"][0]
    E = D["pyt16_full_s1234_80"][0]
    for i in range(len(A)):
        p("| %d | %.6f | %.6f | %+.4f | %.6f | %+.4f | %.6f | %+.4f |"
          % (i, A[i], B[i], B[i] - A[i], C[i], C[i] - A[i], E[i], E[i] - A[i]))

    # gnorm
    p("\n## gnorm（裁剪前）")
    for name in ("orv_16L_200", "pyt16_fp16_s1234_200", "pyt16_fp16_s7_200",
                 "pyt16_full_s1234_80", "orv_12L_200", "pyt12_fp16_200", "pyt12_fp32_200"):
        if name in D and len(D[name][1]):
            g = D[name][1]
            p("%s: n=%d first=%.2f median=%.2f mean(last20)=%.2f max=%.1f"
              % (name, len(g), g[0], float(np.median(g)), float(np.mean(g[-20:])),
                 float(g.max())))
    g1, g2 = D["orv_16L_200"][1], D["pyt16_fp16_s1234_200"][1]
    n = min(len(g1), len(g2))
    p("corr(gnorm ORV, PyT16) = %.5f" % np.corrcoef(g1[:n], g2[:n])[0, 1])
    g3, g4 = D["orv_12L_200"][1], D["pyt12_fp16_200"][1]
    n = min(len(g3), len(g4))
    p("corr(gnorm ORV, PyT12) = %.5f" % np.corrcoef(g3[:n], g4[:n])[0, 1])

    with open(os.path.join(OUT, "report_data.md"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print("\n[saved] %s" % os.path.join(OUT, "report_data.md"))


if __name__ == "__main__":
    main()
