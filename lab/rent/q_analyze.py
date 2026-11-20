# -*- coding: utf-8 -*-
"""把三份远端日志里的 `losses [...]` / 每步 `gnorm=` 抽出来做同口径对比。

用法（本地）：
    python lab/rent/q_analyze.py
输出：逐步表 + 汇总统计（末 10 步均值 / 最小 / 均值差 / 相关 / 斜率）。
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "q_out")
LOGS = {
    "orv": "orv_16L_80.log",
    "match": "pyt_match_16L_80.log",
    "match_s7": "pyt_match_16L_80_s7.log",
    "full": "pyt_full_16L_80.log",
}


def _read(path):
    """Tee-Object 在 Windows PowerShell 下写 UTF-16LE；远端转发也可能是 UTF-8/GBK。"""
    raw = open(path, "rb").read()
    for enc in ("utf-8", "utf-16", "gbk"):
        try:
            t = raw.decode(enc)
        except Exception:
            continue
        if "losses" in t:
            return t
    raise SystemExit("%s 编码无法识别（%d 字节）" % (path, len(raw)))


def parse(path):
    txt = _read(path)
    m = None
    for m in re.finditer(r"^losses\s+(\[.*\])\s*$", txt, re.M):
        pass
    if m is None:
        raise SystemExit("%s 里没有 losses 行" % path)
    losses = json.loads(m.group(1))
    gn = [float(x) for x in re.findall(r"gnorm=\s*([0-9.]+)", txt)]
    res = None
    for mm in re.finditer(r"^RESULT (\{.*\})\s*$", txt, re.M):
        res = json.loads(mm.group(1))
    return np.asarray(losses, float), np.asarray(gn, float), res


def slope(y, x0, x1):
    x = np.arange(x0, min(x1, len(y)), dtype=float)
    yy = y[x0:min(x1, len(y))]
    A = np.vstack([x, np.ones_like(x)]).T
    k, _b = np.linalg.lstsq(A, yy, rcond=None)[0]
    return float(k)


def main():
    data = {k: parse(os.path.join(OUT, v)) for k, v in LOGS.items()}
    L = {k: v[0] for k, v in data.items()}
    G = {k: v[1] for k, v in data.items()}
    R = {k: v[2] for k, v in data.items()}
    n = min(len(v) for v in L.values())
    print("parsed: %s  (n=%d)" % ({k: len(v) for k, v in L.items()}, n))

    orv, mt, m7, fu = L["orv"][:n], L["match"][:n], L["match_s7"][:n], L["full"][:n]
    d = mt - orv
    d7 = m7 - orv
    df = fu - orv

    print("\n## 逐步 loss 表（16 层 / seq512 / lr3e-4 恒定 / AdamW / AMP fp16 / B=1）")
    hdr = ("step | ORV        | PyT-match  | d(match) | PyT-s7     | d(s7)    | PyT-full   | d(full)")
    print(hdr)
    print("-" * len(hdr))
    for i in range(n):
        print("%4d | %10.6f | %10.6f | %+8.4f | %10.6f | %+8.4f | %10.6f | %+8.4f"
              % (i, orv[i], mt[i], d[i], m7[i], d7[i], fu[i], df[i]))

    def summ(name, y, ref=None):
        last10 = float(np.mean(y[-10:]))
        s = dict(first=float(y[0]), last=float(y[-1]), min=float(y.min()),
                 argmin=int(y.argmin()), last10=last10,
                 slope_20_79=slope(y, 20, 80), slope_60_79=slope(y, 60, 80))
        if ref is not None:
            s["mean_d"] = float(np.mean(y - ref))
            s["mean_abs_d"] = float(np.mean(np.abs(y - ref)))
            s["max_abs_d"] = float(np.max(np.abs(y - ref)))
            s["last10_d"] = float(np.mean((y - ref)[-10:]))
        print("%-9s first=%.4f last=%.4f min=%.4f@%d last10=%.4f "
              "slope[20:80]=%+.2e/step slope[60:80]=%+.2e/step%s"
              % (name, s["first"], s["last"], s["min"], s["argmin"], s["last10"],
                 s["slope_20_79"], s["slope_60_79"],
                 ("  mean_d=%+.4f mean|d|=%.4f max|d|=%.4f last10_d=%+.4f"
                  % (s["mean_d"], s["mean_abs_d"], s["max_abs_d"], s["last10_d"]))
                 if ref is not None else ""))
        return s

    print("\n## 汇总")
    so = summ("ORV", orv)
    sm = summ("PyT-match", mt, orv)
    s7 = summ("PyT-s7", m7, orv)
    sf = summ("PyT-full", fu, orv)

    # 噪声包络：两个 PyT init 之间的差
    print("\n## init 噪声包络（PyT 两个 init seed 之间）")
    print("mean|d|=%.4f  max|d|=%.4f  last10_d=%+.4f"
          % (float(np.mean(np.abs(m7 - mt))), float(np.max(np.abs(m7 - mt))),
             float(np.mean((m7 - mt)[-10:]))))

    # 逐步增量的相关（对拍同一批数据时，逐步升降应高度相关）
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    print("\n## 相关性")
    print("loss 水平 corr(ORV, PyT-match) = %.5f" % corr(orv, mt))
    do, dm = np.diff(orv), np.diff(mt)
    print("逐步增量 corr(dORV, dPyT)      = %.5f" % corr(do, dm))
    print("loss 水平 corr(ORV, PyT-full)  = %.5f" % corr(orv, fu))
    print("loss 水平 corr(ORV, PyT-s7)    = %.5f" % corr(orv, m7))

    # gnorm
    print("\n## gnorm（裁剪前）")
    for k in ("orv", "match", "match_s7", "full"):
        g = G[k]
        print("%-9s n=%d  first=%.2f  median=%.2f  mean(last20)=%.2f  max=%.1f"
              % (k, len(g), g[0], float(np.median(g)), float(np.mean(g[-20:])),
                 float(np.max(g))))
    if len(G["orv"]) == len(G["match"]):
        print("corr(gnorm ORV, PyT-match) = %.5f"
              % corr(G["orv"][:n], G["match"][:n]))

    print("\n## 资源 / 吞吐（各自 RESULT）")
    for k in ("orv", "match", "full"):
        r = R[k] or {}
        print("%-6s params=%s  step_s=%s  peak=%s"
              % (k, r.get("params"), r.get("step_s_mean", r.get("step_s")),
                 r.get("peak_mib") or r.get("nvidia_peak_mib")))

    np.savez(os.path.join(OUT, "curves.npz"), orv=orv, match=mt, match_s7=m7, full=fu)
    print("\n[saved] %s" % os.path.join(OUT, "curves.npz"))


if __name__ == "__main__":
    sys.exit(main())
