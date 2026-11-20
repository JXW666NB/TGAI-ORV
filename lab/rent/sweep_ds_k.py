# -*- coding: utf-8 -*-
"""固定缩放 k 的扫描 + 误差归因（纯 CPU，numpy float64 参考）。

输入 = 真实训练里 dump 出来的 dS（`ds_sdpa_g3_s*.npy`，global layer3，2 个 head，
形状 (2,2048,2048)）与同一层的 k/q（(2,2048,64)）。参考一律用 float64。

产出四件事：
  1. **输入量化误差** vs k 曲线：relFro( ||Q_k(dS)-dS|| / ||dS|| )，Q = fp16 / bf16；
  2. **真实 dq/dk 的 relFro**（float64 matmul 对拍）在若干关键 k 上，校验 (1) 的代理是否可信；
  3. **误差归因**：把误差拆成「下溢（指数范围）」与「尾数（11 bit）」两部分，
     各自独立实现（mantissa-only 量化器 / underflow-only 截断），说明 0.38 是谁造成的；
  4. **k 的可行窗口**：满足 `absmax*2^k<65504` 且 relFro<=阈值 的 k 区间。

跑法：python lab/rent/sweep_ds_k.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "ds_out")
OUT = os.path.join(D, "sweep_ds_k.json")

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.103515625e-05
FP16_MIN_SUB = 5.960464477539063e-08
BF16_MIN_NORMAL = 1.1805916207174113e-38      # 2^-126
STEPS = [2, 10, 50]


def q_fp16(x):
    return np.asarray(x, np.float16).astype(np.float64)


def q_bf16(x):
    """fp32 -> bf16（RTNE）-> fp64。用位运算实现，避免依赖 ml_dtypes。"""
    a = np.asarray(x, np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    # RTNE: 加 0x7FFF + lsb
    lsb = (u >> 16) & 1
    u = u + 0x7FFF + lsb
    u = (u & 0xFFFF0000).astype(np.uint32)
    return u.view(np.float32).astype(np.float64)


def q_mant_only(x, bits):
    """只砍尾数到 `bits` 位有效位（不设指数范围下限/上限）。"""
    a = np.asarray(x, np.float64)
    m, e = np.frexp(a)                       # a = m * 2^e, |m| in [0.5,1)
    scale = float(2 ** (bits - 1))
    mq = np.rint(m * scale) / scale
    return np.ldexp(mq, e)


def q_underflow_only(x, min_normal, max_val, k):
    """只保留 fp16 的指数范围（先乘 2^k 再判），尾数不动。"""
    y = np.asarray(x, np.float64) * (2.0 ** k)
    y = np.where(np.abs(y) < min_normal, 0.0, y)
    y = np.where(np.abs(y) > max_val, np.copysign(max_val, y), y)
    return y / (2.0 ** k)


def _safe(x):
    if x is None or not np.isfinite(x):
        return None
    return float(x)


def relfro(a, b):
    num = np.sqrt(np.mean((a - b) ** 2))
    den = np.sqrt(np.mean(b ** 2))
    return _safe(num / max(den, 1e-300))


def main():
    res = {"steps": {}, "meta": {}}
    for st in STEPS:
        f = os.path.join(D, "ds_sdpa_g3_s%03d.npy" % st)
        if not os.path.exists(f):
            print("skip step %d (no file)" % st)
            continue
        dS = np.load(f).astype(np.float64)          # (2,2048,2048)
        kk = np.load(os.path.join(D, "k_sdpa_g3_s%03d.npy" % st)).astype(np.float64)
        qq = np.load(os.path.join(D, "q_sdpa_g3_s%03d.npy" % st)).astype(np.float64)
        absmax = float(np.abs(dS).max())
        k_safe = int(np.floor(np.log2(FP16_MAX / absmax))) if absmax > 0 else 99
        k_ideal = int(np.floor(np.log2(32768.0 / absmax))) if absmax > 0 else 99
        rec = {"absmax": absmax, "k_safe": k_safe, "k_ideal": k_ideal,
               "shape": list(dS.shape), "n": int(dS.size)}

        # ---- 1) 输入量化误差 vs k ----
        cur = {}
        for k in range(0, 49):
            sc = 2.0 ** k
            cur[k] = {
                "fp16": relfro(q_fp16(dS * sc) / sc, dS),
                "bf16": relfro(q_bf16(dS * sc) / sc, dS),
            }
        rec["relfro_input"] = cur
        # 不缩放（现状候选）
        rec["relfro_fp16_k0"] = cur[0]["fp16"]
        rec["relfro_bf16_k0"] = cur[0]["bf16"]

        # ---- 3) 误差归因（在 k_ideal 与 k=0 两个点）----
        rec["attrib"] = {}
        for k in (0, k_ideal):
            sc = 2.0 ** k
            mant = q_mant_only(dS, 11)
            und = q_underflow_only(dS, FP16_MIN_SUB, FP16_MAX, k)
            full = q_fp16(dS * sc) / sc
            rec["attrib"]["k=%d" % k] = {
                "relfro_full_fp16": relfro(full, dS),
                "relfro_mantissa_only_11bit": relfro(mant, dS),
                "relfro_underflow_only": relfro(und, dS),
                "frac_energy_zeroed_by_underflow": float(
                    np.sum((dS * (np.abs(dS * sc) < FP16_MIN_SUB)) ** 2) /
                    max(np.sum(dS ** 2), 1e-300)),
            }

        # ---- 2) 真实 dq/dk 的 relFro（用 1 个 head，float64 matmul）----
        W = 8   # 用前 8 个 head? 这里样本只有 2 个 head，全用
        ref_dq = np.einsum("bsk,bkd->bsd", dS, kk)
        ref_dk = np.einsum("bsk,bsd->bkd", dS, qq)
        rec["dq_dk"] = {}
        for k in sorted({0, k_ideal, k_ideal - 4, k_ideal - 8, k_safe}):
            if k < 0:
                continue
            sc = 2.0 ** k
            dSq = q_fp16(dS * sc) / sc
            dqq = np.einsum("bsk,bkd->bsd", dSq, kk)
            dkq = np.einsum("bsk,bsd->bkd", dSq, qq)
            rec["dq_dk"]["k=%d" % k] = {
                "dq_relfro": relfro(dqq, ref_dq),
                "dk_relfro": relfro(dkq, ref_dk),
            }
        # 累加误差：fp32 累加 vs fp64（CPU BLAS sgemm）
        dS32 = dS.astype(np.float32)
        k32 = kk.astype(np.float32)
        acc32 = np.einsum("bsk,bkd->bsd", dS32, k32).astype(np.float64)
        rec["accum_relfro_fp32_vs_fp64"] = relfro(acc32, ref_dq)

        # ---- 4) k 的可行窗口 ----
        thr = 3.5e-4
        ok = [k for k in range(0, 49) if cur[k]["fp16"] is not None and cur[k]["fp16"] <= thr]
        rec["k_window_relfro_le_3.5e-4"] = [min(ok), max(ok)] if ok else None
        rec["k_window_overflow_free"] = [0, k_safe]
        res["steps"][st] = rec
        print("step %-4d absmax %.4e  k_safe=%d k_ideal=%d  relFro(k0)=%.3e "
              "relFro(k_ideal)=%.3e  k窗口=%r  累加误差=%.2e"
              % (st, absmax, k_safe, k_ideal, cur[0]["fp16"], cur[min(k_ideal, 48)]["fp16"],
                 rec["k_window_relfro_le_3.5e-4"],
                 rec["accum_relfro_fp32_vs_fp64"]), flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[sweep] wrote %s" % OUT)

    # ---- 打印表 ----
    print("\n== 输入量化误差 relFro vs k（真实 dS, global layer3）==")
    ks = list(range(0, 49, 2))
    print("%-6s" % "k" + "".join("%10d" % k for k in ks))
    for st, rec in res["steps"].items():
        row = "".join("%10.2e" % rec["relfro_input"][str(k)]["fp16"] for k in ks)
        print("s%-5d%s" % (st, row))
    print("\n== 归因 ==")
    for st, rec in res["steps"].items():
        for kk2, v in rec["attrib"].items():
            print("step %-4d %-6s full=%.3e  mantissa_only=%.3e  underflow_only=%.3e  "
                  "零化能量占比=%.4f"
                  % (st, kk2, v["relfro_full_fp16"], v["relfro_mantissa_only_11bit"],
                     v["relfro_underflow_only"], v["frac_energy_zeroed_by_underflow"]))
    print("\n== 真实 dq/dk relFro ==")
    for st, rec in res["steps"].items():
        for kk2, v in rec["dq_dk"].items():
            print("step %-4d %-8s dq=%.3e  dk=%.3e"
                  % (st, kk2, v["dq_relfro"], v["dk_relfro"]))


if __name__ == "__main__":
    main()
