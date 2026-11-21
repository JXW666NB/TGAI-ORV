# -*- coding: utf-8 -*-
"""Per-shape GEMM microbench on the ORV runtime (ASCII only).

Families (all column-major cuBLAS calls, exactly as the model issues them):
  nt: out[N,T] = X[T,K] @ W[N,K]^T      -> _sgemm(T, N, N, T, K, W, K, X, K, C, N)
  nn: out[T,N] = dy[T,K] @ W[K,N]       -> _sgemm(N, N, N, T, K, W, N, dy, K, C, N)
  tn: out[N,K] = dy[T,N]^T @ x[T,K]     -> _sgemm(N, T, K, N, T, x, K, dy, N, C, K)
Compare fp32 (cublasSgemm_v2) vs fp16-with-casts vs fp16-raw (no cast) vs cast-only.
"""
import json
import os
import time

import numpy as np

import tgai_go.train as T

rt = T._cuda()
T.set_default_device("cuda")

OP_N, OP_T = 0, 1


def b2(rows, cols):
    return rt.empty((rows, cols), dtype=np.float32)


def bench(fn, iters=20, warm=3):
    for _ in range(warm):
        fn()
    rt.sync()
    evs = [(rt.event_create(), rt.event_create()) for _ in range(iters)]
    h0 = time.perf_counter()
    for i in range(iters):
        rt.event_record(evs[i][0])
        fn()
        rt.event_record(evs[i][1])
    rt.sync()
    h1 = time.perf_counter()
    vals = sorted(rt.event_elapsed_ms(a, b) for a, b in evs)
    for a, b in evs:
        rt.event_destroy(a)
        rt.event_destroy(b)
    return vals[len(vals) // 2], (h1 - h0) * 1e6 / iters


def make(family, nfeat, tokens, kk):
    """Return (A, B, C, call_fp32, call_f16, call_raw)."""
    if family == "nt":
        A, B, C = b2(nfeat, kk), b2(tokens, kk), b2(nfeat, tokens)
        g = (OP_T, OP_N, nfeat, tokens, kk, A.ptr, kk, B.ptr, kk, C.ptr, nfeat)
    elif family == "nn":
        A, B, C = b2(kk, nfeat), b2(tokens, kk), b2(tokens, nfeat)
        g = (OP_N, OP_N, nfeat, tokens, kk, A.ptr, nfeat, B.ptr, kk, C.ptr, nfeat)
    else:  # tn
        A, B, C = b2(tokens, kk), b2(tokens, nfeat), b2(nfeat, kk)
        g = (OP_N, OP_T, kk, nfeat, tokens, A.ptr, kk, B.ptr, nfeat, C.ptr, kk)
    return A, B, C, g


def main():
    # (label, family, Nfeat, tokens, K, calls_per_step)  <- from the real trace
    CASES = [
        ("attn_o",     "nt", 1024, 512, 1024, 60),
        ("moe_gate512", "nt", 3328, 512, 1024, 36),
        ("moe_down512", "nt", 1024, 512, 3328, 36),
        ("moe_dx512",   "nn", 1024, 512, 3328, 24),
        ("moe_dx2_512", "nn", 3328, 512, 1024, 12),
        ("moe_dw512",   "tn", 3328, 512, 1024, 12),
        ("moe_gate200", "nt", 3328, 200, 1024, 40),
        ("moe_dw200",   "tn", 3328, 1024, 200, 40),
        ("head",       "nt", 96000, 512, 1024, 1),
    ]
    out = []
    tot_f32 = tot_f16 = tot_raw = tot_cast = 0.0
    for label, fam, nf, tk, kk, ncall in CASES:
        A, B, C, g = make(fam, nf, tk, kk)
        f32_ms, f32_us = bench(lambda: rt._sgemm(*g))
        f16_ms, f16_us = bench(lambda: rt._sgemm_f16(*g))
        A16 = rt.cast_f32_to_f16(A)
        B16 = rt.cast_f32_to_f16(B)
        raw_ms, raw_us = bench(lambda: rt._gemm_f16_raw(
            g[0], g[1], g[2], g[3], g[4], A16, g[6], B16, g[8], C, g[10]))
        cast_ms, cast_us = bench(lambda: (rt._cast_flat_ptr_into(A.ptr, A.n, A16),
                                          rt._cast_flat_ptr_into(B.ptr, B.n, B16)))
        gf = 2.0 * nf * tk * kk / 1e9
        row = {"label": label, "family": fam, "nfeat": nf, "tokens": tk, "k": kk,
               "calls": ncall, "gflop": gf,
               "fp32_us": f32_ms * 1e3, "f16_us": f16_ms * 1e3,
               "raw_us": raw_ms * 1e3, "cast_us": cast_ms * 1e3,
               "speed_raw": (f32_ms / raw_ms) if raw_ms else 0,
               "speed_cast": (f32_ms / f16_ms) if f16_ms else 0,
               "fp32_host_us": f32_us, "f16_host_us": f16_us}
        out.append(row)
        tot_f32 += ncall * f32_ms * 1e3
        tot_f16 += ncall * f16_ms * 1e3
        tot_raw += ncall * raw_ms * 1e3
        tot_cast += ncall * cast_ms * 1e3
        print("%-12s %s N=%-6d T=%-4d K=%-5d x%-3d | fp32 %7.1fus  f16raw %7.1fus"
              " (%.2fx)  f16+cast %7.1fus (%.2fx)  cast %5.1fus | host fp32 %5.2f"
              " f16 %5.2f us"
              % (label, fam, nf, tk, kk, ncall, f32_ms * 1e3, raw_ms * 1e3,
                 row["speed_raw"], f16_ms * 1e3, row["speed_cast"],
                 cast_ms * 1e3, f32_us, f16_us), flush=True)
        del A, B, C, A16, B16
        rt.empty_cache()
    print()
    print("per-step estimate from these shapes only (us):")
    print("  fp32 total     %9.0f us  (%.1f ms)" % (tot_f32, tot_f32 / 1e3))
    print("  f16 raw total  %9.0f us  (%.1f ms)" % (tot_raw, tot_raw / 1e3))
    print("  f16+cast total %9.0f us  (%.1f ms)" % (tot_f16, tot_f16 / 1e3))
    with open(os.environ.get("MB_OUT", "/root/mb_gemm.json"), "w") as f:
        json.dump({"rows": out, "tot_f32_us": tot_f32, "tot_raw_us": tot_raw,
                   "tot_f16_us": tot_f16}, f, indent=1)


main()
