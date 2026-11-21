# -*- coding: utf-8 -*-
"""Keep fp16 only for GEMMs whose FLOPs justify the cast bill (ASCII only).

PROBE_FLOPSMIN=<flops>  ->  rt._sgemm_f16 calls with 2*m*n*k below it are answered
                           with the fp32 cublasSgemm path instead (same pointer
                           arguments, so the result is numerically fp32-exact).
This is the "profitable subset" experiment: with the step host-bound a cast costs
~11.7 us while a small GEMM's fp16 gain may be 5 us, so gating should win.
"""
import atexit
import ctypes
import json
import os
import sys

FMIN = float(os.environ.get("PROBE_FLOPSMIN", "0"))
OUT = os.environ.get("PROBE_OUT", "/root/flops.json")

import tgai_go.train as T          # noqa: E402
from tgai_go.train import v3_pretrain as VP   # noqa: E402

rt = T._cuda()
OP_N, OP_T = 0, 1
CUBLAS_COMPUTE_32F = 68
CUBLAS_GEMM_DEFAULT = -1
stat = {"f16": 0, "fp32": 0}

_sg = rt._sgemm_f16
_sgemm = rt._sgemm
_alpha = ctypes.c_float(1.0)
_beta = ctypes.c_float(0.0)
sgv2 = rt.cublas.cublasSgemm_v2
hdl = rt.handle


def fp32_gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc):
    rt._activate()
    rt._chk_blas(sgv2(hdl, int(transa), int(transb), int(m), int(n), int(k),
                      ctypes.byref(_alpha), int(A), int(lda),
                      ctypes.byref(_beta), int(B), int(ldb),
                      int(C), int(ldc)), "cublasSgemm_v2(gate)")


def sgemm_f16(transa, transb, m, n, k, A, lda, B, ldb, C, ldc):
    if FMIN > 0 and 2.0 * m * n * k < FMIN:
        stat["fp32"] += 1
        return fp32_gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)
    stat["f16"] += 1
    return _sg(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)


if FMIN > 0:
    rt._sgemm_f16 = sgemm_f16


def dump():
    try:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({"fmin": FMIN, "stat": stat,
                       "casts": stat["f16"]}, f)
    except Exception:      # noqa: BLE001
        pass


atexit.register(dump)
raise SystemExit(VP.main(sys.argv[1:]))
