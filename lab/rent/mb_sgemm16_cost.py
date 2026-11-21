# -*- coding: utf-8 -*-
"""Decompose _sgemm_f16 host cost (ASCII only)."""
import ctypes
import time

import numpy as np

import tgai_go.train as T

rt = T._cuda()
T.set_default_device("cuda")
OP_N, OP_T = 0, 1
pc = time.perf_counter_ns
M = N = K = 64


def buf(r, c):
    return rt.empty((r, c), dtype=np.float32)


A, B, C = buf(M, K), buf(N, K), buf(M, N)
rt.f16_cache_pin([A.ptr, B.ptr])
rt.f16_cache_enable(True)
A16 = rt.cast_f32_to_f16(A)
B16 = rt.cast_f32_to_f16(B)


def timeit(fn, iters=6000, warm=200):
    for _ in range(warm):
        fn()
    rt.sync()
    t0 = pc()
    for _ in range(iters):
        fn()
    dt = pc() - t0
    rt.sync()
    return dt / 1e3 / iters


one = ctypes.c_float(1.0)
zero = ctypes.c_float(0.0)
gx = rt.cublas.cublasGemmEx
hdl = rt.handle


def gemmex18():
    rt._chk_blas(gx(hdl, OP_N, OP_T, M, N, K,
                    ctypes.byref(one),
                    ctypes.c_uint64(int(A16.ptr)), 2, K,
                    ctypes.c_uint64(int(B16.ptr)), 2, K,
                    ctypes.byref(zero),
                    ctypes.c_uint64(int(C.ptr)), 0, N,
                    68, -1), "x")


r = {}
r["_bind_gemm_ex"] = timeit(lambda: rt._bind_gemm_ex())
r["_fp16_state"] = timeit(lambda: rt._fp16_state())
r["_cast_f16_cached_hit"] = timeit(lambda: rt._cast_f16_cached(A.ptr, int(A.n)))
r["gemmex18_reuse_alpha"] = timeit(gemmex18)
r["_gemm_f16_raw"] = timeit(lambda: rt._gemm_f16_raw(OP_N, OP_T, M, N, K,
                                                     A16, K, B16, K, C, N))
r["_sgemm_f16_pinned"] = timeit(lambda: rt._sgemm_f16(OP_N, OP_T, M, N, K,
                                                      A.ptr, K, B.ptr, K, C.ptr, N))
r["_sgemm_f32"] = timeit(lambda: rt._sgemm(OP_N, OP_T, M, N, K,
                                           A.ptr, K, B.ptr, K, C.ptr, N))
r["_cast_flat_ptr_into"] = timeit(lambda: rt._cast_flat_ptr_into(A.ptr, A.n, A16),
                                  iters=3000)
print("%-26s %8s" % ("piece", "us"))
for k, v in r.items():
    print("%-26s %8.3f" % (k, v))
print()
print("residual python in _sgemm_f16 = %.2f us"
      % (r["_sgemm_f16_pinned"] - r["_gemm_f16_raw"] - 2 * r["_cast_f16_cached_hit"]))
