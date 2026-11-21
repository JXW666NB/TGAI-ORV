# -*- coding: utf-8 -*-
"""Host-call cost model on the ORV runtime (ASCII only).

Times pure host submit cost (no GPU wait) for the primitives that `--amp gemm`
adds per fp16 GEMM: the ctypes GemmEx call, the cast launch, the shadow-cache
lookup, and the pool alloc/free pair.
"""
import ctypes
import os
import time

import numpy as np

import tgai_go.train as T

rt = T._cuda()
T.set_default_device("cuda")
OP_N, OP_T = 0, 1
pc = time.perf_counter_ns


def buf(r, c):
    return rt.empty((r, c), dtype=np.float32)


def timeit(fn, iters=8000, warm=200):
    for _ in range(warm):
        fn()
    rt.sync()
    t0 = pc()
    for _ in range(iters):
        fn()
    dt = pc() - t0
    rt.sync()
    return dt / 1e3 / iters          # us/call (host submit only)


M = N = K = 64
A, B, C = buf(M, K), buf(N, K), buf(M, N)

# both operands pinned -> 2 cache hits, 0 casts
rt.f16_cache_pin([A.ptr, B.ptr])
A16 = rt.cast_f32_to_f16(A)
B16 = rt.cast_f32_to_f16(B)

res = {}
res["sgemm_f32"] = timeit(lambda: rt._sgemm(OP_N, OP_T, M, N, K,
                                            A.ptr, K, B.ptr, K, C.ptr, N))
res["gemmex_f16_pinned"] = timeit(lambda: rt._sgemm_f16(OP_N, OP_T, M, N, K,
                                                        A.ptr, K, B.ptr, K, C.ptr, N))
# unpinned -> 2 casts per call
res["sgemm_f16_2cast"] = timeit(lambda: rt._sgemm_f16(OP_N, OP_T, M, N, K,
                                                      A.ptr + 0, K, B.ptr, K, C.ptr, N),
                                iters=3000)
rt.f16_cache_pin([])
res["sgemm_f16_2cast_nopin"] = timeit(lambda: rt._sgemm_f16(OP_N, OP_T, M, N, K,
                                                            A.ptr, K, B.ptr, K, C.ptr, N),
                                      iters=3000)
res["raw_gemmex"] = timeit(lambda: rt._gemm_f16_raw(OP_N, OP_T, M, N, K,
                                                    A16, K, B16, K, C, N))
res["cast_launch_small"] = timeit(lambda: rt._cast_flat_ptr_into(A.ptr, A.n, A16),
                                  iters=3000)
res["launch_only"] = timeit(lambda: rt.launch("cast_f32_to_f16", (4, 1, 1), (64, 1, 1),
                                              [(int(A.ptr), ctypes.c_uint64),
                                               (int(A16.ptr), ctypes.c_uint64),
                                               (256, ctypes.c_int)]), iters=3000)
res["activate"] = timeit(lambda: rt._activate())
tmp = buf(64, 64)
res["empty"] = timeit(lambda: rt.empty((64, 64), dtype=np.float32))
res["cast_f16_cached_hit"] = timeit(lambda: rt._cast_f16_cached(A.ptr, int(A.n)))
res["f16_cache_drop_miss"] = timeit(lambda: rt._f16_cache_drop_ptr(0xdeadbeef))

print("%-24s %8s us/call" % ("call", "host"))
for k in ("sgemm_f32", "gemmex_f16_pinned", "raw_gemmex", "sgemm_f16_2cast_nopin",
          "cast_launch_small", "launch_only", "activate", "empty",
          "cast_f16_cached_hit", "f16_cache_drop_miss"):
    print("%-24s %8.3f" % (k, res[k]))
print()
print("delta sgemm_f16(both pinned) - sgemm_f32 = %.3f us"
      % (res["gemmex_f16_pinned"] - res["sgemm_f32"]))
print("delta per cast (2cast_nopin - pinned)/2      = %.3f us"
      % ((res["sgemm_f16_2cast_nopin"] - res["gemmex_f16_pinned"]) / 2.0))
