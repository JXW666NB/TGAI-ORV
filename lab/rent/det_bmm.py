# -*- coding: utf-8 -*-
"""确定性检查：同一个 GEMM 连发 3 次，输出是否**逐位相同**（fp32 与 fp16 各测）。

为什么必须先测：cuBLAS 的 CUBLAS_GEMM_DEFAULT 可能对「N 很小、K 很长」的形状选
split-K + atomicAdd 的 kernel —— 那样同一个输入多次运行的结果就**不是逐位可复现**的。
这会直接决定报告里能不能写「结果会变、但可复现」。

跑法（远端）：D:\\orv_pip_0962\\Scripts\\python.exe -X utf8 D:\\orv_bench\\det_bmm.py
"""
import ctypes
import gc
import json
import os
import sys

WORK = r"D:\orv_bench"
OUT = os.path.join(WORK, "det_bmm.json")
os.chdir(WORK)
sys.path.insert(0, WORK)

import numpy as np                                              # noqa: E402

import tgai_go.train as T                                       # noqa: E402

RT = T._cuda()
CUDA_OP_N, CUDA_OP_T = 0, 1
CUDA_R_16F, CUDA_R_32F = 2, 0
CUBLAS_COMPUTE_32F = 68
CUBLAS_GEMM_DEFAULT = -1


def _bind():
    cb = RT.cublas
    cb.cublasSgemmStridedBatched.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
        ctypes.c_int]
    cb.cublasSgemmStridedBatched.restype = ctypes.c_int
    cb.cublasGemmStridedBatchedEx.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_longlong,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_longlong,
        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    cb.cublasGemmStridedBatchedEx.restype = ctypes.c_int


def _args(op, b, M, N, K, A, B):
    if op == "NN":
        return (CUDA_OP_N, CUDA_OP_N, N, M, K,
                int(B.ptr), N, K * N, int(A.ptr), K, M * K, M * N, b)
    if op == "NT":
        return (CUDA_OP_T, CUDA_OP_N, N, M, K,
                int(B.ptr), K, N * K, int(A.ptr), K, M * K, M * N, b)
    return (CUDA_OP_N, CUDA_OP_T, N, M, K,
            int(B.ptr), N, K * N, int(A.ptr), M, K * M, M * N, b)


def _f32(g, C):
    a = ctypes.c_float(1.0)
    z = ctypes.c_float(0.0)
    RT._activate()
    RT._chk_blas(RT.cublas.cublasSgemmStridedBatched(
        RT.handle, g[0], g[1], g[2], g[3], g[4], ctypes.byref(a),
        ctypes.c_void_p(g[5]), g[6], g[7], ctypes.c_void_p(g[8]), g[9], g[10],
        ctypes.byref(z), ctypes.c_void_p(int(C.ptr)), g[2], g[11], g[12]),
        "ssb")


def _f16(g, C):
    a = ctypes.c_float(1.0)
    z = ctypes.c_float(0.0)
    RT._activate()
    RT._chk_blas(RT.cublas.cublasGemmStridedBatchedEx(
        RT.handle, g[0], g[1], g[2], g[3], g[4], ctypes.byref(a),
        ctypes.c_void_p(g[5]), CUDA_R_16F, g[6], g[7],
        ctypes.c_void_p(g[8]), CUDA_R_16F, g[9], g[10],
        ctypes.byref(z), ctypes.c_void_p(int(C.ptr)), CUDA_R_32F, g[2],
        g[11], g[12], CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "ssbEx")


CASES = [
    ("proj_fwd_x_wq NN b=1 2048x1024x1024", "NN", 1, 2048, 1024, 1024),
    ("dW_main      TN b=1 1024x1024x2048", "TN", 1, 1024, 1024, 2048),
    ("sdpa_scores  NT b=16 2048x2048x64", "NT", 16, 2048, 2048, 64),
    ("sdpa_dq      NN b=16 2048x64x2048", "NN", 16, 2048, 64, 2048),
    ("csa_dk       TN b=16 64x512x2048", "TN", 16, 64, 512, 2048),
    ("sdpa_pv      NN b=16 2048x64x2048", "NN", 16, 2048, 64, 2048),
]


def main():
    _bind()
    rng = np.random.default_rng(7)
    out = []
    for tag, op, b, M, N, K in CASES:
        a_shp = {"NN": (b, M, K), "NT": (b, M, K), "TN": (b, K, M)}[op]
        b_shp = {"NN": (b, K, N), "NT": (b, N, K), "TN": (b, K, N)}[op]
        A = RT.h2d(rng.standard_normal(a_shp).astype(np.float32))
        B = RT.h2d(rng.standard_normal(b_shp).astype(np.float32))
        A16 = RT.empty(a_shp, dtype=np.float16)
        B16 = RT.empty(b_shp, dtype=np.float16)
        RT.cast_f32_to_f16_into(A, A16)
        RT.cast_f32_to_f16_into(B, B16)
        g32 = _args(op, b, M, N, K, A, B)
        g16 = _args(op, b, M, N, K, A16, B16)
        r = {"tag": tag}
        for name, fn, g in (("fp32", _f32, g32), ("fp16", _f16, g16)):
            Cs = []
            for _ in range(3):
                C = RT.empty((b, M, N), dtype=np.float32)
                fn(g, C)
                Cs.append(RT.d2h(C))
            same = all(np.array_equal(Cs[0], Cs[i]) for i in (1, 2))
            # 再和「同一次运行里重算一遍」比：完全相同才算逐位可复现
            r[name] = {"bitwise_repeatable": bool(same),
                       "max_abs_diff": float(np.abs(Cs[0] - Cs[1]).max()),
                       "nonzero_frac_diff": float(np.mean(Cs[0] != Cs[1]))}
            del Cs
        out.append(r)
        print("%-38s fp32 repeatable=%s (diff %.3g, frac %.3g) | fp16 repeatable=%s "
              "(diff %.3g, frac %.3g)"
              % (tag, r["fp32"]["bitwise_repeatable"], r["fp32"]["max_abs_diff"],
                 r["fp32"]["nonzero_frac_diff"], r["fp16"]["bitwise_repeatable"],
                 r["fp16"]["max_abs_diff"], r["fp16"]["nonzero_frac_diff"]),
              flush=True)
        del A, B, A16, B16, g32, g16
        gc.collect()
        RT.sync()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[det] wrote %s" % OUT)


if __name__ == "__main__":
    main()
