# -*- coding: utf-8 -*-
"""探测：运行时 libcublas 是否有 `cublasGemmGroupedBatchedEx`，以及能否正确调用。

为什么先探这个：
  * 非 Ex 版 `cublasSgemmGroupedBatched` 要求所有组共享 m/n/k（MoE 各专家行数不同，得 pad）；
  * Ex 版支持逐组 m/n/k（不用 pad，更干净）。
所以先确认符号在不在，再决定走方案 A（pad）还是方案 B（Ex）。

运行（需独占 GPU）：python grouped_ex_probe.py
"""
import ctypes
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

CUDA_OP_N = 0
CUDA_OP_T = 1


def log(m):
    print(m, flush=True)


def main():
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    lib = rt.cublas
    for sym in ("cublasSgemmGroupedBatched", "cublasGemmGroupedBatchedEx",
                "cublasSgemmGroupedBatchedEx"):
        try:
            getattr(lib, sym)
            log("符号存在: %s" % sym)
        except AttributeError:
            log("符号缺失: %s" % sym)

    if not hasattr(lib, "cublasGemmGroupedBatchedEx"):
        log("⇒ 只能走方案 A（pad 到统一 M + 标量签名）")
        return

    # ---- 用 Ex 版做一次正确调用：2 组、行数不同，与逐次 gemm_nt 对照 ----
    rng = np.random.default_rng(0)
    K, N = 64, 32
    Ms = [5, 9]
    As = [rt.h2d(rng.normal(0, 0.1, (m, K)).astype(np.float32)) for m in Ms]
    Bs = [rt.h2d(rng.normal(0, 0.1, (N, K)).astype(np.float32)) for _ in Ms]
    Cs = [rt.empty((m, N), dtype=np.float32) for m in Ms]

    fn = lib.cublasGemmGroupedBatchedEx
    fn.restype = ctypes.c_int
    # Ex 签名：handle, transa, transb, const int* groupedM, groupedN, groupedK,
    #          const void* const Aarray[], const void* const Barray[], const void* const Carray[],
    #          int groupCount, int groupSize[], cudaDataType, cublasComputeType, cublasGemmAlgo_t
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    g = len(Ms)
    gm = (ctypes.c_int * g)(*Ms)
    gn = (ctypes.c_int * g)(*([N] * g))
    gk = (ctypes.c_int * g)(*([K] * g))
    # Ex 版 A/B/C 都是 /const void* 数组；转置语义由 transa/transb 决定
    aa = (ctypes.c_uint64 * g)(*[int(x.ptr) for x in As])
    ba = (ctypes.c_uint64 * g)(*[int(x.ptr) for x in Bs])
    ca = (ctypes.c_uint64 * g)(*[int(x.ptr) for x in Cs])
    gsize = (ctypes.c_int * g)(*([1] * g))

    CUDA_R_32F, CUBLAS_COMPUTE_32F, ALGO_DEFAULT = 0, 1, 0
    try:
        rc = fn(rt.handle, CUDA_OP_T, CUDA_OP_N, gm, gn, gk, aa, ba, ca,
                g, gsize, CUDA_R_32F, CUBLAS_COMPUTE_32F, ALGO_DEFAULT)
        rt.sync()
    except Exception as exc:
        log("Ex 调用异常：%s: %s" % (type(exc).__name__, str(exc)[:160]))
        return
    log("Ex 调用返回 rc=%d" % int(rc))
    if int(rc) != 0:
        log("⇒ Ex 版不可用（rc≠0），走方案 A")
        return

    for i, m in enumerate(Ms):
        ref = np.asarray(rt.d2h(rt.gemm_nt(As[i], Bs[i])), np.float32)
        got = np.asarray(rt.d2h(Cs[i]), np.float32)
        rel = float(np.abs(got - ref).max() / max(1e-9, float(np.abs(ref).max())))
        log("第 %d 组（M=%d）：相对差 %.2e ⇒ %s" % (i, m, rel, "一致" if rel <= 1e-5 else "**不一致**"))
    log("⇒ Ex 版可用：走方案 B（无需 pad）")


if __name__ == "__main__":
    main()
