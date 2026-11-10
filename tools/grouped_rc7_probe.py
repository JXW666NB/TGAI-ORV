# -*- coding: utf-8 -*-
"""定位分组 GEMM 的 rc=7：到底是 transa 不被接受，还是别的原因。

一次跑完候选组合，避免来回猜：
  1. (T, N) —— 当前用法，已知 rc=7，作为基线
  2. (N, N) —— 若通，说明这个版本只接受 CUBLAS_OP_N
  3. (N, N) 且 alpha/beta 走设备指针 —— 排除指针模式
  4. 单组（batchCount=1）(T, N) —— 排除多组参数

运行（需独占 GPU）：python grouped_rc7_probe.py
"""
import ctypes
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

OP_N, OP_T = 0, 1
CUDA_R_32F = 0


def log(m):
    print(m, flush=True)


def main():
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)

    fn = rt._grouped_ready()
    if fn is None:
        log("分组符号不可用")
        return
    log("符号可用，argtypes 已按标量版声明")

    K, N, M, G = 64, 32, 8, 3
    As = [rt.h2d(rng.normal(0, 0.1, (M, K)).astype(np.float32)) for _ in range(G)]
    Bs = [rt.h2d(rng.normal(0, 0.1, (N, K)).astype(np.float32)) for _ in range(G)]
    Cs = [rt.empty((M, N), dtype=np.float32) for _ in range(G)]

    def call(transa, transb, m, n, k, alpha, aa, lda, ba, ldb, beta, ca, ldc, cnt, gsize):
        alpha_c = ctypes.c_float(alpha)
        beta_c = ctypes.c_float(beta)
        try:
            rc = fn(rt.handle, int(transa), int(transb), int(m), int(n), int(k),
                    ctypes.byref(alpha_c), aa, int(lda), ba, int(ldb),
                    ctypes.byref(beta_c), ca, int(ldc), int(cnt), gsize)
        except Exception as exc:
            return -999, str(exc)[:80]
        return int(rc), ""

    def mkarr(vals):
        arr = (ctypes.c_uint64 * len(vals))()
        for i, v in enumerate(vals):
            arr[i] = ctypes.c_uint64(int(v))
        return arr

    aa = mkarr([x.ptr for x in As])
    ba = mkarr([x.ptr for x in Bs])
    ca = mkarr([x.ptr for x in Cs])
    gs = (ctypes.c_int * G)(*([1] * G))
    gs1 = (ctypes.c_int * 1)(1)

    cases = [
        ("1) (T,N) 多组 —— 当前基线", OP_T, OP_N, M, N, K, G, gs),
        ("2) (N,N) 多组", OP_N, OP_N, M, K, K, G, gs),
        ("3) (T,N) 单组", OP_T, OP_N, M, N, K, 1, gs1),
        ("4) (N,N) 单组", OP_N, OP_N, M, K, K, 1, gs1),
    ]
    for name, ta, tb, m, n, k, cnt, gsize in cases:
        rc, err = call(ta, tb, m, n, k, 1.0, aa, k if ta == OP_T else k,
                       ba, k if tb == OP_T else k,
                       0.0, ca, m if ta == OP_T else m, cnt, gsize)
        rt.sync()
        log("%-26s rc=%d %s" % (name, rc, ("(" + err + ")") if err else ""))
        if rc != 0:
            # 换一种 lda/ldb/ldc 试试（列主序下的常见取值）
            for lda_v, ldb_v, ldc_v in ((m, n, m), (k, k, m), (m, k, m)):
                rc2, _ = call(ta, tb, m, n, k, 1.0, aa, lda_v, ba, ldb_v, 0.0, ca, ldc_v, cnt, gsize)
                rt.sync()
                if rc2 == 0:
                    log("      ↑ 用 lda=%d ldb=%d ldc=%d 时 rc=0" % (lda_v, ldb_v, ldc_v))
                    break


if __name__ == "__main__":
    main()
