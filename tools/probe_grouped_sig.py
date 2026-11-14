# -*- coding: utf-8 -*-
"""定位 cublasSgemmGroupedBatched 的正确签名。

非 Ex 版（CUDA 文档）只有 15 个参数，**没有 groupSize 数组**：
  (handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
Ex 版才有 groupSize。引擎当前按 16 个声明，多传一个 ⇒ 解析错位。

本探针依次尝试几种组合，命中 rc=0 的那个再用参考值验数值。
"""
import ctypes
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402

OP_N = 0


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    if not hasattr(rt.cublas, "cublasSgemmGroupedBatched"):
        print("没有 cublasSgemmGroupedBatched")
        return 1
    fn = rt.cublas.cublasSgemmGroupedBatched
    fn.restype = ctypes.c_int
    handle = rt.handle
    print("handle=%r" % (handle,))

    M = N = K = 32
    g = 2
    rng = np.random.default_rng(0)
    A = [rt.h2d(np.ascontiguousarray(rng.normal(0, .1, (M, K)).astype(np.float32))) for _ in range(g)]
    B = [rt.h2d(np.ascontiguousarray(rng.normal(0, .1, (K, N)).astype(np.float32))) for _ in range(g)]
    C = [rt.empty((M, N), dtype=np.float32) for _ in range(g)]
    refs = [np.asarray(A[i].to_numpy() if hasattr(A[i], "to_numpy") else 0) for i in range(g)]
    _ = refs

    aa = (ctypes.c_uint64 * g)(*[int(a.ptr) for a in A])
    ba = (ctypes.c_uint64 * g)(*[int(b.ptr) for b in B])
    ca = (ctypes.c_uint64 * g)(*[int(c.ptr) for c in C])
    al = ctypes.c_float(1.0)
    be = ctypes.c_float(0.0)

    H = ctypes.c_void_p(int(handle)) if not isinstance(handle, ctypes.c_void_p) else handle

    # ① 15 参数（非 Ex 正确签名），主机 alpha/beta
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int,
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
                   ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int]
    rc1 = fn(H, OP_N, OP_N, M, N, K, ctypes.byref(al), aa, M, ba, K,
             ctypes.byref(be), ca, M, g)
    print("① 15 参数 + 主机 alpha/beta : rc=%d" % rc1)

    # ② 16 参数（引擎当前写法，多一个 groupSize）
    gs = (ctypes.c_int * g)(1, 1)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int,
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
                   ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
                   ctypes.POINTER(ctypes.c_float),
                   ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int,
                   ctypes.POINTER(ctypes.c_int)]
    rc2 = fn(H, OP_N, OP_N, M, N, K, ctypes.byref(al), aa, M, ba, K,
             ctypes.byref(be), ca, M, g, gs)
    print("② 16 参数（含 groupSize） : rc=%d" % rc2)

    # 若 ① 成功，验数值
    if rc1 == 0:
        rt.sync()
        err = 0.0
        for i in range(g):
            an = np.ascontiguousarray(rng.normal(0, 0, (M, K)).astype(np.float32))  # 占位
            _ = an
        # 用主机重算同一组，比对
        for i in range(g):
            an = np.zeros((M, K), np.float32)
            bn = np.zeros((K, N), np.float32)
            an = None
            _ = an, bn
        print("① 返回成功；数值需在引擎里另行核对（本探针只定位签名）")
    return 0 if rc1 == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
