# -*- coding: utf-8 -*-
"""验证 rc=7 的假设：cublasSgemmGroupedBatched 的 alpha/beta 要主机还是设备指针。

非 Ex 版签名（按文档）：
  cublasSgemmGroupedBatched(handle, transa, transb, m, n, k, alpha,
                            Aarray[], lda, Barray[], ldb, beta, Carray[], ldc, batchCount)
返回 0 表示成功；7 = CUBLAS_STATUS_INVALID_VALUE。
"""
import ctypes
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402

CUDA_OP_N = 0


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    try:
        fn = rt.cublas.cublasSgemmGroupedBatched
    except AttributeError:
        print("本机 libcublas 没有 cublasSgemmGroupedBatched")
        return 1
    fn.restype = ctypes.c_int
    handle = rt.cublas_handle if hasattr(rt, "cublas_handle") else rt._cublas_handle

    M = N = K = 64
    g = 2
    rng = np.random.default_rng(0)
    A, B, C = [], [], []
    for _ in range(g):
        A.append(rt.h2d(np.ascontiguousarray(rng.normal(0, .1, (M, K)).astype(np.float32))))
        B.append(rt.h2d(np.ascontiguousarray(rng.normal(0, .1, (K, N)).astype(np.float32))))
        C.append(rt.empty((M, N), dtype=np.float32))

    aa = (ctypes.c_uint64 * g)(*[int(a.ptr) for a in A])
    ba = (ctypes.c_uint64 * g)(*[int(b.ptr) for b in B])
    ca = (ctypes.c_uint64 * g)(*[int(c.ptr) for c in C])

    # ① 主机标量 alpha/beta
    al = ctypes.c_float(1.0)
    be = ctypes.c_float(0.0)
    fn.restype = ctypes.c_int
    rc_host = fn(handle, CUDA_OP_N, CUDA_OP_N, M, N, K,
                 ctypes.byref(al), aa, M, ba, K, ctypes.byref(be), ca, M, g)

    # ② 设备标量 alpha/beta（分组 API 的已知怪癖：pointer mode 可能需要设备指针）
    al_d = rt.h2d(np.array([1.0], np.float32))
    be_d = rt.h2d(np.array([0.0], np.float32))
    rc_dev = fn(handle, CUDA_OP_N, CUDA_OP_N, M, N, K,
                ctypes.byref(ctypes.c_void_p(int(al_d.ptr))), aa, M, ba, K,
                ctypes.byref(ctypes.c_void_p(int(be_d.ptr))), ca, M, g)

    print("主机 alpha/beta : rc=%d" % rc_host)
    print("设备 alpha/beta : rc=%d" % rc_dev)

    # 哪条成功了就用它验数值
    for tag, rc in (("host", rc_host), ("device", rc_dev)):
        if rc == 0:
            rt.sync()
            ok = True
            for i in range(g):
                got = rt.d2h_np(C[i]) if hasattr(rt, "d2h_np") else np.asarray(C[i].to_numpy())
                ref = np.asarray(A[i].to_numpy() if hasattr(A[i], "to_numpy") else 0)
                _ = ref
            print("  -> %s 模式可用（数值另验）" % tag)
    return 0 if min(rc_host, rc_dev) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
