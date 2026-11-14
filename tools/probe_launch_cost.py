# -*- coding: utf-8 -*-
"""测单次内核发射的固定开销（Python + ctypes + 驱动）。"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    n = 4096
    a = rt.h2d(np.ones(n, np.float32))
    b = rt.h2d(np.ones(n, np.float32))
    c = rt.empty((n,), dtype=np.float32)

    def bench(label, fn, reps=2000):
        fn(); rt.sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        enq = (time.perf_counter() - t0) / reps          # 只入队
        rt.sync()
        tot = (time.perf_counter() - t0) / reps          # 含执行
        print("%-22s 入队 %.1f us   入队+执行 %.1f us" % (label, enq * 1e6, tot * 1e6))

    bench("空内核(add_ n=4096)", lambda: rt.add_(a, b))
    bench("小 GEMM 128x128x128",
          lambda: rt._sgemm(0, 0, 128, 128, 128, a.ptr, 128, b.ptr, 128, c.ptr, 128))

    # 分配/释放：4 MiB 与 16 MiB（训练里最常见的两种尺寸）
    big = None
    def alloc_free(nbytes, reps=1000):
        nonlocal big
        shape = (nbytes // 4,)
        buf = rt.empty(shape, dtype=np.float32)      # 预热
        del buf
        rt.sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            big = rt.empty(shape, dtype=np.float32)
            del big
        return (time.perf_counter() - t0) / reps

    for mb in (4, 16, 64):
        us = alloc_free(mb * 2**20) * 1e6
        print("分配+释放 %-3d MiB      %.1f us" % (mb, us))
    st = rt.pool_stats()
    print("池：hits=%s trims=%s live=%.1f MiB" % (st.get("hits"), st.get("trims"),
                                                 st.get("live_bytes", 0) / 2**20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
