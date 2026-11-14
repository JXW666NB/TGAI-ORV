# -*- coding: utf-8 -*-
"""GEMM 效率随 M 的变化：验证"专家行数太少导致小 GEMM 低效"的假设。"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    K, N = 512, 2048
    b = rt.h2d(np.random.rand(K, N).astype(np.float32))
    print("%6s %10s %10s %10s" % ("M", "耗时ms", "GFLOPS", "相对效率"))
    base = None
    for M in (128, 256, 512, 1024, 2048, 4096):
        a = rt.h2d(np.random.rand(M, K).astype(np.float32))
        c = rt.empty((M, N), dtype=np.float32)
        f = lambda: rt._sgemm(0, 0, N, M, K, b.ptr, N, a.ptr, K, c.ptr, N)
        f(); rt.sync()
        t0 = time.perf_counter()
        for _ in range(20):
            f()
        rt.sync()
        dt = (time.perf_counter() - t0) / 20
        gflops = 2.0 * M * K * N / dt / 1e9
        if base is None:
            base = gflops
        print("%6d %10.3f %10.1f %10.2fx" % (M, dt * 1e3, gflops, gflops / base))
        del a, c
        rt.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
