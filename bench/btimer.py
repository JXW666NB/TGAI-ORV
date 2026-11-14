# -*- coding: utf-8 -*-
"""微基准的计时工具：用 CUDA event 量**真实 GPU 时间**，并单独量**主机入队时间**。

为什么要两套：
  * 逐算子 `sync()` 会把同步开销算进去（实测 1189 次 sync ≈ 0.36 s，把真凶盖住）
  * 只量墙钟会把"主机慢"和"GPU 慢"混在一起
所以：GPU 时间用 event 夹在连续 K 次调用之间；主机时间用**极小张量**跑同样次数，
      此时 GPU 几乎不干活，量到的就是纯主机成本（Python + ctypes + 驱动）。
"""
import ctypes
import time

import numpy as np

CU_EVENT_DEFAULT = 0


class Timer:
    def __init__(self, rt=None):
        import tgai_go.train as T
        self.T = T
        self.rt = rt if rt is not None else T._cuda()
        self._ev = None

    # ---- 内部：事件 ----
    def _make_event(self):
        cu = self.rt.cu
        ev = ctypes.c_void_p()
        rc = cu.cuEventCreate(ctypes.byref(ev), ctypes.c_uint(CU_EVENT_DEFAULT))
        if rc != 0 or not ev.value:
            return None
        return ev

    def gpu_ms(self, fn, reps=20, warmup=3):
        """返回一次调用的平均 GPU 时间（毫秒）。事件不可用时返回 None。"""
        rt = self.rt
        for _ in range(warmup):
            fn()
        rt.sync()
        if self._ev is None:
            self._ev = self._make_event()
        ev = self._ev
        if ev is None:
            return None
        cu = rt.cu
        stream = getattr(rt, "_stream", None)
        stream = stream if stream is not None else ctypes.c_void_p(0)
        t0 = time.perf_counter()
        cu.cuEventRecord(ev, stream)
        for _ in range(reps):
            fn()
        end = self._make_event()
        if end is None:
            return None
        cu.cuEventRecord(end, stream)
        rt.sync()
        ms = ctypes.c_float(0.0)
        cu.cuEventElapsedTime(ctypes.byref(ms), ev, end)
        _ = t0
        return float(ms.value) / reps

    def host_us(self, fn, reps=500):
        """返回一次调用的**主机入队**成本（微秒）：先排空，再只入队不同步。

        注意：若 GPU 比主机慢，入队会在队列里堆积，量到的仍接近主机成本
        （因为 cuLaunchKernel 不等 GPU）。用极小张量时 GPU 时间可忽略。
        """
        rt = self.rt
        for _ in range(10):
            fn()
        rt.sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        enq = time.perf_counter() - t0
        rt.sync()
        return enq / reps * 1e6

    def wall_ms(self, fn, reps=20, warmup=3):
        """返回一次调用的墙钟时间（毫秒），含同步。"""
        rt = self.rt
        for _ in range(warmup):
            fn()
        rt.sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        rt.sync()
        return (time.perf_counter() - t0) / reps * 1e3

    def report(self, name, fn, flops=None, bytes_moved=None, reps=20):
        """打印一行：GPU / 主机 / 墙钟，并给出等效算力或带宽。"""
        g = self.gpu_ms(fn, reps=reps)
        h = self.host_us(fn, reps=200)
        w = self.wall_ms(fn, reps=reps)
        extra = ""
        if flops and g:
            extra += "  %8.1f GFLOPS" % (flops / (g * 1e-3) / 1e9)
        if bytes_moved and g:
            extra += "  %6.1f GB/s" % (bytes_moved / (g * 1e-3) / 1e9)
        print("%-34s GPU %8.3f ms  主机 %7.1f us  墙钟 %8.3f ms%s"
              % (name, g if g else -1, h, w, extra))
        return g, h, w


def rnd(*shape, seed=0, scale=0.3):
    rng = np.random.default_rng(seed)
    return rng.normal(0, scale, shape).astype(np.float32)
