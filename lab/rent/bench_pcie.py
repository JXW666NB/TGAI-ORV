# -*- coding: utf-8 -*-
"""测 PCIe 带宽地板：d2h / h2d 各 3.16 GiB 要多久。

为什么先测这个：DP 的归约必须把梯度搬到主机再搬回（无 P2P），
所以「归约时间下界 = 2 × 梯度量 / PCIe 有效带宽」。不先测这个，
就没法判断"优化到 0.7 s"是可达还是物理不可能。
"""
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()

N = 12 * 0 + 848_645_257            # 12 层 V3 的参数量（= 3.16 GiB fp32）
sizes = [N // 4, N // 2, N]

print("设备：%s" % (rt.device_caps() if hasattr(rt, "device_caps") else "?"))
for n in sizes:
    gb = n * 4 / float(1 << 30)
    buf = rt.empty((n,), dtype=np.float32)
    arr = np.ones(n, dtype=np.float32)

    # 预热
    _ = rt.d2h(buf)
    _ = rt.h2d(arr)

    t0 = time.perf_counter()
    _ = rt.d2h(buf)
    t_d2h = time.perf_counter() - t0

    t0 = time.perf_counter()
    b2 = rt.h2d(arr)
    t_h2d = time.perf_counter() - t0

    print("  %.2f GiB：d2h %6.3f s (%5.2f GiB/s)   h2d %6.3f s (%5.2f GiB/s)  "
          "往返合计 %6.3f s"
          % (gb, t_d2h, gb / t_d2h, t_h2d, gb / t_h2d, t_d2h + t_h2d))
    del buf, b2, arr, _
    rt.sync()
