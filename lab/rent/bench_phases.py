# -*- coding: utf-8 -*-
"""把 all_reduce 的各个阶段单独计时，找出真正的瓶颈（3.16 GiB 量级）。

阶段：① GPU 内打包(D2D) ② d2h ③ 写入共享内存 ④ barrier ⑤ 求和 ⑥ h2d ⑦ 拆回
"""
import time

import numpy as np
from multiprocessing import shared_memory

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()

N = 848_645_257                      # 12 层 V3 的参数量
GB = N * 4 / float(1 << 30)

src = rt.empty((N,), dtype=np.float32)
pack = rt.empty((N,), dtype=np.float32)
host = rt.host_pinned(N * 4, slot="phase")

shm = shared_memory.SharedMemory(name="phase_bench", create=True, size=N * 4 * 2)
own = np.ndarray((N,), dtype=np.float32, buffer=shm.buf, offset=0)
peer = np.ndarray((N,), dtype=np.float32, buffer=shm.buf, offset=N * 4)
allv = np.ndarray((2, N), dtype=np.float32, buffer=shm.buf)


def t(name, fn, n=2):
    fn()
    rt.sync()
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        rt.sync()
        best = min(best, time.perf_counter() - t0)
    print("  %-16s %6.3f s  (%5.2f GiB/s)" % (name, best, GB / best if best else 0))
    return best


print("=== 3.16 GiB 各阶段（取两次最好）===")
t("1_pack_d2d", lambda: rt.copy_d2d(pack, src, N * 4))
t("2_d2h_pinned", lambda: rt.d2h_into(pack, host.ptr))
hv = host.array(np.float32, N)
t("3_copyto_shm", lambda: np.copyto(own, hv))
t("4_sum_axis0", lambda: np.sum(allv, axis=0, out=hv))
peer[:] = 1.0
t("5_add_peer", lambda: np.add(hv, peer, out=hv))
t("6_h2d_pinned", lambda: rt.h2d_into(pack, host.ptr))
t("7_unpack_d2d", lambda: rt.copy_d2d(src, pack, N * 4))

# 纯主机内存对比：pinned↔pinned 的加法有多快（说明 sum 慢是不是 shm 的锅）
h2 = rt.host_pinned(N * 4, slot="phase2")
hv2 = h2.array(np.float32, N)
hv2[:] = 1.0
t("5b_add_pinned", lambda: np.add(hv, hv2, out=hv))

shm.close()
shm.unlink()
