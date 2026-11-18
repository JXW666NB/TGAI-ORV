# -*- coding: utf-8 -*-
"""单独测 DP 的通信开销：all_reduce_grads 到底花多少时间、带宽多少。

用法（两个进程，各指定自己的卡）：
    CUDA_VISIBLE_DEVICES=0 TGAI_DP_RANK=0 TGAI_DP_WORLD=2 TGAI_DP_JOB=bench python3 bench_reduce.py
    CUDA_VISIBLE_DEVICES=1 TGAI_DP_RANK=1 TGAI_DP_WORLD=2 TGAI_DP_JOB=bench python3 bench_reduce.py

分别测 1.5B 量级的梯度（5.6 GiB fp32）与 1/4 量级，给出有效带宽。
"""
import os
import time

import numpy as np

import tgai_go.train as T
from tgai_go.train import dist as D

T.set_default_device("cuda")
rt = T._cuda()
dp = D.DataParallel()

N = 1_500_681_490          # V3 参数量
sizes = [N // 64, N // 16, N // 4, N // 2, N]   # 阶梯，看带宽是否随规模稳定


def make_params(n):
    """构造 n 个参数（总元素数 n），每个是一维张量。"""
    out = []
    left = n
    chunk = max(1, n // 64)
    while left > 0:
        m = min(chunk, left)
        p = T.tensor(np.zeros(m, np.float32), requires_grad=True)
        p.grad = T.tensor(np.ones(m, np.float32))
        out.append(p)
        left -= m
    return out


print("rank=%d world=%d dir=%s" % (dp.rank, dp.world, dp.dir), flush=True)
for n in sizes:
    ps = make_params(n)
    rt.sync()
    dp.all_reduce_grads(ps)                     # 预热（首轮要建共享内存槽）
    rt.sync()
    t0 = time.perf_counter()
    dp.all_reduce_grads(ps)
    rt.sync()
    dt = time.perf_counter() - t0
    gb = n * 4 / (1 << 30)
    print("  元素 %-12d (%.2f GiB fp32)  归约 %7.3f s  ⇒ %6.1f GiB/s"
          % (n, gb, dt, gb / dt if dt else 0), flush=True)
    del ps
