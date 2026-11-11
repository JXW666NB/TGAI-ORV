# -*- coding: utf-8 -*-
"""Muon 的 Newton-Schulz 拆解：35 s/步到底花在哪。

## 背景

每步 3 个 GEMM（8.6 + 2.1 + 8.6 GFLOP），5 步 = 96.5 GFLOP/矩阵；
224 个矩阵 ⇒ 21.6 TFLOP/步。按实测 cuBLAS 8.7 TFLOP/s，**纯 GEMM 应 ≈2.5 s**。
实测 35 s ⇒ 差 14×。要找出这 14× 在哪，不能猜。

## 本脚本测什么

逐层剥开，看每一层加进去多少时间：

1. 裸 GEMM（[1024,4096]×[4096,1024]）—— 纯算力基准
2. 裸 GEMM（[1024,1024]×[1024,1024]）—— NS 第 2 个 GEMM 的形状
3. `transpose2d(x)` 单独成本
4. `read_scalar(sumsq_sequential(x))` —— 每矩阵一次的**同步**
5. 一个完整的 NS 轮次（3 GEMM + 若干 elementwise）
6. 完整的 5 步 NS（单个矩阵）
7. 实际的 `_step_one`（含所有开销）

据此判断：瓶颈是算力、分配、同步，还是 Python 调度。

运行（需独占 GPU）：python muon_profile.py
"""
import io
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "TGAI-ORV-main", "python"))

OUT = os.path.join(HERE, "mprof.txt")
R, C = 1024, 4096          # 典型 Muon 参数形状（转置后 r<=c）


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.train import _newton_schulz5_gpu

    T.set_default_device("cuda")
    rt = T._cuda()
    rep = []
    rng = np.random.default_rng(0)

    def ev(fn, n=5, warm=3):
        for _ in range(warm):
            fn()
        rt.sync()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(n):
            fn()
        e1.record(); e1.synchronize()
        return e0.elapsed_time(e1) / n

    x = rt.h2d(rng.normal(0, 0.02, (R, C)).astype(np.float32))
    s = rt.h2d(rng.normal(0, 0.02, (R, R)).astype(np.float32))

    # 1) 裸 GEMM：[1024,4096] @ [4096,1024]
    t1 = ev(lambda: rt.gemm_nt(x, x))
    f1 = 2 * R * R * C
    rep.append("1) gemm_nt [1024,4096]x[4096,1024]  %8.3f ms  %6.2f TFLOP/s"
               % (t1, f1 / (t1 * 1e9)))

    # 2) 裸 GEMM：[1024,1024] @ [1024,1024]
    t2 = ev(lambda: rt.gemm_nt(s, s))
    f2 = 2 * R * R * R
    rep.append("2) gemm_nt [1024,1024]x[1024,1024]  %8.3f ms  %6.2f TFLOP/s"
               % (t2, f2 / (t2 * 1e9)))

    # 3) transpose2d
    t3 = ev(lambda: rt.transpose2d(x))
    rep.append("3) transpose2d [1024,4096]          %8.3f ms" % t3)

    # 4) read_scalar —— 每矩阵一次的同步点（对照顺序版与并行版）
    t4a = ev(lambda: rt.read_scalar(rt.sumsq_sequential(x)), n=10)
    t4b = ev(lambda: rt.read_scalar(rt.sumsq_parallel(x)), n=10)
    rep.append("4a) read_scalar(sumsq_sequential)  %8.3f ms  ← 旧：单线程归约" % t4a)
    rep.append("4b) read_scalar(sumsq_parallel)    %8.3f ms  ← 新：两趟并行" % t4b)

    # 5) 一个完整 NS 轮次（和 fast 路径同构）
    xt = rt.transpose2d(x)

    def one_round():
        ss = rt.gemm_nt(s, s)
        y = rt.add(rt.scale(s, -4.7750), rt.scale(ss, 2.0315))
        return rt.add(rt.scale(x, 3.4445), rt.gemm_nt(y, xt))

    t5 = ev(one_round, n=3)
    rep.append("5) 一轮 NS（3 GEMM + elementwise）  %8.3f ms  ← 理论应 ≈%.3f ms"
               % (t5, t1 + t2 + t1))

    # 6) 完整 5 步 NS（单矩阵，fast=True）
    t6 = ev(lambda: _newton_schulz5_gpu(rt, x, steps=5, fast=True), n=3)
    rep.append("6) 完整 5 步 NS（单矩阵）            %8.3f ms" % t6)
    rep.append("   ⇒ 224 个矩阵 extrapolate:          %8.1f s" % (t6 * 224 / 1000.0))

    # 7) 单矩阵纯 GEMM 理论下限
    theo = 5 * (t1 + t2 + t1)
    rep.append("7) 224 矩阵纯 GEMM 下限（实测 t1/t2）  %8.1f s" % (theo * 224 / 1000.0))

    # 8) 分配开销：反复新建同尺寸缓冲
    t8 = ev(lambda: rt.empty((R, C), dtype=np.float32), n=20)
    rep.append("8) rt.empty [1024,4096] 单次          %8.4f ms  ← 若不复用会累计" % t8)

    # 9) Python 调用开销（无计算）
    t9 = ev(lambda: None, n=1000)
    rep.append("9) 空函数（Python 侧基准）            %8.4f ms" % t9)

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
