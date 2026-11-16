# -*- coding: utf-8 -*-
"""验证机制：瘦长 GEMM 的耗时对 n 有多敏感？

MoE 逐专家 GEMM 的形状是 (n_tokens, 1024) @ (2816, 1024)^T，
即 cublas 视角 m=2816, k=1024, n=n_tokens。

若 n=9 与 n=128 耗时接近 => 「固定开销主导」成立，但那样负载不均**不该**拖慢整步；
若 n=9 明显更慢（按单位工作量算）=> 负载不均才是元凶。
本脚本给出 n 扫描曲线，用来判定到底是哪种。
"""
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
M, K = 2816, 1024          # 专家权重形状
rng = np.random.default_rng(0)
W = rt.h2d(rng.normal(0, 0.02, (M, K)).astype(np.float32))


def timed(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    rt.sync()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    xs = {}
    for nt in (1, 9, 15, 34, 50, 64, 79, 128, 256, 512):
        xs[nt] = rt.h2d(rng.normal(0, 1.0, (nt, K)).astype(np.float32))

    print("=== 逐专家 GEMM 耗时 vs token 数（m=%d k=%d）===" % (M, K))
    print("  tokens |   ms   | GFLOP/s | 相对 n=128")
    base = None
    for nt in (1, 9, 15, 34, 50, 64, 79, 128, 256, 512):
        ms = timed(lambda nt=nt: rt.gemm_nt(xs[nt], W))
        fl = 2.0 * nt * M * K
        if nt == 128:
            base = ms
        print("  %6d | %6.3f | %7.1f |" % (nt, ms, fl / (ms * 1e-3) / 1e9), end="")
        if base:
            print(" %.2fx" % (ms / base))
        else:
            print(" -")

    # 关键判定：4 个专家「均衡 128」 vs 「不均 9/15/34/128」总耗时
    print("\n=== 四种专家组合的总耗时（每层 1 次 w1 类 GEMM）===")
    combos = {"均衡 128×4": (128, 128, 128, 128),
              "不均 9/15/34/128": (9, 15, 34, 128),
              "不均 50/79/34/42": (50, 79, 34, 42)}
    for name, combo in combos.items():
        ms = timed(lambda combo=combo: [rt.gemm_nt(xs[t], W) for t in combo], n=10)
        print("  %-20s %6.3f ms" % (name, ms))


if __name__ == "__main__":
    main()
