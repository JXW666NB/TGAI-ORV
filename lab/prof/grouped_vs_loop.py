# -*- coding: utf-8 -*-
"""MoE 专家 GEMM：逐专家循环 vs grouped（cublasSgemmGroupedBatched）。

模型实测：一步里 252 次 `_sgemm`，其中专家 GEMM 是瘦长形状
（每专家只有约 131 个 token，seq=256 / top_k=2 / E=4）。
本脚本量同样工作量下两者的差距。
"""
import argparse
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()


def timed(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    rt.sync()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=131)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--d_ff", type=int, default=2816)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    E, N, D, F = a.E, a.tokens, a.d, a.d_ff
    xs = [rt.h2d(rng.normal(0, 1.0, (N, D)).astype(np.float32)) for _ in range(E)]
    W1 = [rt.h2d(rng.normal(0, 0.02, (F, D)).astype(np.float32)) for _ in range(E)]
    hs = [rt.h2d(rng.normal(0, 1.0, (N, F)).astype(np.float32)) for _ in range(E)]
    W2 = [rt.h2d(rng.normal(0, 0.02, (D, F)).astype(np.float32)) for _ in range(E)]

    flops = E * (2.0 * N * F * D * 2)          # w1 类 + w2 类各一次

    loop_ms = timed(lambda: [rt.gemm_nt(xs[e], W1[e]) for e in range(E)]
                    + [rt.gemm_nt(hs[e], W2[e]) for e in range(E)])
    try:
        grp_ms = timed(lambda: (rt.grouped_nt(xs, W1), rt.grouped_nt(hs, W2)))
        grp_ok = True
    except Exception as e:
        grp_ms = float("nan")
        grp_ok = False
        print("grouped 调用失败：%r" % (e,))

    print("=== MoE 专家 GEMM：E=%d 每专家 %d token  d=%d d_ff=%d ===" % (E, N, D, F))
    print("  逐专家循环（%d 次 gemm_nt）: %8.3f ms  (%.1f GFLOP/s 等效 %.2f TFLOPS)"
          % (2 * E, loop_ms, flops / (loop_ms * 1e-3) / 1e9, flops / (loop_ms * 1e-3) / 1e12))
    if grp_ok:
        print("  grouped_nt（2 次调用）    : %8.3f ms  (%.2f TFLOPS)"
              % (grp_ms, flops / (grp_ms * 1e-3) / 1e12))
        print("  → grouped 快 %.2f×" % (loop_ms / grp_ms))
    else:
        print("  grouped 不可用")


if __name__ == "__main__":
    main()
