# -*- coding: utf-8 -*-
"""复现 head/embedding 反向的两个大 GEMM 的真实成本。

backward 实测：vocab 4096 → 21.5 ms；vocab 96000 → 186 ms。
差 165 ms 与词表成正比。这里量该路径上的两个 GEMM：
  dW = g^T @ x :  A(S,V)^T @ B(S,D)  -> (V,D)，K = S（很短！）
  dX = g @ W   :  A(S,V)   @ B(V,D)  -> (S,D)，K = V（很长）
短 K 的 dW 在 cublas 里效率极差是已知问题 —— 本脚本给出真实数字。
"""
import argparse
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()


def timed(fn, n=10, warm=3):
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
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=96000)
    a = ap.parse_args()
    S, D, V = a.seq, a.d, a.vocab
    rng = np.random.default_rng(0)

    g = rt.h2d(rng.normal(0, 1e-3, (S, V)).astype(np.float32))     # (S,V) 梯度
    x = rt.h2d(rng.normal(0, 0.02, (S, D)).astype(np.float32))     # (S,D) 激活
    W = rt.h2d(rng.normal(0, 0.02, (V, D)).astype(np.float32))     # (V,D) 权重

    # dW = g^T @ x  →  gemm_tn(g, x)   A(S,V)^T @ B(S,D) = (V,D)
    f_dw = 2.0 * V * D * S
    ms_dw = timed(lambda: rt.gemm_tn(g, x), n=5)
    # dX = g @ W    →  sgemm(g, W)     (S,V) @ (V,D) = (S,D)
    f_dx = 2.0 * S * D * V
    ms_dx = timed(lambda: rt.sgemm(g, W), n=5)
    # 零化 (V,D) 的成本（如果实现里先 zero 再 scatter）
    ms_zero = timed(lambda: rt.empty((V, D), dtype=np.float32), n=10)

    print("=== head/embedding 反向（S=%d D=%d V=%d）===" % (S, D, V))
    print("  dW = g^T @ x  (K=%d): %8.3f ms  %.1f GFLOP  -> %.2f TFLOPS"
          % (S, ms_dw, f_dw / 1e9, f_dw / (ms_dw * 1e-3) / 1e12))
    print("  dX = g  @ W   (K=%d): %8.3f ms  %.1f GFLOP  -> %.2f TFLOPS"
          % (V, ms_dx, f_dx / 1e9, f_dx / (ms_dx * 1e-3) / 1e12))
    print("  (V,D) 缓冲分配:        %8.3f ms" % ms_zero)
    print("  两者合计: %.3f ms" % (ms_dw + ms_dx))
    print("\n  对照：backward 实测差 %d ms（vocab 96000 减 4096 的线性部分）" % 165)


if __name__ == "__main__":
    main()
