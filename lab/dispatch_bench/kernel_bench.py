# -*- coding: utf-8 -*-
"""内核级微基准（老办法：拆开逐个测，用数字定位）。

要回答三个问题：
  1. fp16 GEMM 到底慢在哪 —— cast 还是 GEMM？
  2. 单个 GEMM 的实际 TFLOPS 是多少？相对 fp32 峰值（V100 = 15.7 TFLOPS）
     和 fp16 峰值（125 TFLOPS）差多少？
  3. 一次内核派发（launch）到底要多久？如果它和一次小 GEMM 同量级，
     那"派发瓶颈"就不是猜测而是事实。

在 V100 上跑：D:\\orv_pip_0962\\Scripts\\python.exe kernel_bench.py
"""
import sys
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
rng = np.random.default_rng(0)

V100_FP32_TFLOPS = 15.7
V100_FP16_TFLOPS = 125.0
V100_BW_GBS = 900.0


def _sync():
    rt.sync()


def timed(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    _sync()
    return (time.perf_counter() - t0) / n * 1e3      # ms


def bench_gemm(M, K, N, n=20):
    """同一个逻辑 GEMM（C = A @ B^T）的四种走法。"""
    A = rng.normal(0, 1.0, (M, K)).astype(np.float32)
    B = rng.normal(0, 0.02, (N, K)).astype(np.float32)
    dA, dB = rt.h2d(A), rt.h2d(B)
    A16 = rt.cast_f32_to_f16(dA)
    B16 = rt.cast_f32_to_f16(dB)
    flops = 2.0 * M * N * K

    r = {}
    r["fp32 gemm_nt"] = timed(lambda: rt.gemm_nt(dA, dB), n)
    r["fp16 热路径（含 cast）"] = timed(lambda: rt.sgemm_fp16_nt(dA, dB), n)
    r["fp16 仅 GEMM（已预 cast）"] = timed(lambda: rt.gemm_f16_nt_raw(A16, B16, M, N, K), n)
    r["单独 cast 一次权重"] = timed(lambda: rt.cast_f32_to_f16(dB), n)

    print("\n=== GEMM  M=%d K=%d N=%d  (%.2f GFLOP) ===" % (M, K, N, flops / 1e9))
    for k, ms in r.items():
        tf = flops / (ms * 1e-3) / 1e12
        tag = ""
        if "fp32" in k:
            tag = "  = %.1f%% fp32 峰值" % (tf / V100_FP32_TFLOPS * 100)
        elif "GEMM" in k or "热路径" in k:
            tag = "  = %.1f%% fp16 峰值 / %.1f%% fp32 峰值" % (
                tf / V100_FP16_TFLOPS * 100, tf / V100_FP32_TFLOPS * 100)
        print("  %-28s %8.3f ms   %7.1f TFLOPS%s" % (k, ms, tf, tag))
    cast_ms = r["单独 cast 一次权重"]
    hot, pure = r["fp16 热路径（含 cast）"], r["fp16 仅 GEMM（已预 cast）"]
    print("  → cast 占热路径的时间: %.0f%%（%.3f / %.3f ms）"
          % (cast_ms / hot * 100, cast_ms, hot))
    print("  → fp16 GEMM 本身 vs fp32: %.2f× 提速" % (r["fp32 gemm_nt"] / pure))


def bench_launch(n=2000):
    """一次内核派发的最小成本：对极小张量做一次逐元素操作。"""
    x = rt.h2d(np.ones(256, np.float32))
    ms = timed(lambda: rt.launch("cast_f32_to_f16", (1, 1, 1), (256, 1, 1),
                                 [rt._ptr(x), rt._ptr(x), (256, __import__("ctypes").c_int)]),
               n=n, warm=10)
    print("\n=== 单次内核派发 ===\n  %d 次平均 %.4f ms/次（%.1f µs）" % (n, ms, ms * 1e3))
    return ms


def bench_memcpy(bytes_n):
    """内存带宽：一次 fp32→fp16 cast 的等效带宽（读 4B + 写 2B）。"""
    n = bytes_n // 4
    x = rt.h2d(np.ones(n, np.float32))
    ms = timed(lambda: rt.cast_f32_to_f16(x), n=20)
    moved = n * 6
    print("  cast %d 元素: %.3f ms  ⇒ %.1f GB/s（峰值 %.0f）"
          % (n, ms, moved / (ms * 1e-3) / 1e9, V100_BW_GBS))


def main():
    print("=== V100 内核微基准（fp32 峰值 15.7 / fp16 峰值 125 TFLOPS）===")
    # 模型里最重的几个形状
    bench_gemm(512, 1024, 2816)      # MoE 专家 fwd
    bench_gemm(512, 1024, 96000)     # lm_head
    bench_gemm(1024, 512, 2816)      # dW（反向）
    bench_launch()
    bench_memcpy(1024 * 1024 * 64)   # 64M 元素


if __name__ == "__main__":
    main()
