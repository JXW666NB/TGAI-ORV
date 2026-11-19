# -*- coding: utf-8 -*-
"""同形状微基准：GEMM 的 TFLOPS 与逐元素带宽，ORV vs PyTorch。

判据：
  * 若 GEMM 的 TFLOPS 接近 ⇒ 单个 kernel 不慢，差异来自"做了多少遍访存"
  * 若逐元素带宽接近（2 MB 与 8 MB 各测一次）⇒ 内存路径也不慢
  * 2 MB → 8 MB 的性能掉档 ⇒ 验证 L2 溢出假设

用法：
  python microbench.py orv     # ORV 运行时
  python microbench.py pyt     # PyTorch
"""
import json
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "orv"
M, K, N = 2048, 1024, 3328          # 本模型里常见的一个 GEMM 形状
REP = 20


def bench(fn, rep=REP, sync=None):
    """★ 必须把同步放进来：CUDA 调用是异步的，不同步测到的只是发射耗时
    （第一次测出 V100 上 672 TFLOPS fp16 的荒谬数字，就是这个坑）。"""
    fn()                             # 预热
    if sync:
        sync()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    if sync:
        sync()
    return (time.perf_counter() - t0) / rep


if MODE == "orv":
    import numpy as np
    import tgai_go.train as T

    T.set_default_device("cuda")
    rt = T._cuda()
    rt.sync()

    out = {}
    # ── fp16 GEMM（走 tensor core）──
    import ctypes as ct
    A16 = rt.empty((M, K), dtype=np.uint16)
    B16 = rt.empty((N, K), dtype=np.uint16)
    C16 = rt.empty((M, N), dtype=np.uint16)
    fn = getattr(rt, "gemm_f16_nt_raw", None)
    if fn is not None:
        t = bench(lambda: fn(A16, B16, M, N, K), sync=rt.sync)
        flops = 2.0 * M * N * K
        out["gemm_f16_tflops"] = flops / t / 1e12
    # ── fp32 GEMM ──
    A32 = rt.empty((M, K))
    B32 = rt.empty((N, K))
    C32 = rt.empty((M, N))
    if hasattr(rt, "sgemm_nt"):
        t = bench(lambda: rt.sgemm_nt(A32, B32, M, N, K), sync=rt.sync)
        out["gemm_f32_tflops"] = 2.0 * M * N * K / t / 1e12
    # ── 逐元素：2 MB 与 8 MB 各测一次（c = a + b）──
    for elems, tag in ((524288, "2MB"), (2097152, "8MB")):
        a = rt.empty((elems,))
        b = rt.empty((elems,))
        c = rt.empty((elems,))
        t = bench(lambda: rt.add(a, b), sync=rt.sync)
        gb = elems * 4 * 3 / 1e9          # 读 a、读 b、写 c
        out["add_gbps_" + tag] = gb / t
    rt.sync()
    print(json.dumps(out))

else:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    out = {}
    a16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b16 = torch.randn(K, N, device="cuda", dtype=torch.float16)
    t = bench(lambda: torch.mm(a16, b16), sync=torch.cuda.synchronize)
    out["gemm_f16_tflops"] = 2.0 * M * N * K / t / 1e12

    a32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b32 = torch.randn(K, N, device="cuda", dtype=torch.float32)
    t = bench(lambda: torch.mm(a32, b32), sync=torch.cuda.synchronize)
    out["gemm_f32_tflops"] = 2.0 * M * N * K / t / 1e12

    for elems, tag in ((524288, "2MB"), (2097152, "8MB")):
        a = torch.randn(elems, device="cuda")
        b = torch.randn(elems, device="cuda")
        c = torch.empty(elems, device="cuda")
        t = bench(lambda: torch.add(a, b, out=c), sync=torch.cuda.synchronize)
        out["add_gbps_" + tag] = elems * 4 * 3 / 1e9 / t
    print(json.dumps(out))
