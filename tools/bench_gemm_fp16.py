# -*- coding: utf-8 -*-
"""fp16 GEMM（fp32 累加）vs fp32 GEMM：V100 Tensor Core 值不值得上。

引擎里已有 `rt.sgemm_fp16`。这里按模型真实 shape 对比 fp32/fp16 的 GPU 时间，
并和 torch 的 fp16 路径对照。够快才谈"训练精度策略"。
"""
import sys
import time

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")
import numpy as np                        # noqa: E402
import torch                              # noqa: E402

import tgai_go.train as T                 # noqa: E402

T.set_default_device("cuda")
rt = T._cuda()
torch.backends.cuda.matmul.allow_tf32 = False

N, D, FF, V = 2048, 896, 4864, 151936
SHAPES = [("mlp_w1", N, D, FF), ("mlp_w3", N, FF, D), ("lm_head", N, D, V)]


def bench(fn, iters=2):
    fn()
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
        rt.sync()
    return (time.perf_counter() - t0) / iters


print("%-10s %-20s %10s %10s %8s" % ("op", "shape", "fp32(ms)", "fp16(ms)", "speedup"))
for name, m, k, n in SHAPES:
    a32 = T.tensor(np.random.randn(m, k).astype(np.float32))
    b32 = T.tensor(np.random.randn(k, n).astype(np.float32))
    fl = 2 * m * k * n
    t32 = bench(lambda: a32 @ b32)
    # fp16 路径：缓冲区用 uint16 存，内容是 fp16 位模式
    try:
        a16 = rt.h2d(np.random.randn(m, k).astype(np.float16), dtype=np.uint16)
        b16 = rt.h2d(np.random.randn(k, n).astype(np.float16), dtype=np.uint16)
        t16 = bench(lambda: rt.sgemm_fp16(a16, b16))
        print("%-10s %-20s %10.1f %10.1f %7.2fx  (%.1f -> %.1f TFLOPS)"
              % (name, "%dx%d@%dx%d" % (m, k, k, n), t32 * 1e3, t16 * 1e3, t32 / t16,
                 fl / t32 / 1e12, fl / t16 / 1e12), flush=True)
    except Exception as exc:
        msg = str(exc).encode("ascii", "replace").decode()[:110]
        print("%-10s fp16 失败: %s" % (name, msg), flush=True)

# torch 参照
for name, m, k, n in SHAPES:
    a16 = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b16 = torch.randn(k, n, device="cuda", dtype=torch.float16)
    a32 = torch.randn(m, k, device="cuda")
    b32 = torch.randn(k, n, device="cuda")
    f16 = bench(lambda: a16 @ b16)
    f32 = bench(lambda: a32 @ b32)
    print("torch %-8s fp32=%.1f ms  fp16=%.1f ms  (%.2fx)"
          % (name, f32 * 1e3, f16 * 1e3, f32 / f16), flush=True)
