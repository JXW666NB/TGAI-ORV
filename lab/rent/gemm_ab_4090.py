# -*- coding: utf-8 -*-
"""在 4090 上比对真实形状的 fp16 vs fp32 GEMM（都走 cuBLAS），定位 AMP 变慢的原因。"""
import time

import torch

SHAPES = [
    ("qkv  (512,1024)x(1024,3328)", 512, 1024, 3328),
    ("down (512,3328)x(3328,1024)", 512, 3328, 1024),
    ("head (511,1024)x(1024,96000)", 511, 1024, 96000),
    ("csa  (512,1024)x(1024,512)", 512, 1024, 512),
]


def bench(dtype, m, k, n, iters=50):
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    for _ in range(5):
        c = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


print("%-32s %10s %10s %8s" % ("shape", "fp32(ms)", "fp16(ms)", "fp16/fp32"))
for name, m, k, n in SHAPES:
    t32 = bench(torch.float32, m, k, n) * 1e3
    t16 = bench(torch.float16, m, k, n) * 1e3
    print("%-32s %10.3f %10.3f %8.2f" % (name, t32, t16, t16 / t32))

print("\n=== fp16 输入 + fp32 输出（ORV 的 mixed 模式，torch 里用 out= 模拟）===")
for name, m, k, n in SHAPES:
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    out = torch.empty(m, n, device="cuda", dtype=torch.float32)
    for _ in range(5):
        torch.mm(a, b, out=out)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        torch.mm(a, b, out=out)
    torch.cuda.synchronize()
    print("%-32s %10.3f ms" % (name, (time.perf_counter() - t0) / 50 * 1e3))

print("\n=== cast 成本：848M 参数 fp32<->fp16 一次 ===")
x = torch.randn(848_645_257 // 4, 4, device="cuda", dtype=torch.float32)
for _ in range(2):
    y = x.half()
torch.cuda.synchronize()
t0 = time.perf_counter()
y = x.half()
torch.cuda.synchronize()
print("  fp32->fp16 848M: %.2f ms" % ((time.perf_counter() - t0) * 1e3))
