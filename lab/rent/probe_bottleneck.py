# -*- coding: utf-8 -*-
"""定位瓶颈：GPU 算力本身够不够？还是主机侧被限流？"""
import json
import os
import subprocess
import time

import torch

print("=== cgroup CPU 配额 ===")
for p in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
          "/sys/fs/cgroup/cpu/cpu.cfs_period_us"):
    try:
        print(" ", p, "=", open(p).read().strip())
    except Exception:                                    # noqa: BLE001
        print(" ", p, "= (读不到)")

print("=== 主机名/负载 ===")
for cmd in (["nproc"], ["cat", "/proc/loadavg"]):
    try:
        print(" ", subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=10).stdout.strip())
    except Exception:                                    # noqa: BLE001
        pass

print("=== GPU 纯算力：fp16 GEMM 4096^3 ===")
torch.backends.cuda.matmul.allow_tf32 = True
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(3):
    c = a @ b
torch.cuda.synchronize()
t0 = time.perf_counter()
N = 20
for _ in range(N):
    c = a @ b
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / N
tflops = 2 * 4096 ** 3 / dt / 1e12
print("  fp16 GEMM: %.4f s/次 -> %.1f TFLOP/s" % (dt, tflops))

print("=== fp32 GEMM（TF32）===")
a32 = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
b32 = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
for _ in range(3):
    c = a32 @ b32
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    c = a32 @ b32
torch.cuda.synchronize()
dt32 = (time.perf_counter() - t0) / N
print("  fp32 GEMM: %.4f s/次 -> %.1f TFLOP/s" % (dt32, 2 * 4096 ** 3 / dt32 / 1e12))

print("=== 主机侧 dispatch 成本：1000 次小 kernel 发射 ===")
x = torch.zeros(1024, device="cuda")
t0 = time.perf_counter()
for _ in range(1000):
    x.add_(1.0)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 1000
print("  每次 add_ 往返: %.1f us" % (dt * 1e6))

print("=== Python 纯 CPU 循环（不受 GPU 影响）===")
t0 = time.perf_counter()
s = 0
for i in range(3_000_000):
    s += i & 7
dt = time.perf_counter() - t0
print("  300 万次简单循环: %.3f s" % dt)

print("=== ctypes 调用成本（ORV 的发射路径）===")
import ctypes
libc = ctypes.CDLL("libc.so.6")
t0 = time.perf_counter()
for _ in range(200_000):
    libc.getpid()
dt = time.perf_counter() - t0
print("  20 万次 ctypes 调用: %.3f s (%.2f us/次)" % (dt, dt / 200_000 * 1e6))
