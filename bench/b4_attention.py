# -*- coding: utf-8 -*-
"""b4：注意力专项基准 —— 物化 vs 融合，ORV vs PyTorch，在"群友形状"上。

形状取自 P100 对照（d=1152 / 18 头 × 64 / S=2048 / B=4），只测注意力本身，
把 FFN 与头部的干扰排除掉。

判读要点：
  · 融合档相对物化档应显著省显存；若速度没优势，说明内核本身有效率问题
  · 与 torch 的同档对比才公平（物化 vs math、融合 vs 默认）
"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import functional as Fn  # noqa: E402


def bench_orv(BH, S, DH, lowmem, reps=5):
    q = T.tensor(np.random.default_rng(0).normal(0, .3, (BH, S, DH)).astype(np.float32), requires_grad=True)
    k = T.tensor(np.random.default_rng(1).normal(0, .3, (BH, S, DH)).astype(np.float32), requires_grad=True)
    v = T.tensor(np.random.default_rng(2).normal(0, .3, (BH, S, DH)).astype(np.float32), requires_grad=True)
    rt = T._cuda()

    def one():
        y = Fn.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=lowmem)
        (y * y).sum().backward()

    one()
    rt.sync()
    rt.empty_cache()
    t0 = time.time()
    for _ in range(reps):
        one()
    rt.sync()
    return (time.time() - t0) / reps, rt.pool_stats()["peak_live_bytes"] / 2**20


def bench_torch(BH, S, DH, backend, reps=5):
    import torch
    import torch.nn.functional as F
    if backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(BH, S, DH, device=dev, generator=g) * 0.3
    k = torch.randn(BH, S, DH, device=dev, generator=g) * 0.3
    v = torch.randn(BH, S, DH, device=dev, generator=g) * 0.3
    for t in (q, k, v):
        t.requires_grad_(True)

    def one():
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        (y * y).sum().backward()

    one()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(reps):
        one()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    return dt, torch.cuda.max_memory_allocated() / 2**20


def main():
    T.set_default_device("cuda")
    B, S, NH, DH = 4, 2048, 18, 64
    BH = B * NH
    print("注意力形状：BH=%d S=%d DH=%d（b=4, 18 头）" % (BH, S, DH), flush=True)

    try:
        dt, mem = bench_orv(BH, S, DH, lowmem=False)
        print("  ORV  物化   %6.3f s   peak %.2f GB" % (dt, mem / 1024), flush=True)
    except Exception as exc:
        print("  ORV  物化   失败：%s" % str(exc)[:100], flush=True)

    try:
        dt, mem = bench_orv(BH, S, DH, lowmem=True)
        print("  ORV  融合   %6.3f s   peak %.2f GB" % (dt, mem / 1024), flush=True)
    except Exception as exc:
        print("  ORV  融合   失败：%s" % str(exc)[:100], flush=True)

    import torch  # noqa: F401
    dt, mem = bench_torch(BH, S, DH, "math")
    print("  torch 物化  %6.3f s   peak %.2f GB" % (dt, mem / 1024), flush=True)
    dt, mem = bench_torch(BH, S, DH, "default")
    print("  torch 融合  %6.3f s   peak %.2f GB" % (dt, mem / 1024), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
