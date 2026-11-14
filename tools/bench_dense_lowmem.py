# -*- coding: utf-8 -*-
"""[本仓库变体] 可切 lowmem 的版本，源自 bench/external/bench_p100_full.py

原版 P100 上 torch vs TGAI-ORV 的同形状对比（21 层 / d=1152 / 18 头 × 64 / FF=2560 / S=2048）。

为什么用"稠密等价"形状：
  · 我们的真实模型是 MLA + SMS2-MoE（36 专家 + 1 常驻），ORV 里没有对应算子，
    没法逐算子对拍；于是把 MoE 全换成稠密 FFN(2560)，只保留
    "21 层 × (MLA 投影 + 注意力 + FFN)" 的公共骨架。
  · 这样两边跑的是**同一份手写实现**，差异只来自框架本身。

⚠ 注意：ORV 对 MoE 有专门优化，对稠密 FFN 没有；本脚本测的是稠密路径，
   所以 ORV 的 MoE 优势在这个基准里体现不出来（它的真实强项不在这里）。

运行（Windows / P100，torch 2.7.1+cu118，tgai-orv 0.9.64）：
  set HB=4            &  REM batch（默认 8）
  set FP32=1          &  REM torch 侧用 fp32（默认 fp16 autocast）
  set POL=standard    &  REM ORV 档位：standard,speed,extreme,memory
  set AMP=1           &  REM ORV 侧开 T.amp()（只 cast GEMM 输入）
  python bench_p100_full.py torch|orv|both

实测（P100 16GB，B=4，检查点开）：
  torch fp32  3.33 s/step  2,459 tok/s  7.37 GB
  torch fp16  3.34 s/step  2,459 tok/s  7.37 GB   ← P100 无 Tensor Core，fp16 无收益
  ORV         4.47 s/step  1,832 tok/s  11.33 GB
  ORV 各档位参数相同（lowmem 在 sm_60 需要 68 KB smem > 48 KB 上限，编不出来）
"""
import os
import sys
import time

import numpy as np

V, D, L, NH, DH, FF, S = 36864, 1152, 21, 18, 64, 2560, 2048
B = int(os.environ.get("HB", "8"))


# ─────────────────────────── torch 版 ───────────────────────────
def bench_torch(iters=3, ckpt=True):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
    if os.environ.get("TORCHMATH", "0") == "1":
        # 强制 torch 用 math 后端（物化注意力），与 ORV 的物化档同算法
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    dev = "cuda"
    W = {}
    def p(*shape):
        return nn.Parameter(torch.randn(*shape, device=dev) * 0.02)
    emb = nn.Embedding(V, D).to(dev)
    head = p(D, V)
    layers = []
    for _ in range(L):
        layers.append(dict(
            q=p(D, D), kv=p(D, 256), ku=p(256, D), vu=p(256, D),
            o=p(D, D), w1=p(D, FF), w2=p(FF, D)))
    params = [emb.weight, head] + [v for l in layers for v in l.values()]
    opt = torch.optim.AdamW(params, lr=1e-4)
    _fp32 = os.environ.get("FP32", "0") == "1"
    scaler = torch.amp.GradScaler("cuda", enabled=not _fp32)
    ids = torch.randint(0, V, (B, S), device=dev)
    tgt = torch.randint(0, V, (B, S), device=dev)

    def blk(x, l):
        if os.environ.get("DEBUG", "0") == "1":
            print("    dtype", x.dtype, "autocast", torch.is_autocast_enabled(), flush=True)
        z = x.reshape(-1, D)
        q = (z @ l["q"]).reshape(B, S, NH, DH).transpose(1, 2)
        kv = z @ l["kv"]
        k = (kv @ l["ku"]).reshape(B, S, NH, DH).transpose(1, 2)
        v = (kv @ l["vu"]).reshape(B, S, NH, DH).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + (y.transpose(1, 2).reshape(-1, D) @ l["o"]).reshape(B, S, D)
        f = x.reshape(-1, D)
        return x + (F.silu(f @ l["w1"]) @ l["w2"]).reshape(B, S, D)

    def step():
        if _fp32:
            return _step_body()
        with torch.autocast("cuda", dtype=torch.float16):
            return _step_body()

    def _step_body():
        x = emb(ids)
        for l in layers:
            if ckpt:
                x = torch.utils.checkpoint.checkpoint(blk, x, l, use_reentrant=False)
            else:
                x = blk(x, l)
        loss = F.cross_entropy((x.reshape(-1, D) @ head), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return float(loss)

    for tag, it in (("warm", 1), ("run", iters)):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(it):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / it
        print("  torch %s ckpt=%s B=%d  %.2f s/step  %.0f tok/s  峰值 %.2f GB"
              % ("fp32" if _fp32 else "fp16", ckpt, B, dt, B * S / dt,
                 torch.cuda.max_memory_allocated() / 2**30),
              flush=True)
    return dt


# ─────────────────────────── ORV 版 ───────────────────────────
def bench_orv(policy="standard", iters=3, ckpt=True):
    import tgai_go.train as T
    from tgai_go.train import functional as Fn

    T.set_default_device("cuda")
    try:
        T.set_policy(policy)
    except Exception as e:
        print("    set_policy(%s) 失败: %s" % (policy, e))
    rt = T._cuda()
    rng = np.random.default_rng(0)

    def dev(n, m):
        return T.Tensor._from_gpu(rt.h2d(rng.normal(0, 0.02, (n, m)).astype(np.float32)))

    emb = T.nn.Embedding(V, D)
    head = dev(V, D)
    layers = []
    for _ in range(L):
        layers.append(dict(q=dev(D, D), kv=dev(256, D), ku=dev(D, 256),
                           vu=dev(D, 256), o=dev(D, D),
                           w1=dev(FF, D), w2=dev(D, FF)))
    params = [emb.weight, head] + [v for l in layers for v in l.values()]
    for t in params:
        t.requires_grad = True
    opt = T.optim.AdamW(params, lr=1e-4)
    ids = T.tensor(rng.integers(0, V, (B, S)).astype(np.int64))
    tgt = T.tensor(rng.integers(0, V, (B, S)).astype(np.int64))

    def blk(x, l):
        if os.environ.get("DEBUG", "0") == "1":
            print("    dtype", x.dtype, "autocast", torch.is_autocast_enabled(), flush=True)
        z = x.reshape(-1, D)
        q = (z @ l["q"].T).reshape(B, S, NH, DH).permute(0, 2, 1, 3)
        kv = z @ l["kv"].T
        k = (kv @ l["ku"].T).reshape(B, S, NH, DH).permute(0, 2, 1, 3)
        v = (kv @ l["vu"].T).reshape(B, S, NH, DH).permute(0, 2, 1, 3)
        y = Fn.scaled_dot_product_attention(q.reshape(B * NH, S, DH),
                                            k.reshape(B * NH, S, DH),
                                            v.reshape(B * NH, S, DH),
                                            is_causal=True, lowmem=(os.environ.get("LOWMEM", "0") == "1"))
        y = y.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(-1, D)
        x = x + (y @ l["o"].T).reshape(B, S, D)
        f = x.reshape(-1, D)
        return x + (Fn.silu(f @ l["w1"].T) @ l["w2"].T).reshape(B, S, D)

    def step():
        if os.environ.get("AMP", "0") == "1":
            with T.amp():
                return _step_body()
        return _step_body()

    def _step_body():
        x = emb(ids.reshape(-1)).reshape(B, S, D)
        for l in layers:
            if ckpt:
                x = T.checkpoint(blk, x, l)
            else:
                x = blk(x, l)
        lg = x.reshape(-1, D) @ head.T
        loss = Fn.cross_entropy(lg, tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        rt.sync()
        return float(np.asarray(loss.to_numpy()).reshape(-1)[0])

    for tag, it in (("warm", 1), ("run", iters)):
        try:
            t0 = time.time()
            for _ in range(it):
                step()
            rt.sync()
        except Exception as e:
            print("  ORV[%s] ckpt=%s 失败: %s" % (policy, ckpt, str(e)[:90]),
                  flush=True)
            return None
        dt = (time.time() - t0) / it
        st = rt.pool_stats()
        print("  ORV[%s] ckpt=%s  %.2f s/step  %.0f tok/s  峰值 %.2f GB"
              % (policy, ckpt, dt, B * S / dt,
                 st["peak_live_bytes"] / (1 << 30)), flush=True)
    return dt


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("torch", "both"):
        bench_torch(ckpt=True)
    if what in ("orv", "both"):
        pols = os.environ.get("POL", "standard,speed,extreme,memory").split(",")
        for pol in pols:
            bench_orv(pol, ckpt=True)
