#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P100 公平对照：把"精度"和"注意力路径"两个轴拆开，避免把配置差异当成引擎差距。

## 为什么要这个脚本（2026-11-21）

用户朋友的第一份 P100 对比（`bench_p100_full.py` / `bench_v3exp2.py`）里有**两处不对等**，
而且两处都恰好压着 ORV：

1. **精度不等**：torch 侧 `FP32` 默认是 `"0"` ⇒ 走 **fp16 autocast + GradScaler**；
   ORV 侧 `AMP` 默认 `"0"`、策略默认 `standard` ⇒ 走 **fp32**。
   fp16 的激活只有一半大小，GEMM 在 P100 上也有 1:2 的 fp16 速率优势。
2. **注意力路径不等**：ORV 的 `standard` 策略里 `attn="materialized"`（源码
   `_POLICY_PRESETS`）⇒ 物化 `[B*heads, S, S]` 分数；torch 用
   `F.scaled_dot_product_attention`（P100 上通常会走 memory-efficient 后端，不物化）。
   ORV 自己另有 `memory` 策略（`attn="flash"`）没被用上。

所以旧数字**不能用来判断"引擎强不强"**，只能说明"这组配置下谁快"。

## 这个脚本做什么

在**同一台机器、同一骨架、同一 batch/序列**下跑 2×2 矩阵：

|          | 低显存注意力 | 物化注意力 |
| -------- | ------------ | ---------- |
| **fp32** | A            | B          |
| **fp16** | C            | D          |

* ORV 侧：`policy=memory`（attn=flash）vs `policy=standard`（attn=materialized）；
  fp16 用 `TG_FP16=1`（GEMM）或 `with T.amp()`（整网 AMP，按你的版本能力选）。
* torch 侧：`torch.backends.cuda.sdp_kernel` 强制 memory-efficient vs math（物化）；
  fp32 = 不加 autocast，fp16 = `torch.autocast`。
* **显存统一口径**：两边都报 (a) 各自框架的 live 峰值、(b) 各自框架的 reserved/池占用、
  (c) **`nvidia-smi` 实际占用峰值**（唯一能回答"16 GB 装不装得下"的数字）。
* 计时：先 warmup ≥2 步，再计时 ≥5 步，**循环外统一 sync**（不在每步内 sync）。
* CE 分块：ORV 用 `TG_CE_CHUNK`，torch 用等价的手工分块，避免词表 36864 的
  `[B*S, V]` logits 把两边都顶爆（是否分块必须两边一致）。

## 用法

```bash
# 先确认 torch 的 SDPA 后端（P100 上很关键）
python bench_fair_p100.py --probe

# 跑完整 2x2（推荐 B=2 与 B=4 各一次）
python bench_fair_p100.py --bs 2,4 --iters 5

# 只跑 ORV 侧某个组合（排障用）
python bench_fair_p100.py --bs 2 --only orv --dtype fp32 --attn low
```

## 读数注意

* 若 **C 组（两边都 fp16、都低显存注意力）** 的显存差距明显小于旧报告，
  说明旧差距**主要是配置**，不是引擎设计。
* 若 A 组仍差 20%+，才说明是**实现层**的显存问题，需要 `pool_dump_live()` 归因。
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

V, D, S = 36864, 1152, 2048
L, NH, DH, FF = 21, 18, 64, 2560


# --------------------------------------------------------------- nvidia-smi 口径
def smi_used_mib():
    """当前进程所在 GPU 的实际占用（MiB）。这是"装不装得下"的唯一可信口径。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        vals = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
        return max(vals) if vals else None
    except Exception:                             # noqa: BLE001
        return None


class PeakSmi(object):
    """跑 step 时后台采样 nvidia-smi（1 Hz 太粗，这里用 20 ms）。"""

    def __init__(self):
        self.best = 0

    def sample(self):
        v = smi_used_mib()
        if v and v > self.best:
            self.best = v
        return self.best


# --------------------------------------------------------------------- ORV 侧
def run_orv(bs, dtype, attn, iters, ckpt=True, ce_chunk=0):
    import tgai_go.train as T
    from tgai_go.train import functional as Fn

    if dtype == "fp16":
        os.environ["TG_FP16"] = "1"          # GEMM 走 fp16（P100 支持 fp16 算术）
    else:
        os.environ.pop("TG_FP16", None)
    if ce_chunk:
        os.environ["TG_CE_CHUNK"] = str(ce_chunk)

    T.set_default_device("cuda")
    T.set_policy("memory" if attn == "low" else "standard")
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
    ids = T.tensor(rng.integers(0, V, (bs, S)).astype(np.int64))
    tgt = T.tensor(rng.integers(0, V, (bs, S)).astype(np.int64))

    def blk(x, l):
        z = x.reshape(-1, D)
        q = (z @ l["q"].T).reshape(bs, S, NH, DH).permute(0, 2, 1, 3)
        kv = z @ l["kv"].T
        k = (kv @ l["ku"].T).reshape(bs, S, NH, DH).permute(0, 2, 1, 3)
        v = (kv @ l["vu"].T).reshape(bs, S, NH, DH).permute(0, 2, 1, 3)
        y = Fn.scaled_dot_product_attention(q.reshape(bs * NH, S, DH),
                                            k.reshape(bs * NH, S, DH),
                                            v.reshape(bs * NH, S, DH),
                                            is_causal=True)
        y = y.reshape(bs, NH, S, DH).permute(0, 2, 1, 3).reshape(-1, D)
        x = x + (y @ l["o"].T).reshape(bs, S, D)
        f = x.reshape(-1, D)
        return x + (Fn.silu(f @ l["w1"].T) @ l["w2"].T).reshape(bs, S, D)

    def body():
        x = emb(ids.reshape(-1)).reshape(bs, S, D)
        for l in layers:
            x = T.checkpoint(blk, x, l) if ckpt else blk(x, l)
        lg = x.reshape(-1, D) @ head.T
        loss = Fn.cross_entropy(lg, tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss

    def step():
        if dtype == "fp16" and hasattr(T, "amp"):
            with T.amp():
                body()
        else:
            body()

    own = PeakSmi()
    rt.reset_peak() if hasattr(rt, "reset_peak") else None
    # warmup（不计入）
    for _ in range(2):
        step()
    rt.sync()
    t0 = time.time()
    for _ in range(iters):
        step()
        own.sample()
    rt.sync()
    dt = (time.time() - t0) / iters
    st = rt.pool_stats() if hasattr(rt, "pool_stats") else {}
    return {
        "impl": "orv", "dtype": dtype, "attn": attn, "bs": bs,
        "s_per_step": dt, "tok_s": bs * S / dt,
        "peak_live_gib": st.get("peak_live_bytes", 0) / 2 ** 30,
        "peak_pool_gib": st.get("peak_bytes", 0) / 2 ** 30,
        "peak_smi_gib": own.best / 1024.0 if own.best else None,
    }


# ------------------------------------------------------------------- torch 侧
def run_torch(bs, dtype, attn, iters, ckpt=True, ce_chunk=0):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
    dev = "cuda"
    emb = nn.Embedding(V, D).to(dev)
    head = nn.Parameter(torch.randn(D, V, device=dev) * 0.02)
    layers = []
    for _ in range(L):
        layers.append(dict(q=nn.Parameter(torch.randn(D, D, device=dev) * 0.02),
                           kv=nn.Parameter(torch.randn(256, D, device=dev) * 0.02),
                           ku=nn.Parameter(torch.randn(D, 256, device=dev) * 0.02),
                           vu=nn.Parameter(torch.randn(D, 256, device=dev) * 0.02),
                           o=nn.Parameter(torch.randn(D, D, device=dev) * 0.02),
                           w1=nn.Parameter(torch.randn(FF, D, device=dev) * 0.02),
                           w2=nn.Parameter(torch.randn(D, FF, device=dev) * 0.02)))
    params = [emb.weight, head] + [v for l in layers for v in l.values()]
    opt = torch.optim.AdamW(params, lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "fp16"))
    ids = torch.randint(0, V, (bs, S), device=dev)
    tgt = torch.randint(0, V, (bs, S), device=dev)

    def attn_ctx():
        """强制 SDPA 后端：low = memory-efficient（不物化），mat = math（物化）。"""
        if attn == "low":
            return torch.backends.cuda.sdp_kernel(enable_flash=False,
                                                  enable_math=False,
                                                  enable_mem_efficient=True)
        return torch.backends.cuda.sdp_kernel(enable_flash=False,
                                              enable_math=True,
                                              enable_mem_efficient=False)

    def blk(x, l):
        z = x.reshape(-1, D)
        q = (z @ l["q"]).reshape(bs, S, NH, DH).transpose(1, 2)
        kv = z @ l["kv"]
        k = (kv @ l["ku"]).reshape(bs, S, NH, DH).transpose(1, 2)
        v = (kv @ l["vu"]).reshape(bs, S, NH, DH).transpose(1, 2)
        with attn_ctx():
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + (y.transpose(1, 2).reshape(-1, D) @ l["o"]).reshape(bs, S, D)
        f = x.reshape(-1, D)
        return x + (F.silu(f @ l["w1"]) @ l["w2"]).reshape(bs, S, D)

    def body():
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dtype == "fp16")):
            x = emb(ids)
            for l in layers:
                if ckpt:
                    x = torch.utils.checkpoint.checkpoint(blk, x, l, use_reentrant=False)
                else:
                    x = blk(x, l)
            logits = x.reshape(-1, D) @ head
            loss = F.cross_entropy(logits, tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    own = PeakSmi()
    for _ in range(2):
        body()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        body()
        own.sample()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return {
        "impl": "torch", "dtype": dtype, "attn": attn, "bs": bs,
        "s_per_step": dt, "tok_s": bs * S / dt,
        "peak_live_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "peak_pool_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
        "peak_smi_gib": own.best / 1024.0 if own.best else None,
    }


def probe():
    import torch
    if not torch.cuda.is_available():
        print("这台机器的 torch 没有 CUDA（torch=%s cuda=%s）。"
              "修法：在有 NVIDIA 卡的机器上跑，或装 CUDA 版 torch；"
              "本脚本只做 GPU 对照，CPU 上没有意义。"
              % (torch.__version__, torch.version.cuda))
        return 2
    print("torch %s cuda=%s device=%s" % (torch.__version__, torch.version.cuda,
                                          torch.cuda.get_device_name(0)))
    caps = torch.cuda.get_device_capability(0)
    print("cc=%d.%d（<7.0 表示没有 Tensor Core）" % caps)
    for backend, flags in (("mem_efficient", dict(enable_flash=False, enable_math=False,
                                                 enable_mem_efficient=True)),
                           ("math", dict(enable_flash=False, enable_math=True,
                                         enable_mem_efficient=False))):
        try:
            with torch.backends.cuda.sdp_kernel(**flags):
                q = torch.randn(4, 64, 512, device="cuda", dtype=torch.float32)
                torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
            print("  SDPA backend %-14s 可用" % backend)
        except Exception as e:                    # noqa: BLE001
            print("  SDPA backend %-14s 不可用: %s" % (backend, str(e)[:80]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="P100 公平对照（精度 × 注意力路径 2x2）")
    ap.add_argument("--bs", default="2", help="逗号分隔的 batch 列表")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--dtypes", default="fp32,fp16")
    ap.add_argument("--attns", default="low,mat")
    ap.add_argument("--only", default=None, choices=("orv", "torch"))
    ap.add_argument("--no-ckpt", action="store_true", help="关闭梯度重算（两边同时关）")
    ap.add_argument("--ce-chunk", type=int, default=0, help="CE 分块行数（两边同时，0=不分块）")
    ap.add_argument("--probe", action="store_true", help="只打印环境/SDPA 后端")
    a = ap.parse_args(argv)

    if a.probe:
        return probe() or 0

    rows = []
    for bs in [int(x) for x in a.bs.split(",")]:
        for dtype in a.dtypes.split(","):
            for attn in a.attns.split(","):
                for impl in (["orv"] if a.only == "orv" else
                             ["torch"] if a.only == "torch" else ["torch", "orv"]):
                    fn = run_orv if impl == "orv" else run_torch
                    try:
                        r = fn(bs, dtype, attn, a.iters, ckpt=not a.no_ckpt,
                               ce_chunk=a.ce_chunk)
                    except Exception as e:        # noqa: BLE001
                        print("  %-5s bs=%d %-5s attn=%-3s FAILED: %s"
                              % (impl, bs, dtype, attn, str(e)[:100]))
                        continue
                    rows.append(r)
                    print("  %-5s bs=%d %-5s attn=%-3s | %6.2f s/step %6.0f tok/s"
                          " | live %5.2f pool %5.2f smi %s GiB"
                          % (r["impl"], bs, dtype, attn, r["s_per_step"], r["tok_s"],
                             r["peak_live_gib"], r["peak_pool_gib"],
                             ("%5.2f" % r["peak_smi_gib"]) if r["peak_smi_gib"] else "  n/a"))
    if not rows:
        return 1
    print("\n== 同配置对比（ORV/torch 步时比，<1 表示 ORV 更快）==")
    for bs in sorted({r["bs"] for r in rows}):
        for dtype in sorted({r["dtype"] for r in rows}):
            for attn in sorted({r["attn"] for r in rows}):
                sel = {r["impl"]: r for r in rows
                       if r["bs"] == bs and r["dtype"] == dtype and r["attn"] == attn}
                if len(sel) == 2:
                    o, t = sel["orv"], sel["torch"]
                    print("  bs=%-2d %-5s attn=%-3s  ORV %.2fs vs torch %.2fs  R=%.2f"
                          "  | 显存 ORV %.2f vs torch %.2f GiB"
                          % (bs, dtype, attn, o["s_per_step"], t["s_per_step"],
                             o["s_per_step"] / t["s_per_step"],
                             o["peak_live_gib"], t["peak_live_gib"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
