# -*- coding: utf-8 -*-
"""P100 公平对照：把「精度」和「注意力路径」两个轴拆开，避免把配置差异当成引擎差距。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
为什么需要这份脚本（2026-11-21，读了你原来的 bench_p100_full.py / bench_v3exp2.py 之后）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
那两份脚本里有**两处不对等**，而且都恰好压着 ORV：

1) 精度不等  —— torch 侧 `FP32` 默认 "0" ⇒ fp16 autocast + GradScaler；
   ORV 侧 `AMP` 默认 "0"、策略默认 standard ⇒ **fp32**。
   fp16 激活只有一半大小，P100 上 fp16 GEMM 还有 1:2 速率优势。
   ⇒ 那张表其实是 "fp16 PyTorch vs fp32 ORV"。

2) 注意力路径不等 —— 物化 `[B*heads, S, S]` 分数矩阵：
   * ORV 侧 bench_p100_full.py 里写死 `lowmem=False` ⇒ **强制物化**；
   * torch 侧 `F.scaled_dot_product_attention` 在 P100 上通常走
     memory-efficient 后端 ⇒ **不物化**。
   S=2048、18 头、B=4 时，物化一份分数就是约 4×18×2048²×4B ≈ 1.2 GB/层。

另有几处小的：ORV 每步内部 `rt.sync()`（torch 只在循环外 sync）、只 warmup 1 步
（ORV 首步含 NVRTC 编译，会明显拉高均值）、以及两边显存口径不同
（`peak_live_bytes` vs `max_memory_allocated`，两者都不等于"16 GB 装不装得下"）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
这个脚本做了什么
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
同一台机器、同一骨架、同 batch/序列，跑 2×2：

              低显存注意力        物化注意力
    fp32        A                  B
    fp16        C                  D

* ORV：显式传 `lowmem=True/False`（**不依赖 set_policy**，因为旧版 ORV 的策略键
  `attn` 是死键、设了也不生效；新版已修，但显式传参在任何版本都正确）。
* torch：用 `torch.backends.cuda.sdp_kernel` 强制指定后端。
* 梯度重算、CE 分块：两边**同开或同关**。
* 显存给三个数：框架 live 峰值 / 框架 reserved（ORV 是池占用）/ **nvidia-smi 实际占用**
  —— 只有最后一个是"能不能装进 16 GB"的答案。
* 计时：warmup 2 步不计，计时 `--iters` 步，**循环外统一 sync**，不在每步内 sync。

用法
    python bench_fair.py --probe                 # 先看 torch 的 SDPA 后端可用性
    python bench_fair.py --bs 2,4 --iters 5      # 跑完整 2×2
    python bench_fair.py --bs 4 --only orv --attns low --dtypes fp16
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

V, D, S = 36864, 1152, 2048
L, NH, DH, FF = 21, 18, 64, 2560


# ----------------------------------------------------------- 显存：nvidia-smi 口径
def smi_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        vals = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
        return max(vals) if vals else None
    except Exception:                             # noqa: BLE001
        return None


class PeakSmi(object):
    def __init__(self):
        self.best = 0

    def sample(self):
        v = smi_used_mib()
        if v and v > self.best:
            self.best = v
        return self.best


# ------------------------------------------------------------------------- ORV
def run_orv(bs, dtype, attn, iters, ckpt=True, ce_chunk=0):
    import tgai_go.train as T
    from tgai_go.train import functional as Fn

    if dtype == "fp16":
        os.environ["TG_FP16"] = "1"
    else:
        os.environ.pop("TG_FP16", None)

    T.set_default_device("cuda")
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

    lowmem = True if attn == "low" else False

    def blk(x, l):
        z = x.reshape(-1, D)
        q = (z @ l["q"].T).reshape(bs, S, NH, DH).permute(0, 2, 1, 3)
        kv = z @ l["kv"].T
        k = (kv @ l["ku"].T).reshape(bs, S, NH, DH).permute(0, 2, 1, 3)
        v = (kv @ l["vu"].T).reshape(bs, S, NH, DH).permute(0, 2, 1, 3)
        y = Fn.scaled_dot_product_attention(q.reshape(bs * NH, S, DH),
                                            k.reshape(bs * NH, S, DH),
                                            v.reshape(bs * NH, S, DH),
                                            is_causal=True, lowmem=lowmem)
        y = y.reshape(bs, NH, S, DH).permute(0, 2, 1, 3).reshape(-1, D)
        x = x + (y @ l["o"].T).reshape(bs, S, D)
        f = x.reshape(-1, D)
        return x + (Fn.silu(f @ l["w1"].T) @ l["w2"].T).reshape(bs, S, D)

    def body():
        x = emb(ids.reshape(-1)).reshape(bs, S, D)
        for l in layers:
            x = T.checkpoint(blk, x, l) if ckpt else blk(x, l)
        # CE：分块（两边同口径），V 很大时这一项能占 1 GiB+
        if ce_chunk and hasattr(Fn, "linear_cross_entropy"):
            loss = Fn.linear_cross_entropy(x.reshape(-1, D), head, tgt.reshape(-1),
                                           chunk=ce_chunk)
        else:
            loss = Fn.cross_entropy(x.reshape(-1, D) @ head.T, tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    def step():
        if dtype == "fp16" and hasattr(T, "amp"):
            with T.amp():
                body()
        else:
            body()

    own = PeakSmi()
    for _ in range(2):                     # warmup：把 NVRTC 编译排除在计时外
        step()
    rt.sync()
    t0 = time.time()
    for _ in range(iters):
        step()
        own.sample()
    rt.sync()                              # 只在循环外 sync 一次
    dt = (time.time() - t0) / iters
    st = rt.pool_stats() if hasattr(rt, "pool_stats") else {}
    return {"impl": "orv", "dtype": dtype, "attn": attn, "bs": bs,
            "s_per_step": dt, "tok_s": bs * S / dt,
            "peak_live_gib": st.get("peak_live_bytes", 0) / 2 ** 30,
            "peak_reserved_gib": st.get("peak_bytes", 0) / 2 ** 30,
            "peak_smi_gib": own.best / 1024.0 if own.best else None}


# ----------------------------------------------------------------------- torch
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

    def loss_of(x):
        flat = x.reshape(-1, D)
        if ce_chunk:                       # 分块算 logits+CE（与 ORV 侧同口径）
            total = None
            n = flat.shape[0]
            w = 1.0 / float(n)
            for i in range(0, n, ce_chunk):
                part = F.cross_entropy(flat[i:i + ce_chunk] @ head,
                                       tgt.reshape(-1)[i:i + ce_chunk])
                total = part * (min(ce_chunk, n - i) * w) if total is None \
                    else total + part * (min(ce_chunk, n - i) * w)
            return total
        return F.cross_entropy(flat @ head, tgt.reshape(-1))

    def body():
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dtype == "fp16")):
            x = emb(ids)
            for l in layers:
                x = (torch.utils.checkpoint.checkpoint(blk, x, l, use_reentrant=False)
                     if ckpt else blk(x, l))
            loss = loss_of(x)
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
    return {"impl": "torch", "dtype": dtype, "attn": attn, "bs": bs,
            "s_per_step": dt, "tok_s": bs * S / dt,
            "peak_live_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
            "peak_smi_gib": own.best / 1024.0 if own.best else None}


def probe():
    import torch
    if not torch.cuda.is_available():
        print("这台机器 torch 没有 CUDA（torch=%s cuda=%s）。"
              "修法：在有 N 卡的机器跑，或装 CUDA 版 torch。" % (torch.__version__,
                                                              torch.version.cuda))
        return 2
    caps = torch.cuda.get_device_capability(0)
    print("torch %s | cuda %s | %s | cc=%d.%d（<7.0 = 无 Tensor Core）"
          % (torch.__version__, torch.version.cuda,
             torch.cuda.get_device_name(0), caps[0], caps[1]))
    for name, flags in (("mem_efficient", dict(enable_flash=False, enable_math=False,
                                               enable_mem_efficient=True)),
                        ("math", dict(enable_flash=False, enable_math=True,
                                      enable_mem_efficient=False))):
        try:
            with torch.backends.cuda.sdp_kernel(**flags):
                q = torch.randn(4, 64, 512, device="cuda", dtype=torch.float32)
                torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
            print("  SDPA backend %-14s 可用" % name)
        except Exception as e:                    # noqa: BLE001
            print("  SDPA backend %-14s 不可用: %s" % (name, str(e)[:90]))
    try:
        import tgai_go
        import tgai_go.train as T
        print("ORV %s | native_available=%s" % (tgai_go.__version__,
                                                tgai_go.native_available()))
        pol = T.policy() if hasattr(T, "policy") else {}
        print("  默认策略: %s" % (json.dumps(pol, ensure_ascii=False),))
    except Exception as e:                        # noqa: BLE001
        print("ORV 探测失败: %s" % (str(e)[:120],))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="P100 公平对照：精度 × 注意力路径 2×2")
    ap.add_argument("--bs", default="2", help="逗号分隔 batch 列表，如 2,4")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--dtypes", default="fp32,fp16")
    ap.add_argument("--attns", default="low,mat", help="low=不物化注意力, mat=物化")
    ap.add_argument("--only", default=None, choices=("orv", "torch"))
    ap.add_argument("--no-ckpt", action="store_true", help="两边都关掉梯度重算")
    ap.add_argument("--ce-chunk", type=int, default=0, help="CE 分块行数（两边同值，0=不分块）")
    ap.add_argument("--json", default=None, help="把结果追加写到这个 json 文件")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args(argv)

    if a.probe:
        return probe() or 0

    rows = []
    impls = ["orv"] if a.only == "orv" else ["torch"] if a.only == "torch" \
        else ["torch", "orv"]
    for bs in [int(x) for x in a.bs.split(",")]:
        for dtype in a.dtypes.split(","):
            for attn in a.attns.split(","):
                for impl in impls:
                    fn = run_orv if impl == "orv" else run_torch
                    try:
                        r = fn(bs, dtype, attn, a.iters, ckpt=not a.no_ckpt,
                               ce_chunk=a.ce_chunk)
                    except Exception as e:        # noqa: BLE001
                        print("  %-5s bs=%d %-5s attn=%-3s FAILED: %s"
                              % (impl, bs, dtype, attn, str(e)[:110]), flush=True)
                        continue
                    rows.append(r)
                    print("  %-5s bs=%d %-5s attn=%-3s | %6.2f s/step %6.0f tok/s"
                          " | live %5.2f reserved %5.2f smi %s GiB"
                          % (r["impl"], bs, dtype, attn, r["s_per_step"], r["tok_s"],
                             r["peak_live_gib"], r["peak_reserved_gib"],
                             ("%5.2f" % r["peak_smi_gib"]) if r["peak_smi_gib"] else "  n/a"),
                          flush=True)

    if not rows:
        return 1

    print("\n== 同配置对比（R = ORV/torch 步时比；<1 表示 ORV 更快）==")
    for bs in sorted({r["bs"] for r in rows}):
        for dtype in ("fp32", "fp16"):
            for attn in ("low", "mat"):
                sel = {r["impl"]: r for r in rows
                       if r["bs"] == bs and r["dtype"] == dtype and r["attn"] == attn}
                if len(sel) == 2:
                    o, t = sel["orv"], sel["torch"]
                    print("  bs=%-2d %-4s attn=%-3s | ORV %6.2fs vs torch %6.2fs  R=%.2f"
                          " | 显存 ORV %5.2f vs torch %5.2f GiB"
                          % (bs, dtype, attn, o["s_per_step"], t["s_per_step"],
                             o["s_per_step"] / t["s_per_step"],
                             o["peak_live_gib"], t["peak_live_gib"]))

    if a.json:
        old = []
        if os.path.isfile(a.json):
            try:
                with open(a.json, encoding="utf-8") as fh:
                    old = json.load(fh)
            except Exception:                     # noqa: BLE001
                old = []
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(old + rows, fh, ensure_ascii=False, indent=2)
        print("\n结果已追加写入 %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
