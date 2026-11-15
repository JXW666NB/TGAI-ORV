# -*- coding: utf-8 -*-
"""按模块隔离测量：一步训练拆成 6 块，各自量「峰值增量」并与 torch 对拍。

用法（在装了 tgai_go 与 torch 的机器上）：
    python probe_modules.py <mode> <module> [B] [S]

  mode   : orv | torch
  module : emb | up | attn | ffn | block | ce | all
  B, S   : batch 与序列长度（默认 4 与 2048）

设计要点
--------
* **每块单独一个进程**（或用 --each 逐个跑）：ORV 的 peak 是单调的，同进程里量第二块
  会把第一块的峰值带进来。
* 每块都做「前向 + 反向」，因为显存大头通常在反向。
* 输出统一格式 `MODULE <name> peak_gb <x.xxx>`，方便脚本比对。

本文件**不修改任何被测代码**，只调用公开 API。
"""
import os
import sys
import time

B = int(sys.argv[3]) if len(sys.argv) > 3 else 4
S = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
D = 1024          # 隐层
HD = 64           # head_dim
H = D // HD       # 头数
INNER = 4 * D     # FFN 中间维
V = 36864         # 词表

MODE = sys.argv[1] if len(sys.argv) > 1 else "orv"
MOD = sys.argv[2] if len(sys.argv) > 2 else "all"


# ---------------------------------------------------------------- ORV
def orv_peak(fn):
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    rt.sync()
    base = rt.pool_stats()["live_bytes"]
    fn(T)
    rt.sync()
    st = rt.pool_stats()
    return (st["peak_live_bytes"] - base) / (1 << 30)


def orv_module(name, T):
    import numpy as np
    rng = np.random.default_rng(0)

    def rnd(*shape):
        a = rng.normal(0, 0.02, shape).astype(np.float32)
        return T.from_numpy(a)

    if name == "emb":
        w = rnd(V, 768)
        idx = T.from_numpy(rng.integers(0, V, B * S).astype(np.int64))
        x = T.take_rows(w, idx) if hasattr(T, "take_rows") else w.embed(idx)
        (x.sum()).backward()

    elif name == "up":
        w = rnd(768, D)
        x = rnd(B * S, 768)
        y = x @ w
        (y.sum()).backward()

    elif name == "attn":
        q = rnd(B * H, S, HD); k = rnd(B * H, S, HD); v = rnd(B * H, S, HD)
        y = T.Fn.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=True)
        (y.sum()).backward()

    elif name == "ffn":
        w1 = rnd(INNER, D); w2 = rnd(D, INNER)
        x = rnd(B * S, D)
        y = T.Fn.silu(x @ w1) @ w2
        (y.sum()).backward()

    elif name == "block":
        wq = rnd(D, D); wk = rnd(D, D); wv = rnd(D, D); wo = rnd(D, D)
        w1 = rnd(INNER, D); w2 = rnd(D, INNER)
        x = rnd(B, S, D)
        f = x.reshape(-1, D)
        q = (f @ wq).reshape(B * H, S, HD)
        k = (f @ wk).reshape(B * H, S, HD)
        vv = (f @ wv).reshape(B * H, S, HD)
        y = T.Fn.scaled_dot_product_attention(q, k, vv, is_causal=True, lowmem=True)
        y = y.reshape(B, H, S, HD).permute(0, 2, 1, 3).reshape(-1, D)
        x = x + (y @ wo).reshape(B, S, D)
        f2 = x.reshape(-1, D)
        out = x + (T.Fn.silu(f2 @ w1) @ w2).reshape(B, S, D)
        (out.sum()).backward()

    elif name == "ce":
        head = rnd(V, D)
        x = rnd(B * S, D)
        tgt = T.from_numpy(rng.integers(0, V, B * S).astype(np.int64))
        loss = T.Fn.cross_entropy(x @ head, tgt)
        loss.backward()

    else:
        raise SystemExit("unknown module %r" % (name,))


# ---------------------------------------------------------------- torch
def torch_module(name):
    import torch
    import torch.nn.functional as F
    dev = "cuda"
    torch.manual_seed(0)
    g = lambda *s: torch.randn(*s, device=dev, requires_grad=True)

    if name == "emb":
        w = g(V, 768)
        idx = torch.randint(0, V, (B * S,), device=dev)
        y = w[idx]
        y.sum().backward()
    elif name == "up":
        w = g(768, D); x = g(B * S, 768)
        (x @ w).sum().backward()
    elif name == "attn":
        q = g(B * H, S, HD); k = g(B * H, S, HD); v = g(B * H, S, HD)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y.sum().backward()
    elif name == "ffn":
        w1 = g(INNER, D); w2 = g(D, INNER); x = g(B * S, D)
        (F.silu(x @ w1) @ w2).sum().backward()
    elif name == "block":
        wq = g(D, D); wk = g(D, D); wv = g(D, D); wo = g(D, D)
        w1 = g(INNER, D); w2 = g(D, INNER)
        x = g(B, S, D)
        f = x.reshape(-1, D)
        q = (f @ wq).reshape(B * H, S, HD)
        k = (f @ wk).reshape(B * H, S, HD)
        vv = (f @ wv).reshape(B * H, S, HD)
        y = F.scaled_dot_product_attention(q, k, vv, is_causal=True)
        y = y.reshape(B, H, S, HD).permute(0, 2, 1, 3).reshape(-1, D)
        x = x + (y @ wo).reshape(B, S, D)
        f2 = x.reshape(-1, D)
        out = x + (F.silu(f2 @ w1) @ w2).reshape(B, S, D)
        out.sum().backward()
    elif name == "ce":
        head = g(V, D); x = g(B * S, D)
        tgt = torch.randint(0, V, (B * S,), device=dev)
        F.cross_entropy(x @ head, tgt).backward()
    else:
        raise SystemExit("unknown module %r" % (name,))


def main():
    names = ["emb", "up", "attn", "ffn", "block", "ce"] if MOD == "all" else [MOD]
    for nm in names:
        t0 = time.time()
        try:
            if MODE == "orv":
                peak = orv_peak(lambda T, n=nm: orv_module(n, T))
            else:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch_module(nm)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / (1 << 30)
            print("MODULE %-6s peak_gb %.3f  (%.1fs)" % (nm, peak, time.time() - t0), flush=True)
        except Exception as e:  # noqa: BLE001  单块失败不该中断整轮
            print("MODULE %-6s FAILED %s: %s" % (nm, type(e).__name__, str(e)[:120]), flush=True)


if __name__ == "__main__":
    main()
