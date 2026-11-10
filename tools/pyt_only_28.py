# -*- coding: utf-8 -*-
"""**纯 PyTorch** 进程：同结构 28 层 MoE，已开 gradient checkpointing，看能否在 32 GB 上跑通。

这是"ORV 能训 3B MoE、PyT 装不下"这条卖点的决定性证据：本脚本不加载任何 ORV 权重，
所以 OOM 与"两个模型共存"无关。

运行（需独占 GPU）：python pyt_only_28.py
"""
import sys
import time as _time

import numpy as np

D, NH, DF, NE, TOPK = 1024, 16, 4096, 8, 2
DH = D // NH
SEQ = 512
NL = int(sys.argv[1]) if len(sys.argv) > 1 else 28


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as _ckpt

    dtype = torch.float32
    dev = "cuda"
    torch.manual_seed(0)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(D, dtype=dtype, device=dev)
            self.ln2 = nn.LayerNorm(D, dtype=dtype, device=dev)
            self.q = nn.Linear(D, D, dtype=dtype, device=dev)
            self.k = nn.Linear(D, D, dtype=dtype, device=dev)
            self.v = nn.Linear(D, D, dtype=dtype, device=dev)
            self.o = nn.Linear(D, D, dtype=dtype, device=dev)
            self.router = nn.Linear(D, NE, dtype=dtype, device=dev)
            self.w1 = nn.Parameter(torch.randn(NE, DF, D, dtype=dtype, device=dev) * 0.02)
            self.w3 = nn.Parameter(torch.randn(NE, DF, D, dtype=dtype, device=dev) * 0.02)
            self.w2 = nn.Parameter(torch.randn(NE, D, DF, dtype=dtype, device=dev) * 0.02)

        def forward(self, x):
            S = x.shape[0]
            h = self.ln1(x)
            q = self.q(h).view(S, NH, DH).transpose(0, 1)
            k = self.k(h).view(S, NH, DH).transpose(0, 1)
            v = self.v(h).view(S, NH, DH).transpose(0, 1)
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = x + self.o(att.transpose(0, 1).reshape(S, D))
            h2 = self.ln2(y)
            probs = torch.softmax(self.router(h2), dim=-1)
            w, idx = torch.topk(probs, TOPK, dim=-1)
            out = torch.zeros_like(h2)
            for e in range(NE):
                mask = idx == e
                if not bool(mask.any()):
                    continue
                tok, slot = mask.nonzero(as_tuple=True)
                xe = h2.index_select(0, tok)
                he = F.silu(xe @ self.w1[e].T) * (xe @ self.w3[e].T)
                out.index_add_(0, tok, (he @ self.w2[e].T) * w[tok, slot].unsqueeze(-1))
            return y + out

    print("纯 PyTorch：nl=%d seq=%d fp32 + checkpointing" % (NL, SEQ), flush=True)
    try:
        model = nn.ModuleList([Block() for _ in range(NL)]).to(dev)
        x0 = torch.randn(SEQ, D, dtype=dtype, device=dev)
        gw = torch.randn(SEQ, D, dtype=dtype, device=dev)

        def step():
            cur = x0
            for blk in model:
                cur = _ckpt(blk, cur, use_reentrant=False)
            (cur * gw).sum().backward()

        t0 = _time.time()
        step()
        torch.cuda.synchronize()
        dt = (_time.time() - t0) * 1000.0
        peak = torch.cuda.max_memory_allocated() / (1 << 30)
        print("PyT 成功：第一步 %.0f ms，峰值显存 %.2f GiB（总 31.86）" % (dt, peak), flush=True)
    except torch.OutOfMemoryError as e:
        peak = torch.cuda.max_memory_allocated() / (1 << 30) if torch.cuda.is_available() else -1
        print("PyT OOM：峰值已到 %.2f GiB —— %s" % (peak, str(e).split("\n")[0][:90]), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
