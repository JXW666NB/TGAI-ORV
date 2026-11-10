# -*- coding: utf-8 -*-
"""决定性实验：**纯 PyTorch**，28 层 3B MoE，T=512，真 AdamW（fp32 动量），看能否在 32GB 上跑完一步。

这是在把"PyT 真 AdamW 需要 ~48GB"从算术变成实测。
分两段测峰值：① 只做 fwd+bwd（不含优化器状态）；② 建 AdamW 后走完一步。

运行（需独占 GPU）：python pyt_adamw_oom.py [层数]
"""
import sys

import numpy as np

D, NH, DF, NE, TOPK = 1024, 16, 4096, 8, 2
DH = D // NH
SEQ = 512
NL = int(sys.argv[1]) if len(sys.argv) > 1 else 28
GIB = float(1 << 30)


def log(m):
    print(m, flush=True)


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as ckpt

    dev = "cuda"
    torch.manual_seed(0)
    log("纯 PyTorch：28 层 MoE，seq=%d，fp32 + gradient checkpointing + 真 AdamW" % SEQ)

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(D, D, bias=False)
            self.k = nn.Linear(D, D, bias=False)
            self.v = nn.Linear(D, D, bias=False)
            self.o = nn.Linear(D, D, bias=False)

        def forward(self, x):
            S = x.shape[0]
            q = self.q(x).view(S, NH, DH).permute(1, 0, 2)
            k = self.k(x).view(S, NH, DH).permute(1, 0, 2)
            v = self.v(x).view(S, NH, DH).permute(1, 0, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.o(a.permute(1, 0, 2).reshape(S, D))

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.ModuleList([nn.Linear(D, DF, bias=False) for _ in range(NE)])
            self.w2 = nn.ModuleList([nn.Linear(DF, D, bias=False) for _ in range(NE)])

        def forward(self, x):
            out = None
            for e in range(NE):
                y = self.w2[e](F.silu(self.w1[e](x)))
                out = y if out is None else out + y
            return out * (TOPK / float(NE))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.RMSNorm(D)
            self.att = Attn()
            self.n2 = nn.RMSNorm(D)
            self.moe = MoE()

        def forward(self, x):
            x = x + ckpt(self.att, self.n1(x), use_reentrant=False)
            return x + ckpt(self.moe, self.n2(x), use_reentrant=False)

    model = nn.ModuleList([Block() for _ in range(NL)]).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    log("参数量 %.3f B（权重 %.2f GiB）" % (npar / 1e9, npar * 4 / GIB))
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(SEQ, D, device=dev)
    gw = torch.randn(SEQ, D, device=dev)
    try:
        for m in model:
            x = m(x)
        (x * gw).sum().backward()
        p1 = torch.cuda.max_memory_allocated() / GIB
        log("① 仅 fwd+bwd 峰值 **%.2f GiB**" % p1)
    except Exception as exc:
        log("① fwd+bwd 就失败：%s: %s" % (type(exc).__name__, str(exc)[:130]))
        return

    torch.cuda.reset_peak_memory_stats()
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        log("② AdamW 已建（m/v 需要 %.2f GiB）" % (npar * 8 / GIB))
        st = opt.state_dict()
        opt.zero_grad(set_to_none=True)
        x = torch.randn(SEQ, D, device=dev)
        for m in model:
            x = m(x)
        (x * gw).sum().backward()
        opt.step()
        p2 = torch.cuda.max_memory_allocated() / GIB
        log("② 整步（fwd+bwd+AdamW）峰值 **%.2f GiB** ⇒ 装得下" % p2)
    except Exception as exc:
        msg = str(exc)[:200].replace("\n", " ")
        log("② 整步失败 ⇒ **装不下**：%s: %s" % (type(exc).__name__, msg))
        log("   理论需求 ≈ 权重 %.1f + fp32 动量 %.1f + fp32 梯度 %.1f = **%.1f GiB**（卡的可用为 31.86）"
            % (npar * 4 / GIB, npar * 8 / GIB, npar * 4 / GIB, npar * 16 / GIB))


if __name__ == "__main__":
    main()
