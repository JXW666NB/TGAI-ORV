# -*- coding: utf-8 -*-
"""判据 3 要求的头对头：T=128/512 × fp32/AMP，两种栈各自**独立进程**，报 ms 与峰值显存。

**为什么一个进程只跑一个栈**：同一进程里先后跑 ORV 和 PyT，前面的显存池不会干净归还，
后者的"峰值"会被前者的缓存污染（这个坑上一轮已经踩过一次，得到过反向的结论）。

用法（每种组合单独一次调用）：
    python h2h_table.py pyt  128 fp32
    python h2h_table.py pyt  128 amp
    python h2h_table.py orv  128 fp32
    ...

结构：8 层、d_model=1024、16 头、d_ff=4096、8 专家 top-2（与 V2.7 同构，只是层数少）。
"""
import json
import sys
import time as _time

import numpy as np

ORV_PY = r"D:\tgai_orv\TGAI-ORV-main\python"
sys.path.insert(0, ORV_PY)

D, NH, DF, NE, TOPK = 1024, 16, 4096, 8, 2
DH = D // NH
NL = 8


def log(m):
    print(m, flush=True)


# ═══════════════════ PyTorch 栈 ═══════════════════
def run_pyt(seq, amp):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
    dev = "cuda"

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(D, D, bias=False)
            self.k = nn.Linear(D, D, bias=False)
            self.v = nn.Linear(D, D, bias=False)
            self.o = nn.Linear(D, D, bias=False)

        def forward(self, x):
            B, S, _ = x.shape
            q = self.q(x).view(B, S, NH, DH).transpose(1, 2)
            k = self.k(x).view(B, S, NH, DH).transpose(1, 2)
            v = self.v(x).view(B, S, NH, DH).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(B, S, D)
            return self.o(a)

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(D, NE, bias=False)
            self.w1 = nn.ModuleList([nn.Linear(D, DF, bias=False) for _ in range(NE)])
            self.w2 = nn.ModuleList([nn.Linear(DF, D, bias=False) for _ in range(NE)])

        def forward(self, x):
            B, S, _ = x.shape
            logits = self.gate(x)
            w, idx = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
            out = torch.zeros_like(x)
            for e in range(NE):
                m = (idx == e).any(-1)
                if not m.any():
                    continue
                ww = (idx == e).float() * w
                ww = ww.sum(-1, keepdim=True)
                y = self.w2[e](F.silu(self.w1[e](x))) * ww
                out = out + torch.where(m.unsqueeze(-1), y, torch.zeros_like(y))
            return out

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.RMSNorm(D)
            self.att = Attn()
            self.n2 = nn.RMSNorm(D)
            self.moe = MoE()

        def forward(self, x):
            # CausalSelfAttention 要 3D [B,S,D]；这里 batch=1，包一层再拆掉
            S = int(x.shape[0])
            y = self.att(self.n1(x).reshape(1, S, D)).reshape(S, D)
            x = x + y
            return x + self.moe(self.n2(x))

    model = nn.Sequential(*[Block() for _ in range(NL)]).to(dev)
    x = torch.randn(1, seq, D, device=dev, requires_grad=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    torch.cuda.reset_peak_memory_stats()

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(dev, dtype=torch.float16, enabled=amp):
            y = model(x)
            loss = (y * y).mean()
        loss.backward()
        opt.step()

    step(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(3):
        step()
    e1.record(); e1.synchronize()
    ms = e0.elapsed_time(e1) / 3.0
    peak = torch.cuda.max_memory_allocated() / (1 << 30)
    log(json.dumps({"stack": "pyt", "seq": seq, "amp": amp, "ms": round(ms, 1),
                    "peak_gib": round(peak, 2)}))


# ═══════════════════ ORV 栈 ═══════════════════
def run_orv(seq, amp):
    import torch
    import tgai_go.train as T

    T.set_default_device("cuda")
    rt = T._cuda()
    total = rt.total_device_bytes() / (1 << 30) if hasattr(rt, "total_device_bytes") else 31.86
    low = [total]

    def sample():
        f = rt.free_device_bytes() / (1 << 30)
        low[0] = min(low[0], f)

    class MoE(T.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = T.nn.Linear(D, NE, bias=False)
            self.w1 = T.nn.ModuleList([T.nn.Linear(D, DF, bias=False) for _ in range(NE)])
            self.w2 = T.nn.ModuleList([T.nn.Linear(DF, D, bias=False) for _ in range(NE)])

        def forward(self, x):
            S = int(x.shape[0])
            out = None
            for e in range(NE):
                y = self.w2[e](T.functional.silu(self.w1[e](x)))
                out = y if out is None else out + y
            return out * (TOPK / float(NE))

    class Block(T.nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = T.nn.RMSNorm(D)
            self.att = T.nn.CausalSelfAttention(D, NH)
            self.n2 = T.nn.RMSNorm(D)
            self.moe = MoE()

        def forward(self, x):
            # CausalSelfAttention 要 3D [B,S,D]；这里 batch=1，包一层再拆掉
            S = int(x.shape[0])
            y = self.att(self.n1(x).reshape(1, S, D)).reshape(S, D)
            x = x + y
            return x + self.moe(self.n2(x))

    model = T.nn.Sequential(*[Block() for _ in range(NL)])
    params = list(model.parameters())
    x = T.tensor(np.random.default_rng(0).normal(0, 1, (seq, D)).astype(np.float32))
    opt = T.optim.AdamW(params, lr=1e-5)
    sample()

    def step():
        y = model(x)
        loss = (y * y).mean()
        loss.backward()
        opt.step()

    if amp:
        with T.amp():
            step()
    else:
        step()
    rt.sync()

    rep = 3
    rt.sync()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(rep):
        if amp:
            with T.amp():
                step()
        else:
            step()
    e1.record(); e1.synchronize()
    ms = e0.elapsed_time(e1) / rep
    sample()
    log(json.dumps({"stack": "orv", "seq": seq, "amp": amp, "ms": round(ms, 1),
                    "peak_gib": round(total - low[0], 2)}))


if __name__ == "__main__":
    stack = sys.argv[1] if len(sys.argv) > 1 else "orv"
    seq = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    amp = (sys.argv[3] if len(sys.argv) > 3 else "fp32").lower() in ("amp", "fp16", "1", "true")
    (run_pyt if stack == "pyt" else run_orv)(seq, amp)
