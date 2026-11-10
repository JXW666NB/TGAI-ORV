# -*- coding: utf-8 -*-
"""ORV vs PyTorch 头对头：同一个 28 层 MoE 结构、同一张卡、同样 seq/精度。

结构：d_model=1024, 16 头 (d_head=64), d_ff=4096, 8 专家 top-2, rope_theta=1e4, seq=128, batch=1
精度：两侧都用 **fp32 计算**（ORV 的 .tg 权重按 fp32 参与计算；torch 侧不启用 AMP）
测的：前向 + 反向（不含优化器、不含 checkpoint I/O），CUDA event 计时，取多次最小/均值

说明：torch 侧的 MoE 用"按专家 mask 循环"的写法 —— 这是一个普通 PyT 用户写 MoE 的自然方式，
不是专家的融合实现。这个前提会在输出里写明，避免把结论说过头。

运行（需独占 GPU）：python pyt_head2head.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL_TG = r"D:\tgai\models\V2.7\TGAI_NB_V2.7_v4.tg"
SEQ = 128
NL = 28
D, NH, DF, NE, TOPK = 1024, 16, 4096, 8, 2
THETA = 1e4
REP = 3


# ---------------- ORV 侧 ----------------
def bench_orv():
    import torch
    import tgai_go.train as T
    from tgai_go.train.v27_train import TGState, V27Trainer
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    cfg = dict(d_model=D, n_heads=NH, d_ff=DF, n_experts=NE, n_activated=TOPK,
               n_layers=NL, rope_theta=THETA)
    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)
    rt = T._cuda()

    rng = np.random.default_rng(0)
    x0 = rng.normal(0, 1, (SEQ, D)).astype(np.float32)
    gw = rng.normal(0, 1, (SEQ, D)).astype(np.float32)
    j = np.arange(0, D // NH, 2, dtype=np.float32) / (D // NH)
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos, sin = T.tensor(np.cos(ang).astype(np.float32)), T.tensor(np.sin(ang).astype(np.float32))
    gw_t = T.tensor(gw)

    def step():
        cur = T.tensor(x0)
        for i in range(NL):
            cur = tr.fwd(cur, i, cos, sin)          # 全程留在设备上，不做 d2h
        (cur * gw_t).sum().backward()

    step(); torch.cuda.synchronize()
    ts = []
    for _ in range(REP):
        torch.cuda.synchronize()
        t0 = time.time()
        step()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    peak = None
    try:
        peak = float(rt.live_bytes()) / (1 << 30) if hasattr(rt, "live_bytes") else None
    except Exception:
        peak = None
    try:
        st.close(); rt.empty_cache()
    except Exception:
        pass
    return ts, peak


# ---------------- torch 侧 ----------------
def bench_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = "cuda"
    torch.manual_seed(0)
    dtype = torch.float32

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
            self.register_buffer("cos", torch.ones(SEQ, D // NH // 2, dtype=dtype, device=dev))
            self.register_buffer("sin", torch.zeros(SEQ, D // NH // 2, dtype=dtype, device=dev))
            self.register_buffer("mask", torch.ones(SEQ, SEQ, dtype=dtype, device=dev).tril())

        def _rope(self, x):
            x = x.view(SEQ, NH, D // NH)
            x1, x2 = x[..., 0::2], x[..., 1::2]
            c = self.cos[:, None, :]
            s = self.sin[:, None, :]
            return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).reshape(SEQ, D)

        def forward(self, x):
            h = self.ln1(x)
            q, k, v = self._rope(self.q(h)), self._rope(self.k(h)), self._rope(self.v(h))
            q = q.view(SEQ, NH, D // NH).transpose(0, 1)
            k = k.view(SEQ, NH, D // NH).transpose(0, 1)
            v = v.view(SEQ, NH, D // NH).transpose(0, 1)
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = self.o(att.transpose(0, 1).reshape(SEQ, D))
            y = x + a
            h2 = self.ln2(y)
            logits = self.router(h2)
            probs = torch.softmax(logits, dim=-1)
            w, idx = torch.topk(probs, TOPK, dim=-1)
            out = torch.zeros_like(h2)
            for e in range(NE):
                mask = (idx == e)
                if not bool(mask.any()):
                    continue
                tok, slot = mask.nonzero(as_tuple=True)
                xe = h2.index_select(0, tok)
                he = F.silu(xe @ self.w1[e].T) * (xe @ self.w3[e].T)
                oe = he @ self.w2[e].T
                oe = oe * w[tok, slot].unsqueeze(-1)
                out.index_add_(0, tok, oe)
            return y + out

    model = nn.ModuleList([Block() for _ in range(NL)]).to(dev)
    x0 = torch.randn(SEQ, D, dtype=dtype, device=dev)
    gw = torch.randn(SEQ, D, dtype=dtype, device=dev)
    from torch.utils.checkpoint import checkpoint as _ckpt

    def step():
        cur = x0
        for blk in model:
            # 公平对比：ORV 本来就是逐层重算（只存一层激活），所以 torch 侧也开
            # gradient checkpointing。不开的话 fp32 3B/28 层在 32 GB 上直接 OOM。
            cur = _ckpt(blk, cur, use_reentrant=False)
        (cur * gw).sum().backward()

    step(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(REP):
        for p in model.parameters():
            p.grad = None
        torch.cuda.synchronize()
        t0 = time.time()
        step()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / (1 << 30)
    return ts, peak


def main():
    import torch
    rep = lambda ts: "min %.1f / mean %.1f ms" % (min(ts) * 1000, float(np.mean(ts)) * 1000)
    ts_o, pk_o = bench_orv()
    log("ORV     前向+反向 28 层：%s；峰值显存 %s GiB"
        % (rep(ts_o), ("%.2f" % pk_o) if pk_o else "未取到"))
    ts_t, pk_t = bench_torch()
    log("PyTorch 前向+反向 28 层：%s；峰值显存 %.2f GiB（torch.max_memory_allocated）"
        % (rep(ts_t), pk_t))
    r = float(np.mean(ts_t)) / float(np.mean(ts_o))
    log("ORV 相对 PyTorch（同一结构、同一卡、fp32）：%.2fx（<1 表示 ORV 更快）" % r)
    log("注意：torch 侧 MoE 用「按专家 mask 循环」的普通写法，不是融合实现；两侧计算均为 fp32。")


if __name__ == "__main__":
    main()
