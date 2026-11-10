# -*- coding: utf-8 -*-
"""ORV vs PyTorch 头对头（4 层，同结构、同 seq、fp32）：event 与**墙钟**两种口径 + 峰值显存。

为什么两个口径都要：ORV 的 MoE 有一次主机往返（d2h），CUDA event 只测 GPU 时间段，
**测不到主机停顿**；墙钟（含 torch.cuda.synchronize）才是用户体感。所以以墙钟为主口径。

层数取 4：28 层 fp32 两边都放不下 32 GB（权重+梯度 ~24 GB），4 层才可能 apples-to-apples。
显存：torch 用 max_memory_allocated（精确）；ORV 不走 torch 分配器，用步骤前后设备空闲差近似
（池会保留已释放块，故"下降量"≈峰值），输出里注明是近似。

运行（需独占 GPU）：python head2head.py
"""
import sys
import time as _time

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL_TG = r"D:\tgai\models\V2.7\TGAI_NB_V2.7_v4.tg"
NL = 4
SEQS = (128, 512)
D, NH, DF, NE, TOPK, THETA = 1024, 16, 4096, 8, 2, 1e4
DH = D // NH
REP = 3


def bench_orv(seq, tr, cos, sin, x_np, gw_np):
    import torch
    import tgai_go.train as T
    rt = T._cuda()
    gw_t = T.tensor(gw_np)

    def step():
        cur = T.tensor(x_np, requires_grad=True)
        for i in range(NL):
            cur = tr.fwd(cur, i, cos, sin)
        (cur * gw_t).sum().backward()

    def clear():
        for i in range(NL):
            for p in tr.params_of(i):
                p.grad = None

    step(); torch.cuda.synchronize()
    rt.empty_cache(); torch.cuda.synchronize()
    free0 = rt.free_device_bytes()
    per_event, per_wall = [], []
    for _ in range(REP):
        clear()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0 = _time.time()
        e0.record(); step(); e1.record(); e1.synchronize()
        per_wall.append((_time.time() - t0) * 1000.0)
        per_event.append(e0.elapsed_time(e1))
    used = free0 - rt.free_device_bytes()
    return min(per_event), min(per_wall), used


def bench_torch(seq, device="cuda"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as _ckpt
    dtype = torch.float32
    torch.manual_seed(0)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(D, dtype=dtype, device=device)
            self.ln2 = nn.LayerNorm(D, dtype=dtype, device=device)
            self.q = nn.Linear(D, D, dtype=dtype, device=device)
            self.k = nn.Linear(D, D, dtype=dtype, device=device)
            self.v = nn.Linear(D, D, dtype=dtype, device=device)
            self.o = nn.Linear(D, D, dtype=dtype, device=device)
            self.router = nn.Linear(D, NE, dtype=dtype, device=device)
            self.w1 = nn.Parameter(torch.randn(NE, DF, D, dtype=dtype, device=device) * 0.02)
            self.w3 = nn.Parameter(torch.randn(NE, DF, D, dtype=dtype, device=device) * 0.02)
            self.w2 = nn.Parameter(torch.randn(NE, D, DF, dtype=dtype, device=device) * 0.02)

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

    model = nn.ModuleList([Block() for _ in range(NL)]).to(device)
    x0 = torch.randn(seq, D, dtype=dtype, device=device)
    gw = torch.randn(seq, D, dtype=dtype, device=device)

    def step():
        cur = x0
        for blk in model:
            cur = _ckpt(blk, cur, use_reentrant=False)
        (cur * gw).sum().backward()

    step(); torch.cuda.synchronize()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    per_event, per_wall = [], []
    for _ in range(REP):
        for p in model.parameters():
            p.grad = None
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0 = _time.time()
        e0.record(); step(); e1.record(); e1.synchronize()
        per_wall.append((_time.time() - t0) * 1000.0)
        per_event.append(e0.elapsed_time(e1))
    peak = torch.cuda.max_memory_allocated() / (1 << 30)
    return min(per_event), min(per_wall), peak


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.train.v27_train import TGState, V27Trainer
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    cfg = dict(d_model=D, n_heads=NH, d_ff=DF, n_experts=NE, n_activated=TOPK,
               n_layers=int(h["n_layers"]), rope_theta=THETA)
    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)

    rng = np.random.default_rng(0)
    log("层数=%d，结构 d_model=%d heads=%d d_ff=%d experts=%d top%d，fp32，batch=1" % (NL, D, NH, DF, NE, TOPK))
    for seq in SEQS:
        x_np = rng.normal(0, 1, (seq, D)).astype(np.float32)
        gw_np = rng.normal(0, 1, (seq, D)).astype(np.float32)
        ang = np.repeat(np.arange(seq, dtype=np.float32)[:, None]
                        * (THETA ** (-(np.arange(0, DH, 2, dtype=np.float32) / DH)))[None, :], 2, axis=1)
        cos = T.tensor(np.cos(ang).astype(np.float32))
        sin = T.tensor(np.sin(ang).astype(np.float32))
        e_o, w_o, m_o = bench_orv(seq, tr, cos, sin, x_np, gw_np)
        e_t, w_t, m_t = bench_torch(seq)
        log("seq=%3d | ORV  event %6.1f ms  wall %6.1f ms  mem≈%.2f GiB" % (seq, e_o, w_o, m_o / (1 << 30)))
        log("        | PyT  event %6.1f ms  wall %6.1f ms  mem %.2f GiB" % (e_t, w_t, m_t))
        log("        | 墙钟比 %.2fx（>1 表示 ORV 更快）" % (w_t / max(w_o, 1e-9)))
        torch.cuda.empty_cache()
    try:
        st.close(); T._cuda().empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
