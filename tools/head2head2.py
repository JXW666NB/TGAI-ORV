# -*- coding: utf-8 -*-
"""修正版头对头：同一次 rep 同时取 event 与墙钟；校验梯度确实写出；跑到 8 层找交叉点。

运行（需独占 GPU）：python head2head2.py
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
D, NH, DF, NE, TOPK, THETA = 1024, 16, 4096, 8, 2, 1e4
DH = D // NH
SEQ = 512
LAYERS = (20,)
REP = 3


def _np(x):
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, np.float32)


def bench_orv(nl, tr, cos, sin, x_np, gw_np):
    import torch
    import tgai_go.train as T
    rt = T._cuda()
    gw_t = T.tensor(gw_np)

    def step():
        cur = T.tensor(x_np, requires_grad=True)
        for i in range(nl):
            cur = tr.fwd(cur, i, cos, sin)
        (cur * gw_t).sum().backward()

    def clear():
        for i in range(nl):
            for p in tr.params_of(i):
                p.grad = None

    step(); torch.cuda.synchronize()
    # 校验：梯度真的写出来了（否则"快"是假象）
    gsum = 0.0
    for i in range(nl):
        for p in tr.params_of(i):
            if p.grad is not None:
                gsum += float(np.abs(_np(p.grad)).sum())
    per = []
    for _ in range(REP):
        clear()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0 = _time.time()
        e0.record(); step(); e1.record(); e1.synchronize()
        per.append((e0.elapsed_time(e1), (_time.time() - t0) * 1000.0))
    return per, gsum


def bench_torch(nl, seq, device="cuda"):
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

    model = nn.ModuleList([Block() for _ in range(nl)]).to(device)
    x0 = torch.randn(seq, D, dtype=dtype, device=device)
    gw = torch.randn(seq, D, dtype=dtype, device=device)

    def step():
        cur = x0
        for blk in model:
            cur = _ckpt(blk, cur, use_reentrant=False)
        (cur * gw).sum().backward()

    step(); torch.cuda.synchronize()
    gsum = float(sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None))
    per = []
    for _ in range(REP):
        for p in model.parameters():
            p.grad = None
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0 = _time.time()
        e0.record(); step(); e1.record(); e1.synchronize()
        per.append((e0.elapsed_time(e1), (_time.time() - t0) * 1000.0))
    peak = torch.cuda.max_memory_allocated() / (1 << 30)
    del model
    torch.cuda.empty_cache()
    return per, gsum, peak


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
    rng = np.random.default_rng(0)
    x_np = rng.normal(0, 1, (SEQ, D)).astype(np.float32)
    gw_np = rng.normal(0, 1, (SEQ, D)).astype(np.float32)
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None]
                    * (THETA ** (-(np.arange(0, DH, 2, dtype=np.float32) / DH)))[None, :], 2, axis=1)
    cos = T.tensor(np.cos(ang).astype(np.float32))
    sin = T.tensor(np.sin(ang).astype(np.float32))

    for nl in LAYERS:
        st = TGState(MODEL_TG)
        tr = V27Trainer(cfg, n_layers=nl)
        tr.load_all(st)
        per_o, g_o = bench_orv(nl, tr, cos, sin, x_np, gw_np)
        try:
            per_t, g_t, mem_t = bench_torch(nl, SEQ)
        except torch.OutOfMemoryError as e:
            log("nl=%d | PyTorch OOM（%s）" % (nl, str(e).split(".")[0][:60]))
            tr = None
            try:
                st.close(); T._cuda().empty_cache()
            except Exception:
                pass
            continue
        log("nl=%d seq=%d | ORV 每rep(event/wall) %s | PyT %s"
            % (nl, SEQ, [(round(a, 1), round(b, 1)) for a, b in per_o],
               [(round(a, 1), round(b, 1)) for a, b in per_t]))
        log("        | 梯度合计 ORV %.4g / PyT %.4g（应同量级，否则说明有一侧没算梯度）" % (g_o, g_t))
        log("        | 最好一次：ORV wall %.1f ms vs PyT wall %.1f ms ⇒ %.2fx；PyT 峰值显存 %.2f GiB"
            % (min(b for _a, b in per_o), min(b for _a, b in per_t),
               min(b for _a, b in per_t) / max(min(b for _a, b in per_o), 1e-9), mem_t))
        try:
            st.close(); T._cuda().empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
