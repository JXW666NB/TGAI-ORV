# -*- coding: utf-8 -*-
"""AMP 口径头对头：ORV(fp32) vs ORV(TG_FP16=1) vs PyTorch(autocast fp16)。

V100 是 Volta，**没有 bf16**，所以 AMP 只能 fp16。
注意：`TG_FP16` 在 import 时读环境变量，所以本脚本必须在 import tgai_go 之前设置它；
为了让三种口径各自干净，分别用子进程跑（脚本带 --mode）。

运行（需独占 GPU）：
    python amp_h2h.py orv_fp32
    python amp_h2h.py orv_fp16
    python amp_h2h.py pyt_amp
"""
import os
import sys
import time as _time

import numpy as np

MODE = sys.argv[1] if len(sys.argv) > 1 else "orv_fp32"
if MODE == "orv_fp16":
    os.environ["TG_FP16"] = "1"

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL_TG = r"D:\tgai\models\V2.7\TGAI_NB_V2.7_v4.tg"
D, NH, DF, NE, TOPK, THETA = 1024, 16, 4096, 8, 2, 1e4
DH = D // NH
LAYERS = (8,)
SEQ = 512
REP = 3


def run_orv(nl, seq):
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
    tr = V27Trainer(cfg, n_layers=nl)
    tr.load_all(st)
    rt = T._cuda()
    rng = np.random.default_rng(0)
    x_np = rng.normal(0, 1, (seq, D)).astype(np.float32)
    gw_np = rng.normal(0, 1, (seq, D)).astype(np.float32)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(seq, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos = T.tensor(np.cos(ang).astype(np.float32))
    sin = T.tensor(np.sin(ang).astype(np.float32))
    gw_t = T.tensor(gw_np)

    def step():
        cur = T.tensor(x_np, requires_grad=True)
        for i in range(nl):
            cur = tr.fwd(cur, i, cos, sin)
        loss = (cur * gw_t).sum()
        loss.backward()
        return float(np.asarray(loss.to_numpy()))

    def clear():
        for i in range(nl):
            for p in tr.params_of(i):
                p.grad = None

    l0 = step(); torch.cuda.synchronize()
    per = []
    for _ in range(REP):
        clear()
        torch.cuda.synchronize()
        t0 = _time.time()
        step()
        torch.cuda.synchronize()
        per.append((_time.time() - t0) * 1000.0)
    try:
        st.close(); rt.empty_cache()
    except Exception:
        pass
    return min(per), l0


def run_torch(nl, seq, amp):
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

    model = nn.ModuleList([Block() for _ in range(nl)]).to(dev)
    x0 = torch.randn(seq, D, dtype=dtype, device=dev)
    gw = torch.randn(seq, D, dtype=dtype, device=dev)

    def step():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(amp)):
            cur = x0
            for blk in model:
                cur = _ckpt(blk, cur, use_reentrant=False)
            loss = (cur * gw).sum()
            loss.backward()
        return float(loss)

    l0 = step(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    per = []
    for _ in range(REP):
        for p in model.parameters():
            p.grad = None
        torch.cuda.synchronize()
        t0 = _time.time()
        step()
        torch.cuda.synchronize()
        per.append((_time.time() - t0) * 1000.0)
    peak = torch.cuda.max_memory_allocated() / (1 << 30)
    return min(per), l0, peak


def main():
    for nl in LAYERS:
        if MODE.startswith("orv"):
            t, loss = run_orv(nl, SEQ)
            log("%-9s nl=%d seq=%d | wall %.1f ms | loss %.6f" % (MODE, nl, SEQ, t, loss))
        else:
            t, loss, peak = run_torch(nl, SEQ, MODE == "pyt_amp")
            log("%-9s nl=%d seq=%d | wall %.1f ms | loss %.4f | PyT 峰值显存 %.2f GiB"
                % (MODE, nl, SEQ, t, loss, peak))


if __name__ == "__main__":
    main()
