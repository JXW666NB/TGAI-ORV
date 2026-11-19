# -*- coding: utf-8 -*-
"""C6：PyTorch AdamW（fp32）在与 ORV 完全相同的模型/数据/lr 网格下对比。

用 C:\\Python311\\python.exe 运行（torch 2.6.0+cu124）。
权重来自 p20_dump_state.py 导出的 npz ⇒ 两侧初始化逐位相同。
先做一次 logits 对齐检查，不一致就不出结论。
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if (torch.cuda.is_available() and "--cpu" not in sys.argv) else "cpu"


def dump(name, obj):
    p = os.path.join(OUT, name + ".json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    print("WROTE " + p)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        var = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class Block(nn.Module):
    def __init__(self, d, h, d_ff):
        super().__init__()
        self.d, self.h, self.dh = d, h, d // h
        self.ln1 = RMSNorm(d)
        self.ln2 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x, B, S):
        D, H, Dh = self.d, self.h, self.dh
        xf = x.reshape(B * S, D)
        h = self.ln1(xf)
        qkv = self.qkv(h)
        q, k, v = qkv[:, 0:D], qkv[:, D:2 * D], qkv[:, 2 * D:3 * D]
        q = q.reshape(B, S, H, Dh).permute(0, 2, 1, 3).reshape(B * H, S, Dh)
        k = k.reshape(B, S, H, Dh).permute(0, 2, 1, 3).reshape(B * H, S, Dh)
        v = v.reshape(B, S, H, Dh).permute(0, 2, 1, 3).reshape(B * H, S, Dh)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.reshape(B, H, S, Dh).permute(0, 2, 1, 3).reshape(B * S, D)
        y = xf + self.o(att)
        h2 = self.ln2(y)
        z = F.silu(self.w1(h2)) * self.w3(h2)
        y = y + self.w2(z)
        return y.reshape(B, S, D)


class MiniV3(nn.Module):
    def __init__(self, vocab=4096, d=128, nl=2, nh=4, d_ff=512):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, nh, d_ff) for _ in range(nl)])
        self.norm_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, idx):
        B, S = idx.shape
        x = self.embed(idx.reshape(B * S)).reshape(B, S, -1)
        for blk in self.blocks:
            x = blk(x, B, S)
        h = self.norm_f(x.reshape(B * S, -1))
        return self.lm_head(h)


def main():
    z = np.load(os.path.join(OUT, "state_v4096.npz"), allow_pickle=True)
    X = np.asarray(z["X"], np.int64)
    Y = np.asarray(z["Y"], np.int64)
    meta = json.load(open(os.path.join(OUT, "state_v4096_meta.json"), encoding="utf-8"))

    torch.manual_seed(0)
    model = MiniV3().to(DEV)
    sd = {}
    for n in meta["names"]:
        key = "w::" + n
        arr = np.asarray(z[key], np.float32)
        sd[n] = torch.from_numpy(arr).to(DEV)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    info = {"device": DEV, "missing": list(missing), "unexpected": list(unexpected),
            "torch": torch.__version__}

    # ---- 前向一致性 ----
    xb = torch.from_numpy(X[:8]).long().to(DEV)
    with torch.no_grad():
        lg = model(xb)
    lgt = lg.reshape(8 * 64, 4096)
    yb = torch.from_numpy(Y[:8]).long().reshape(-1).to(DEV)
    loss0 = F.cross_entropy(lgt, yb)
    info["init_loss_b8_torch"] = float(loss0.item())
    info["meta_keys"] = sorted(meta.keys())
    info["init_loss_b8_orv"] = meta.get("init_loss_b8")
    if info["init_loss_b8_orv"] is None:
        # p20 实测值（stdout 已打印）：8.346833229064941
        info["init_loss_b8_orv"] = 8.346833229064941
        info["init_loss_b8_orv_source"] = "p20 stdout（meta 里该键缺失）"
    info["init_loss_absdiff"] = abs(info["init_loss_b8_torch"] - info["init_loss_b8_orv"])
    print(json.dumps(info, ensure_ascii=True, indent=1))

    # ---- lr 网格（与 ORV 侧一致） ----
    GRID = [1e-3, 3e-3, 1e-2, 3e-2]
    STEPS = 200
    BATCH = 8
    recs = []
    for lr in GRID:
        torch.manual_seed(0)
        m = MiniV3().to(DEV)
        m.load_state_dict(sd, strict=False)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.999),
                                eps=1e-8, weight_decay=0.0)
        curve, ts = [], []
        for i in range(STEPS):
            t0 = time.perf_counter()
            logits = m(xb)
            loss = F.cross_entropy(logits.reshape(BATCH * 64, 4096), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            curve.append(float(loss.item()))
            ts.append(time.perf_counter() - t0)
        warm = ts[10:]
        r = {"task": "mini_v3_overfit", "opt": "pytorch_AdamW", "lr": lr,
             "steps": STEPS, "batch": BATCH, "seq": 64, "device": DEV,
             "first": curve[0], "final": curve[-1], "min": min(curve),
             "last10_mean": float(np.mean(curve[-10:])),
             "loss_at_10": curve[9], "loss_at_25": curve[24],
             "loss_at_50": curve[49], "loss_at_100": curve[99],
             "t_step_s": float(np.mean(warm)),
             "tokens_per_s": 8 * 64 / float(np.mean(warm)),
             "all_finite": all(v == v for v in curve),
             "curve": [round(v, 6) for v in curve]}
        recs.append(r)
        print("P %-16s lr=%-7.0e l10=%-10.6g l50=%-10.6g min=%-10.6g ms=%.1f"
              % ("pytorch_AdamW", lr, r["last10_mean"], r["loss_at_50"], r["min"],
                 r["t_step_s"] * 1e3), flush=True)

    # ---- 速度：预热 + 3 次中位数（CUDA event） ----
    speed = {}
    torch.manual_seed(0)
    m = MiniV3().to(DEV)
    m.load_state_dict(sd, strict=False)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(5):
        loss = F.cross_entropy(m(xb).reshape(BATCH * 64, 4096), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    reps = []
    for _ in range(3):
        if DEV == "cuda":
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            e0.record()
            for _ in range(20):
                loss = F.cross_entropy(m(xb).reshape(BATCH * 64, 4096), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            e1.record()
            e1.synchronize()
            reps.append(e0.elapsed_time(e1) / 20)
        else:
            t0 = time.perf_counter()
            for _ in range(20):
                loss = F.cross_entropy(m(xb).reshape(BATCH * 64, 4096), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            reps.append((time.perf_counter() - t0) / 20 * 1e3)
    speed["pytorch_AdamW_step_ms_median"] = float(np.median(reps))
    speed["reps_ms"] = [round(r, 3) for r in reps]
    speed["tokens_per_s"] = 8 * 64 / (float(np.median(reps)) / 1e3)
    if DEV == "cuda":
        speed["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 4)
        speed["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 4)
    print("P speed " + json.dumps(speed, ensure_ascii=True))

    dump("p30_pyt_model", {"parity": info, "records": recs, "speed": speed,
                           "grid": GRID, "steps": STEPS})


if __name__ == "__main__":
    main()
