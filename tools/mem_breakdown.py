# -*- coding: utf-8 -*-
"""显存归因：把峰值拆成「权重 / 梯度 / 优化器状态 / 其余（激活+工作区）」。

## 为什么需要

社区反馈 "ORV 快 1.79×、显存多 1.89×"。但"多 1.89×"是个比值，不是原因。
必须先知道多出来的字节落在哪一类，否则改动只能是猜。

## 方法

两边用**同一份 numpy 权重**、同一个 dense 结构、同样的 batch/seq，
各自独立进程跑（避免互相污染）：

    python mem_breakdown.py --impl orv
    python mem_breakdown.py --impl torch

输出四类字节 + 峰值。`其余 = 峰值 - 权重 - 梯度 - 优化器`，它就是激活与工作区。

AdamW 的状态在两边都是 2 份（m / v），fp32 ⇒ `2 * 参数量 * 4`。
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

D, L, NH, FF, VOCAB, S, B = 1280, 12, 10, 5120, 32000, 1024, 4
DH = D // NH


def build_np():
    rng = np.random.default_rng(0)
    sd = {
        "emb": rng.normal(0, 0.02, (VOCAB, D)).astype(np.float32),
        "head": rng.normal(0, 0.02, (VOCAB, D)).astype(np.float32),
        "blocks": [dict(
            qkv=rng.normal(0, 0.02, (3 * D, D)).astype(np.float32),
            o=rng.normal(0, 0.02, (D, D)).astype(np.float32),
            g=rng.normal(0, 0.02, (FF, D)).astype(np.float32),
            u=rng.normal(0, 0.02, (FF, D)).astype(np.float32),
            dn=rng.normal(0, 0.02, (D, FF)).astype(np.float32),
        ) for _ in range(L)],
    }
    ids = rng.integers(0, VOCAB, (B, S)).astype(np.int64)
    tgt = rng.integers(0, VOCAB, (B, S)).astype(np.int64)
    return sd, ids, tgt


def n_param_bytes(sd):
    tot = sd["emb"].size + sd["head"].size
    for b in sd["blocks"]:
        tot += sum(v.size for v in b.values())
    return tot * 4


def run_orv(sd, ids, tgt, ckpt=False):
    sys.path.insert(0, os.path.join(REPO, "python"))
    import tgai_go.train as T
    from tgai_go.train import functional as Fn

    T.set_default_device("cuda")
    rt = T._cuda()

    def dev(a):
        return T.Tensor._from_gpu(rt.h2d(a))

    emb_w, head = dev(sd["emb"]), dev(sd["head"])
    ps = [emb_w, head]
    blk = []
    for b in sd["blocks"]:
        d = {k: dev(v) for k, v in b.items()}
        blk.append(d)
        ps.extend(d.values())
    for t in ps:
        t.requires_grad = True

    emb = T.nn.Embedding(VOCAB, D, weight=T.nn.Parameter(np.zeros((VOCAB, D), np.float32)))
    emb.weight._gpu = emb_w._gpu
    x = emb(ids.reshape(-1)).reshape(B, S, D)
    def block(x, d):
        z = x.reshape(-1, D) @ d["qkv"].T
        q, k, v = z.reshape(B, S, 3, NH, DH).permute(2, 0, 3, 1, 4)
        y = Fn.scaled_dot_product_attention(q.reshape(B * NH, S, DH),
                                            k.reshape(B * NH, S, DH),
                                            v.reshape(B * NH, S, DH), is_causal=True)
        x = x + (y.reshape(B, NH, S, DH).permute(0, 2, 1, 3)
                 .reshape(-1, D) @ d["o"].T).reshape(B, S, D)
        f = x.reshape(-1, D)
        h = Fn.silu(f @ d["g"].T) * (f @ d["u"].T)
        return x + (h @ d["dn"].T).reshape(B, S, D)

    for d in blk:
        if ckpt:
            x = T.checkpoint(lambda h, _d=d: block(h, _d), x)
        else:
            x = block(x, d)

    opt = T.optim.AdamW(ps, lr=1e-3)
    loss = Fn.cross_entropy((x.reshape(-1, D) @ head.T).reshape(-1, VOCAB),
                            tgt.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    rt.sync()
    st = rt.pool_stats()

    pbytes = n_param_bytes(sd)
    return dict(
        weights=pbytes,
        grads=pbytes,
        optimizer=2 * pbytes,
        peak=st["peak_live_bytes"],
        pool_cached=st.get("total_bytes", 0),
        note="ORV 的 peak 是池的活张量高水位（不含池缓存）",
    )


def run_torch(sd, ids, tgt, ckpt=False):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    dev_ = torch.device("cuda")

    def t(a):
        return torch.tensor(a, device=dev_, requires_grad=True)

    emb_w = t(sd["emb"])
    head = t(sd["head"])
    ps = [emb_w, head]
    blk = []
    for b in sd["blocks"]:
        d = {k: t(v) for k, v in b.items()}
        blk.append(d)
        ps.extend(d.values())

    i_ids = torch.tensor(ids, device=dev_)
    i_tgt = torch.tensor(tgt, device=dev_)
    x = F.embedding(i_ids, emb_w)
    def block(x, d):
        z = x.reshape(-1, D) @ d["qkv"].T
        q, k, v = z.reshape(B, S, 3, NH, DH).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + (y.permute(0, 2, 1, 3).reshape(-1, D) @ d["o"].T).reshape(B, S, D)
        f = x.reshape(-1, D)
        h = F.silu(f @ d["g"].T) * (f @ d["u"].T)
        return x + (h @ d["dn"].T).reshape(B, S, D)

    import torch.utils.checkpoint as _tck
    for d in blk:
        if ckpt:
            x = _tck.checkpoint(block, x, d, use_reentrant=False)
        else:
            x = block(x, d)

    opt = torch.optim.AdamW(ps, lr=1e-3)
    loss = F.cross_entropy((x.reshape(-1, D) @ head.T).reshape(-1, VOCAB),
                           i_tgt.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    torch.cuda.synchronize()

    pbytes = n_param_bytes(sd)
    return dict(
        weights=pbytes,
        grads=pbytes,
        optimizer=2 * pbytes,
        peak=torch.cuda.max_memory_allocated(),
        pool_cached=torch.cuda.max_memory_reserved(),
        note="torch 的 peak 是 allocator 的 max_memory_allocated",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["orv", "torch"], required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--ckpt", action="store_true", help="对每层开梯度检查点")
    a = ap.parse_args()
    sd, ids, tgt = build_np()
    r = (run_orv if a.impl == "orv" else run_torch)(sd, ids, tgt, a.ckpt)
    other = r["peak"] - r["weights"] - r["grads"] - r["optimizer"]

    def gib(n):
        return n / (1 << 30)

    rep = ["impl=%s  配置 D=%d L=%d FF=%d S=%d B=%d  检查点=%s"
           % (a.impl, D, L, FF, S, B, "开" if a.ckpt else "关"),
           "",
           "| 类别 | GiB | 占比 |",
           "| --- | --- | --- |"]
    for k in ("weights", "grads", "optimizer", "peak"):
        rep.append("| %s | %.3f | %s |" % (k, gib(r[k]),
                                           "%.1f%%" % (100.0 * r[k] / r["peak"])))
    rep.append("| **其余（激活+工作区）** | **%.3f** | **%.1f%%** |"
               % (gib(other), 100.0 * other / r["peak"]))
    rep.append("| 池缓存/保留 | %.3f | — |" % gib(r["pool_cached"]))
    rep.append("")
    rep.append(r["note"])
    print("\n".join(rep))
    if a.json:
        r["other"] = other
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(r, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
