# -*- coding: utf-8 -*-
"""HOPE V3 权威参照侧的最小训练 harness（只比 CE loss，恒定 lr，无调度）。

严格照 `_src15b/train_v3.py::micro_step` 的调用方式：
    logits, loss, _ = model(ids, targets=labels, seg_id=seg, return_qk_max=...)
    aux = model.aux_loss()

**取窗顺序完全照 ORV/`train_v3.py::PackedDataset.order` 的规则**（这是关键）：
    shard_order = rng(seed + epoch*100003).permutation(n_shards)
    si          = shard_order[rank]
    mix         = (seed*1000003 + epoch*100003 + si*7919) % 2**62
    window_order= rng(mix).permutation(n_windows_of_si)
两侧因此看到**同样的窗口、同样的顺序**，loss 曲线才可比。
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_v3 as M  # noqa: E402


def hope_order(paths, seed, epoch=0, rank=0):
    """按 train_v3.py / ORV 的规则产出 (shard_idx, window_idx) 序列。"""
    n_shards = len(paths)
    n_wins = []
    for p in paths:
        with np.load(p) as z:
            n_wins.append(int(z["input_ids"].shape[0]))
    shard_perm = np.random.default_rng(int(seed) + int(epoch) * 100003).permutation(n_shards)
    multi = False
    if len(shard_perm) > 1:
        multi = True
    out = []
    for si_raw in shard_perm:
        si = int(si_raw)
        mix = (int(seed) * 1000003 + int(epoch) * 100003 + si * 7919) % (2 ** 62)
        for wi in np.random.default_rng(int(mix)).permutation(n_wins[si]):
            out.append((si, int(wi)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", required=True)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--amp", type=int, default=1)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    cfg = M.TGAIV3Config(
        vocab_size=96000, d_model=1024, n_layers=a.layers, n_heads=16,
        d_ff=2816, max_seq_len=4096, tie_embeddings=True,
        attn_global_every=4, n_kv_heads=0, csa_compress=4,
        csa_conv_kernel=3, csa_local_window=0, attn_qk_norm=True,
        n_experts=8, n_activated=2, n_shared_experts=1,
        moe_balance="quantile", ce_chunk=0,
    )
    model = M.TGAIV3Model(cfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(a.amp))

    paths = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    order = hope_order(paths, a.seed)
    cache = {}

    def win(si, wi):
        z = cache.get(si)
        if z is None:
            z = np.load(paths[si])
            cache.clear()
            cache[si] = z
        ids = z["input_ids"][wi, :a.seq]
        lab = z["labels"][wi, :a.seq]
        seg = z["seg_id"][wi, :a.seq] if "seg_id" in z.files else np.zeros_like(ids)
        return ids, lab, seg

    ces, auxs, tms = [], [], []
    for step in range(a.steps):
        si, wi = order[step % len(order)]
        ids, lab, seg = win(si, wi)
        ti = torch.from_numpy(np.ascontiguousarray(ids)).long().to(dev)[None, :]
        tl = torch.from_numpy(np.ascontiguousarray(lab)).long().to(dev)[None, :]
        ts = torch.from_numpy(np.ascontiguousarray(seg)).long().to(dev)[None, :]
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=bool(a.amp)):
            _logits, loss, _ = model(ti, targets=tl, seg_id=ts,
                                     return_qk_max=(cfg.qk_clip_tau > 0))
            aux = model.aux_loss()
        scaler.scale(loss.float() + aux.float()).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        tms.append((time.perf_counter() - t0) * 1e3)
        ces.append(float(loss.item()))
        auxs.append(float(aux.item()))

    print("[hope_q] " + json.dumps({
        "seed_used": a.seed, "params": int(n_par), "layers": a.layers,
        "seq": a.seq, "steps": a.steps, "lr": a.lr, "clip": a.clip,
        "amp": bool(a.amp), "tie": True, "order": "hope",
        "ce": [round(x, 6) for x in ces],
        "aux": [round(x, 6) for x in auxs],
        "ce_mean_last10": round(float(np.mean(ces[-10:])), 6),
        "ce_min": round(float(min(ces)), 6),
        "step_ms": round(float(np.median(tms)), 1),
        "first_win": [int(order[0][0]), int(order[0][1])],
        "qk_max_max": float(model._qk_max.max().item()),
        "qk_max_per_head_max": [round(float(x), 3) for x in model._qk_max.max(dim=0).values],
    }))


if __name__ == "__main__":
    main()
