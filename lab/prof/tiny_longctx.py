# -*- coding: utf-8 -*-
"""超小模型 + 超长上下文的标定：看 ORV 的渐近行为。

目的：在不受显存墙干扰的前提下把 seq 拉到 64K，回答：
  1. 步时随 seq 是线性（p≈1）还是二次（p≈2）？
  2. **显存**随 seq 是线性还是二次？——物化 scores 时会二次增长。

用 2 层 / d=128 的小模型，因为要看的是**渐近阶**，不是绝对性能。
打开 CSA 分块：`TG_CSA_ATTN_CQ=512`（见 docs/LONG_CONTEXT_ROUTE_V3.md）。
"""
import argparse
import time

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()

KERNS = [0]
_ol = rt.launch


def launch(name, grid, block, args=None, *a, **kw):
    KERNS[0] += 1
    return _ol(name, grid, block, args, *a, **kw)


rt.launch = launch


def one_seq(seq, a, steps):
    cfg = dict(d=a.d, d_ff=a.d_ff, E=a.E, layers=a.layers, n_q=a.n_q, n_kv=2,
               vocab=a.vocab, seq=seq, attn_res=False, attn_res_block=2,
               top_k=2, csa=4, n_kv_attn=a.n_q, rope_theta=500000.0,
               csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    rng = np.random.default_rng(0)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    params = CLI.params_of(P)
    opt = T.optim.AdamW(params, lr=3e-4)
    csac = cfg["csa_cfg"]
    plan = C.make_plan(csac, 1, seq)
    cos, sin = C.rope_tables(csac, seq)
    ids = np.random.default_rng(1).integers(0, cfg["vocab"], seq).astype(np.int64)
    tgt = np.roll(ids, -1)
    base = sum(int(p._gpu.n) for p in params) * 4 / float(1 << 30)

    ts = []
    losses = []
    for i in range(steps):
        for p in params:
            p.grad = None
        t0 = time.perf_counter()
        KERNS[0] = 0
        logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
        loss = T.functional.cross_entropy(logits.reshape(seq, cfg["vocab"]), tgt)
        loss.backward(T.tensor(np.array(1.0, np.float32)))
        opt.step()
        rt.sync()
        dt = time.perf_counter() - t0
        if i >= 1:
            ts.append(dt)
            losses.append(float(np.asarray(loss.to_numpy()).reshape(-1)[0]))
    ts.sort()
    st = rt.pool_stats()
    pk = st["peak_live_bytes"] / float(1 << 30)
    return {"seq": seq, "med": ts[len(ts) // 2], "kerns": KERNS[0],
            "peak": pk, "params": base, "act": max(0.0, pk - base),
            "losses": losses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--E", type=int, default=2)
    ap.add_argument("--n_q", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seqs", default="512,2048,8192,16384,32768,65536")
    a = ap.parse_args()
    seqs = [int(x) for x in a.seqs.split(",")]
    print("=== ORV 超小模型 d=%d L=%d d_ff=%d vocab=%d n_q=%d  (TG_CSA_ATTN_CQ=%s) ==="
          % (a.d, a.layers, a.d_ff, a.vocab, a.n_q,
             C._ATTN_CQ[0]))

    rows = []
    for s in seqs:
        try:
            r = one_seq(s, a, a.steps)
        except Exception as e:
            print("  seq=%-6d FAILED: %s" % (s, str(e).replace("\n", " ")[:110]))
            break
        rows.append(r)
        print("  seq=%-6d med %8.3f s %9.1f tok/s kerns %5d peak %7.2f GiB act %6.2f  loss %s"
              % (s, r["med"], s / r["med"], r["kerns"], r["peak"], r["act"],
                 ", ".join("%.6f" % x for x in r["losses"])), flush=True)

    print("\n  阶数估计 p = log(T2/T1)/log(s2/s1):")
    for i in range(1, len(rows)):
        f = rows[i]["seq"] / float(rows[i - 1]["seq"])
        pt = np.log(rows[i]["med"] / rows[i - 1]["med"]) / np.log(f)
        pa = ""
        if rows[i - 1]["act"] > 0.05 and rows[i]["act"] > 0.05:
            pa = "  显存p=%.2f" % (np.log(rows[i]["act"] / rows[i - 1]["act"]) / np.log(f))
        print("    %6d -> %-6d 时间 p=%.2f%s"
              % (rows[i - 1]["seq"], rows[i]["seq"], pt, pa))


if __name__ == "__main__":
    main()
