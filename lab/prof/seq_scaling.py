# -*- coding: utf-8 -*-
"""跨 seq 的步时标定：ORV 的步时随 seq 是线性还是超线性？

要回答的问题：ORV 相对 PyTorch 的优势随 seq 变长而收窄（512 时 1.93×、1024 时 1.14×），
那到 8K/16K/64K 会不会**反超**？关键看两条曲线的增长阶：
    T(seq) = a + b·seq        （线性：注意力被分块/压缩成 O(seq)）
    T(seq) = a + b·seq²       （二次：注意力物化了 score 矩阵）

做法：对每个 seq 重新建模型（plan/rope 与 seq 有关），跑几步取中位，
同时记录 peak_live 与内核数，最后用相邻两点的比值估计阶数：
    p = log(T2/T1) / log(s2/s1)
若 p ≈ 1 是线性，p ≈ 2 是二次。

用 8 层（不是 24）以便 seq 能拉到 4096 而不 OOM；比较**同一模型的相对增长**
即可，不需要绝对时间与 24 层一致。
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


def one_seq(seq, layers, d_ff, steps):
    cfg = dict(d=1024, d_ff=d_ff, E=4, layers=layers, n_q=16, n_kv=2,
               vocab=96000, seq=seq, attn_res=False, attn_res_block=2,
               top_k=2, csa=4, n_kv_attn=16, rope_theta=500000.0,
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

    ts = []
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
    ts.sort()
    n = sum(int(p._gpu.n) for p in params)
    st = rt.pool_stats()
    return {"seq": seq, "med": ts[len(ts) // 2], "kerns": KERNS[0],
            "peak_gib": st["peak_live_bytes"] / float(1 << 30),
            "params_gib": n * 4 / float(1 << 30)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=3328)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seqs", default="512,1024,2048,4096")
    a = ap.parse_args()
    seqs = [int(x) for x in a.seqs.split(",")]

    rows = []
    for s in seqs:
        try:
            r = one_seq(s, a.layers, a.d_ff, a.steps)
        except Exception as e:
            print("  seq=%-5d 失败：%s" % (s, str(e)[:110]))
            break
        rows.append(r)
        print("  seq=%-5d 中位 %7.3f s  %8.1f tok/s  内核 %5d  peak %6.2f GiB"
              % (s, r["med"], s / r["med"], r["kerns"], r["peak_gib"]), flush=True)

    print("\n  增长阶估计（相邻两点 p = log(T2/T1)/log(s2/s1)）:")
    for i in range(1, len(rows)):
        p = np.log(rows[i]["med"] / rows[i - 1]["med"]) / np.log(
            rows[i]["seq"] / float(rows[i - 1]["seq"]))
        pk = np.log(rows[i]["peak_gib"] / rows[i - 1]["peak_gib"]) / np.log(
            rows[i]["seq"] / float(rows[i - 1]["seq"]))
        print("    %5d -> %-5d  时间 p=%.2f   显存 p=%.2f"
              % (rows[i - 1]["seq"], rows[i]["seq"], p, pk))
    if rows:
        print("\n  参数 %.2f GiB（%d 层 d_ff=%d）" % (rows[0]["params_gib"], a.layers, a.d_ff))


if __name__ == "__main__":
    main()
