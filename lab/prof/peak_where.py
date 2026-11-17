# -*- coding: utf-8 -*-
"""抓住"显存峰值那一刻"的驻留快照。

背景：32K 上 forward 结束只有 6.28 GiB，但 peak_live 报 7.64 GiB —— 峰值出现在
**反向中途**。想优化必须知道那一刻到底谁在占，所以这里给 `_alloc` 打补丁：
每当借出字节数刷新历史高点，就记一份 `pool_dump_live(top=6)`（只留最后一份）。
"""
import argparse
import traceback

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_train_cli as CLI
from tgai_go.train import v3_csa as C

T.set_default_device("cuda")
rt = T._cuda()

PEAK = {"bytes": 0, "top": None, "where": None}

_orig_alloc = rt._alloc


def alloc(size, *a, **kw):
    buf = _orig_alloc(size, *a, **kw)
    try:
        cur = rt.pool_stats()["live_bytes"]
    except Exception:
        return buf
    if cur > PEAK["bytes"]:
        PEAK["bytes"] = cur
        PEAK["top"] = rt.pool_dump_live(top=8)["top"]
        st = traceback.extract_stack()
        PEAK["where"] = [(f.name, f.lineno) for f in st[-6:-1]]
    return buf


rt._alloc = alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=16384)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d", type=int, default=128)
    a = ap.parse_args()
    seq = a.seq

    cfg = dict(d=a.d, d_ff=256, E=2, layers=a.layers, n_q=4, n_kv=2,
               vocab=8192, seq=seq, attn_res=False, attn_res_block=2,
               top_k=2, csa=4, n_kv_attn=4, rope_theta=500000.0,
               csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    rng = np.random.default_rng(0)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    params = CLI.params_of(P)
    csac = cfg["csa_cfg"]
    plan = C.make_plan(csac, 1, seq)
    cos, sin = C.rope_tables(csac, seq)
    ids = np.random.default_rng(1).integers(0, cfg["vocab"], seq).astype(np.int64)
    tgt = np.roll(ids, -1)

    print("=== seq=%d L=%d d=%d cq=%d ===" % (seq, a.layers, a.d, C._ATTN_CQ[0]))
    logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
    rt.sync()
    print("  forward 结束 live=%.2f GiB" % (rt.pool_stats()["live_bytes"] / (1 << 30)))

    loss = T.functional.cross_entropy(logits.reshape(seq, cfg["vocab"]), tgt)
    loss.backward(T.tensor(np.array(1.0, np.float32)))
    rt.sync()

    print("  峰值 live=%.2f GiB" % (PEAK["bytes"] / (1 << 30)))
    print("  峰值时分配点（Python 栈后 5 层）：")
    for nm, ln in (PEAK["where"] or []):
        print("      %s:%s" % (nm, ln))
    print("  峰值时的驻留归属：")
    for sz, cnt, tot in (PEAK["top"] or []):
        print("      n=%-10d x%-4d %9.2f MB" % (sz // 4, cnt, tot / (1 << 20)))


if __name__ == "__main__":
    main()
