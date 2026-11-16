# -*- coding: utf-8 -*-
"""ORV 的 op 级 GPU 耗时归因（CUDA event 版，不改库）。

做法：包住 `rt.launch`，每次发射前后在同一流上各记一个 event，
同步后逐个读 elapsed —— 得到**每个内核的真实 GPU 时间**，按名字聚合。

这是定位"1 层 backward 里那 185 ms"的唯一可靠手段（此前的单次墙钟测量噪声达 60%）。

在 V100 上跑：
    D:\\orv_pip_0962\\Scripts\\python.exe op_time.py --layers 1 --seq 256
"""
import argparse
import collections
import ctypes

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()
cu = rt.cu
STREAM = rt._stream

PAIRS = []
CUR = [None]
FREED = []


def _mk_event():
    if FREED:
        return FREED.pop()
    e = ctypes.c_void_p()
    cu.cuEventCreate(ctypes.byref(e), 0)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--n_q", type=int, default=16)
    ap.add_argument("--d_ff", type=int, default=2816)
    ap.add_argument("--vocab", type=int, default=96000)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--csa", type=int, default=4)
    a = ap.parse_args()

    orig = rt.launch

    def launch(name, grid, block, args=None, *ar, **kw):
        if not CUR[0]:
            return orig(name, grid, block, args, *ar, **kw)
        e0, e1 = _mk_event(), _mk_event()
        cu.cuEventRecord(e0, STREAM)
        r = orig(name, grid, block, args, *ar, **kw)
        cu.cuEventRecord(e1, STREAM)
        PAIRS.append((CUR[0], name, e0, e1))
        return r

    rt.launch = launch

    cfg = dict(d=a.d, d_ff=a.d_ff, E=a.E, layers=a.layers, n_q=a.n_q, n_kv=2,
               vocab=a.vocab, seq=a.seq, attn_res=False, attn_res_block=2,
               top_k=2, csa=a.csa, n_kv_attn=a.n_q, rope_theta=500000.0,
               csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    rng = np.random.default_rng(0)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    params = CLI.params_of(P)
    opt = T.optim.AdamW(params, lr=3e-4)
    seq = int(cfg["seq"])
    csac = cfg["csa_cfg"]
    plan = C.make_plan(csac, 1, seq)
    cos, sin = C.rope_tables(csac, seq)
    ids = np.random.default_rng(1).integers(0, cfg["vocab"], seq).astype(np.int64)
    tgt = np.roll(ids, -1)

    def phase_body():
        logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
        loss = T.functional.cross_entropy(logits.reshape(seq, cfg["vocab"]), tgt)
        return loss

    # 预热（不记录）
    loss = phase_body()
    for p in params:
        p.grad = None
    loss.backward()
    opt.step()
    rt.sync()

    agg = {"forward": collections.Counter(), "backward": collections.Counter(),
           "opt": collections.Counter()}
    cnt = {k: collections.Counter() for k in agg}

    def run_phase(tag, fn):
        PAIRS.clear()
        CUR[0] = tag
        out = fn()          # ← 必须把返回值带出来（backward 要用 forward 的 loss）
        CUR[0] = None
        rt.sync()
        for (tg, name, e0, e1) in PAIRS:
            ms = ctypes.c_float(0.0)
            cu.cuEventElapsedTime(ctypes.byref(ms), e0, e1)
            agg[tg][name] += float(ms.value)
            cnt[tg][name] += 1
            FREED.append(e0)
            FREED.append(e1)
        PAIRS.clear()
        return out

    loss = run_phase("forward", phase_body)
    CORP = [loss]

    def bwd():
        for p in params:
            p.grad = None
        CORP[0].backward()

    run_phase("backward", bwd)
    run_phase("opt", lambda: opt.step())

    print("=== op 级 GPU 耗时（layers=%d seq=%d vocab=%d）===" % (a.layers, a.seq, a.vocab))
    for tag in ("forward", "backward", "opt"):
        tot = sum(agg[tag].values())
        print("\n[%s] 合计 %.1f ms" % (tag, tot))
        for name, ms in agg[tag].most_common(10):
            print("  %-32s %8.2f ms  %5d 次  %6.1f%%"
                  % (name, ms, cnt[tag][name], ms * 100.0 / max(tot, 1e-9)))


if __name__ == "__main__":
    main()
