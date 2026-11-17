# -*- coding: utf-8 -*-
"""CSA 全局分支：分块重算路径 vs 整体物化路径，**逐位对拍**。

判据（L1 级）：out / dx / 每个权重梯度的 max_abs 应为 0（同一批运算、同一顺序），
最小也应 ≤1e-6。任一不达标就不许打开开关。
"""
import argparse

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()


def build(seq, layers, d, d_ff, vocab, E, n_q):
    cfg = dict(d=d, d_ff=d_ff, E=E, layers=layers, n_q=n_q, n_kv=2,
               vocab=vocab, seq=seq, attn_res=False, attn_res_block=2,
               top_k=2, csa=4, n_kv_attn=n_q, rope_theta=500000.0,
               csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    P = CLI.build(cfg, np.random.default_rng(0))       # 只为了拿到填好的 csa_cfg
    return cfg, cfg["csa_cfg"], P


def run_once(csac, seq, d, cq):
    plan = C.make_plan(csac, 1, seq)
    cos, sin = C.rope_tables(csac, seq)
    rng = np.random.default_rng(7)
    w = C.init_weights(rng, csac)
    x = T.tensor(rng.normal(0, 0.02, (1, seq, d)).astype(np.float32),
                 requires_grad=True)
    C.set_attn_chunk(cq)
    out = C.csa_forward(x, w, plan, cos, sin)
    g = T.tensor(np.ones(tuple(int(s) for s in out.shape), np.float32))
    out.backward(g)
    rt.sync()
    og = np.asarray(rt.d2h(out._gpu), dtype=np.float32)
    dg = np.asarray(rt.d2h(x.grad._gpu), dtype=np.float32)
    wg = {}
    for k, v in w.items():
        if getattr(v, "grad", None) is not None:
            wg[k] = np.asarray(rt.d2h(v.grad._gpu), dtype=np.float32)
    C.set_attn_chunk(0)
    return og, dg, wg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--cq", type=int, default=128)
    ap.add_argument("--d", type=int, default=128)
    a = ap.parse_args()

    cfg, csac, P = build(a.seq, 1, a.d, 256, 4096, 2, 4)
    print("=== 对拍：seq=%d d=%d cq=%d ===" % (a.seq, a.d, a.cq))
    o0, d0, w0 = run_once(csac, a.seq, a.d, 0)          # 物化（参考）
    o1, d1, w1 = run_once(csac, a.seq, a.d, a.cq)       # 分块重算

    def rel(a_, b_):
        den = max(1e-30, float(np.abs(a_).max()))
        return float(np.abs(a_ - b_).max()) / den

    print("  out  max_abs=%.3e rel=%.3e" % (float(np.abs(o0 - o1).max()), rel(o0, o1)))
    print("  dx   max_abs=%.3e rel=%.3e" % (float(np.abs(d0 - d1).max()), rel(d0, d1)))
    bad = 0
    worst = 0.0
    for k in sorted(w0):
        if k not in w1:
            print("  权重 %s 缺少梯度" % k)
            bad += 1
            continue
        m = float(np.abs(w0[k] - w1[k]).max())
        worst = max(worst, m)
        if m > 1e-6:
            print("  dw[%s] max_abs=%.3e  ← 超界" % (k, m))
            bad += 1
    print("  权重梯度最差 max_abs=%.3e（%d 个权重）" % (worst, len(w0)))
    ok = (bad == 0 and float(np.abs(o0 - o1).max()) <= 1e-6
          and float(np.abs(d0 - d1).max()) <= 1e-6)
    print("\n  判定：%s" % ("逐位一致 / PASS" if ok else "**不一致**"))


if __name__ == "__main__":
    main()
