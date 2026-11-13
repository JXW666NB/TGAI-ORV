# -*- coding: utf-8 -*-
"""分项计量：注意力 vs MoE 各自为反向留存多少显存（建图模式，2 层，seq2048）。"""
import contextlib
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE  # noqa: E402
from v3_orv_train import build  # noqa: E402


def attn_part(x, L, cos, sin, d_k, n_q, n_kv):
    return x + V.attention_orv(V.rmsnorm_orv(x, L["ln1"]), L["wq"], L["wk"], L["wv"],
                               L["wo"], cos, sin, d_k, n_q, n_kv, flash=True)


def moe_part(x, L):
    Tlen = int(x.shape[1]); d = int(x.shape[2])
    hn = V.rmsnorm_orv(x, L["ln2"]).reshape(Tlen, d)
    eids, g, _s = MOE.route_top1(hn, L["router"])
    mo = MOE.moe_dispatch_orv(hn, eids, g, L["W1"], L["W2"], L["W3"])
    return x + mo.reshape(1, Tlen, d)


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    seq, E, Ln = 2048, 8, 2
    cfg = dict(d=512, d_ff=2048, E=E, layers=Ln, n_q=8, n_kv=2, vocab=4096,
               seq=seq, steps=1)
    rng = np.random.default_rng(0)
    P = build(cfg, rng)
    cos, sin = V.rope_cos_sin(seq, 64, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    ids = rng.integers(0, 4096, seq).astype(np.int64)
    base = T.tensor(P["emb"][ids].to_numpy().reshape(1, seq, 512))

    def measure(name, fn, use_grad):
        rt.empty_cache()
        before = rt.pool_stats()["peak_live_bytes"]
        with (contextlib.nullcontext() if use_grad else T.no_grad()):
            x = T.tensor(np.asarray(base.to_numpy(), np.float32), requires_grad=use_grad)
            y = x
            for L in P["layers"]:
                y = fn(y, L)
        pk = rt.pool_stats()["peak_live_bytes"]
        print("%-12s grad=%-5s peak_live=%.0f MiB" % (name, use_grad, pk / 2**20))
        del x, y
        rt.empty_cache()

    for nm, fn in (("attn", lambda x, L: attn_part(x, L, cos, sin, 64, 8, 2)),
                   ("moe", lambda x, L: moe_part(x, L)),
                   ("full", lambda x, L: moe_part(attn_part(x, L, cos, sin, 64, 8, 2), L))):
        for g in (False, True):
            measure(nm, fn, g)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
