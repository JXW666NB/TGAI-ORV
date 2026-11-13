# -*- coding: utf-8 -*-
"""对照：no_grad 前向 vs 建图前向，跑完后 live 各是多少（判断 no_grad 是否真的省内存）。"""
import contextlib
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE  # noqa: E402
from v3_orv_train import build  # noqa: E402


def layer(x, L, cos, sin, d_k, n_q, n_kv):
    h = x + V.attention_orv(V.rmsnorm_orv(x, L["ln1"]), L["wq"], L["wk"], L["wv"],
                            L["wo"], cos, sin, d_k, n_q, n_kv, flash=True)
    Tlen = int(h.shape[1]); d = int(h.shape[2])
    hn = V.rmsnorm_orv(h, L["ln2"]).reshape(Tlen, d)
    eids, g, _s = MOE.route_top1(hn, L["router"])
    mo = MOE.moe_dispatch_orv(hn, eids, g, L["W1"], L["W2"], L["W3"])
    return h + mo.reshape(1, Tlen, d)


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    seq, E, Ln = 2048, 8, (int(sys.argv[1]) if len(sys.argv) > 1 else 2)
    cfg = dict(d=512, d_ff=2048, E=E, layers=Ln, n_q=8, n_kv=2, vocab=4096,
               seq=seq, steps=1)
    rng = np.random.default_rng(0)
    P = build(cfg, rng)
    cos, sin = V.rope_cos_sin(seq, 64, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    ids = rng.integers(0, 4096, seq).astype(np.int64)

    def run(use_grad):
        rt.empty_cache()
        before = rt.pool_stats()["live_bytes"]
        with (contextlib.nullcontext() if use_grad else T.no_grad()):
            x = P["emb"][ids].reshape(1, seq, 512)
            if use_grad:
                x = T.tensor(np.asarray(x.to_numpy(), np.float32), requires_grad=True)
            y = x
            for L in P["layers"]:
                y = layer(y, L, cos, sin, 64, 8, 2)
                rt.empty_cache()          # ★ 层间回收：ORV 的 live 含"已丢弃未归还"的块
        after = rt.pool_stats()["live_bytes"]
        peak = rt.pool_stats()["peak_live_bytes"]
        print("use_grad=%-5s  live %6.0f → %6.0f MiB（增量 %+.0f），peak_live=%.0f"
              % (use_grad, before / 2**20, after / 2**20,
                 (after - before) / 2**20, peak / 2**20))
        del y, x
        rt.empty_cache()
        return after

    run(False)
    run(True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
