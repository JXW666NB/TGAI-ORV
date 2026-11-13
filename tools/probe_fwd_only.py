# -*- coding: utf-8 -*-
"""只前向 vs 前向+反向 的峰值对比：判断内存是作业缓冲还是留存的激活。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE  # noqa: E402
sys.path.insert(0, r"D:/orv_pip_0962")
from v3_orv_train import build  # noqa: E402


def layer(x, L, cos, sin, d_k, n_q, n_kv):
    h = x + V.attention_orv(V.rmsnorm_orv(x, L["ln1"]), L["wq"], L["wk"], L["wv"],
                            L["wo"], cos, sin, d_k, n_q, n_kv, flash=True)
    hn3 = V.rmsnorm_orv(h, L["ln2"])
    Tlen = int(h.shape[1]); d = int(h.shape[2])
    hn = hn3.reshape(Tlen, d)
    eids, g, _s = MOE.route_top1(hn, L["router"])
    mo = MOE.moe_dispatch_orv(hn, eids, g, L["W1"], L["W2"], L["W3"])
    return h + mo.reshape(1, Tlen, d)


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    seq, E, d, n_q, n_kv = 2048, 8, 512, 8, 2
    cfg = dict(d=d, d_ff=4 * d, E=E, layers=1, n_q=n_q, n_kv=n_kv, vocab=4096,
               seq=seq, steps=1)
    rng = np.random.default_rng(0)
    P = build(cfg, rng)
    L = P["layers"][0]
    d_k = d // n_q
    cos, sin = V.rope_cos_sin(seq, d_k, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    ids = rng.integers(0, 4096, seq).astype(np.int64)

    rt.empty_cache()
    base = rt.pool_stats()["peak_live_bytes"]

    # ① 只前向
    with T.no_grad():
        x = P["emb"][ids].reshape(1, seq, d)
        y = layer(x, L, cos, sin, d_k, n_q, n_kv)
        _ = T.functional.cross_entropy(
            (V.rmsnorm_orv(y, P["ln_f"]).reshape(seq, d)
             @ P["head"].transpose(0, 1)), np.roll(ids, -1))
    fwd_only = rt.pool_stats()["peak_live_bytes"]
    print("只前向 peak_live = %.0f MiB (base %.0f)"
          % (fwd_only / 2**20, base / 2**20))

    # ② 前向 + 反向
    rt.empty_cache()
    x2 = T.tensor(np.asarray(x.to_numpy(), np.float32), requires_grad=True)
    y2 = layer(x2, L, cos, sin, d_k, n_q, n_kv)
    loss = T.functional.cross_entropy(
        (V.rmsnorm_orv(y2, P["ln_f"]).reshape(seq, d) @ P["head"].transpose(0, 1)),
        np.roll(ids, -1))
    loss.backward()
    both = rt.pool_stats()["peak_live_bytes"]
    print("前向+反向 peak_live = %.0f MiB ⇒ 反向额外 %+.0f MiB"
          % (both / 2**20, (both - fwd_only) / 2**20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
