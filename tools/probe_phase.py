# -*- coding: utf-8 -*-
"""在真实训练步里逐阶段记录 live/peak，定位峰值发生在哪一步（8 层, seq2048, E=8, flash）。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE  # noqa: E402
from v3_orv_train import build, params_of  # noqa: E402


def mib(b):
    return b / 2 ** 20


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    d, seq, E, Ln, n_q, n_kv = 512, 2048, 8, 8, 8, 2
    cfg = dict(d=d, d_ff=4 * d, E=E, layers=Ln, n_q=n_q, n_kv=n_kv, vocab=4096,
               seq=seq, steps=1)
    rng = np.random.default_rng(0)
    P = build(cfg, rng)
    d_k = d // n_q
    cos, sin = V.rope_cos_sin(seq, d_k, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    opt = T.optim.AdamW(params_of(P), lr=3e-4)
    ids = rng.integers(0, 4096, seq).astype(np.int64)
    tgt = np.roll(ids, -1)

    def show(tag):
        st = rt.pool_stats()
        print("%-22s live=%6.0f peak=%6.0f MiB" % (tag, mib(st["live_bytes"]),
                                                   mib(st["peak_live_bytes"])))

    show("起始")
    tot = sum(int(np.prod(t.shape)) for t in params_of(P))
    print("参数张量合计 = %.1f M ⇒ fp32 应占 %.0f MiB" % (tot / 1e6, tot * 4 / 2**20))
    x = P["emb"][ids].reshape(1, seq, d)
    show("嵌入后")
    for i, L in enumerate(P["layers"]):
        h = x + V.attention_orv(V.rmsnorm_orv(x, L["ln1"]), L["wq"], L["wk"], L["wv"],
                                L["wo"], cos, sin, d_k, n_q, n_kv, flash=True)
        hn = V.rmsnorm_orv(h, L["ln2"]).reshape(seq, d)
        eids, g, _s = MOE.route_top1(hn, L["router"])
        mo = MOE.moe_dispatch_orv(hn, eids, g, L["W1"], L["W2"], L["W3"])
        x = h + mo.reshape(1, seq, d)
        show("layer %d 前向" % i)
    logits = V.rmsnorm_orv(x, P["ln_f"]).reshape(seq, d) @ P["head"].transpose(0, 1)
    show("头部前向")
    loss = T.functional.cross_entropy(logits, tgt)
    show("CE 后")
    loss.backward()
    show("反向后")
    opt.step()
    show("opt.step 后")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
