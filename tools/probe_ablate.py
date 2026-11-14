# -*- coding: utf-8 -*-
"""消融：分别去掉 MoE / 注意力，看整步（前向+反向）变化。"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE  # noqa: E402
from v3_orv_train import build  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    d, S, E, L, n_q, n_kv = 512, (int(sys.argv[1]) if len(sys.argv) > 1 else 2048), 8, 8, 8, 2
    d_k = d // n_q
    cfg = dict(d=d, d_ff=4 * d, E=E, layers=L, n_q=n_q, n_kv=n_kv, vocab=4096,
               seq=S, steps=1)
    rng = np.random.default_rng(0)
    P = build(cfg, rng)
    cos, sin = V.rope_cos_sin(S, d_k, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    ids = rng.integers(0, 4096, S).astype(np.int64)
    tgt = np.roll(ids, -1)

    def step(use_attn=True, use_moe=True):
        x = P["emb"][ids].reshape(1, S, d)
        for Lw in P["layers"]:
            if use_attn:
                a = V.attention_orv(V.rmsnorm_orv(x, Lw["ln1"]), Lw["wq"], Lw["wk"],
                                    Lw["wv"], Lw["wo"], cos, sin, d_k, n_q, n_kv)
            else:
                a = V.rmsnorm_orv(x, Lw["ln1"]) * 0.0
            h = x + a
            if use_moe:
                hn = V.rmsnorm_orv(h, Lw["ln2"]).reshape(S, d)
                e, g, _ = MOE.route_topk(hn, Lw["router"], 2)
                mo = MOE.moe_topk_orv(hn, e, g, Lw["W1"], Lw["W2"], Lw["W3"])
                mo = mo + V.situ_glu_orv(hn, Lw["S1"], Lw["S2"], Lw["S3"])
                x = h + mo.reshape(1, S, d)
            else:
                x = h
        lg = V.rmsnorm_orv(x, P["ln_f"]).reshape(S, d) @ P["head"].transpose(0, 1)
        return T.functional.cross_entropy(lg, tgt)

    def bench(label, **kw):
        l = step(**kw); l.backward(); rt.sync()      # 预热
        rt.empty_cache()
        t0 = time.time()
        l = step(**kw)
        fwd = time.time() - t0
        t1 = time.time()
        l.backward()
        rt.sync()
        bwd = time.time() - t1
        print("%-16s 前向 %.3fs  反向 %.3fs  合计 %.3fs" % (label, fwd, bwd, fwd + bwd))

    bench("全量")
    bench("去掉 MoE", use_moe=False)
    bench("去掉注意力", use_attn=False)
    bench("两者都去", use_attn=False, use_moe=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
