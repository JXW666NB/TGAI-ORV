# -*- coding: utf-8 -*-
"""给模型里 MoE 的每一段计时（前向+反向合计），定位 43 ms/层的去处。"""
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE  # noqa: E402
from v3_orv_train import build, params_of  # noqa: E402

ACC = defaultdict(float)
CNT = defaultdict(int)


def wrap(mod, name, label):
    orig = getattr(mod, name)
    if getattr(orig, "_timed", False):
        return

    def f(*a, **kw):
        t0 = time.perf_counter()
        r = orig(*a, **kw)
        ACC[label] += time.perf_counter() - t0
        CNT[label] += 1
        return r
    f._timed = True
    setattr(mod, name, f)


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    d, E, L, n_q, n_kv, vocab = 512, 8, 8, 8, 2, 4096
    cfg = dict(d=d, d_ff=4 * d, E=E, layers=L, n_q=n_q, n_kv=n_kv, vocab=vocab,
               seq=S, steps=1)
    rng = np.random.default_rng(0)
    P = build(cfg, rng)
    cos, sin = V.rope_cos_sin(S, d // n_q, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    ids = rng.integers(0, vocab, S).astype(np.int64)
    tgt = np.roll(ids, -1)

    for mod, name, label in (
        (MOE, "route_topk", "route_topk(含D2H)"),
        (MOE, "moe_topk_orv", "moe_topk_orv(合并分派)"),
        (MOE, "moe_dispatch_orv", "  └ moe_dispatch_orv"),
        (V, "situ_glu_orv", "situ_glu_orv(专家+共享)"),
        (V, "rmsnorm_orv", "rmsnorm_orv"),
        (V, "attention_orv", "attention_orv"),
    ):
        wrap(mod, name, label)

    def step():
        x = P["emb"][ids].reshape(1, S, d)
        for Lw in P["layers"]:
            x = x + V.attention_orv(V.rmsnorm_orv(x, Lw["ln1"]), Lw["wq"], Lw["wk"],
                                    Lw["wv"], Lw["wo"], cos, sin, d // n_q, n_q, n_kv)
            h = x
            hn = V.rmsnorm_orv(h, Lw["ln2"]).reshape(S, d)
            e, g, _ = MOE.route_topk(hn, Lw["router"], 2)
            mo = MOE.moe_topk_orv(hn, e, g, Lw["W1"], Lw["W2"], Lw["W3"])
            mo = mo + V.situ_glu_orv(hn, Lw["S1"], Lw["S2"], Lw["S3"])
            x = h + mo.reshape(1, S, d)
        lg = V.rmsnorm_orv(x, P["ln_f"]).reshape(S, d) @ P["head"].transpose(0, 1)
        return T.functional.cross_entropy(lg, tgt)

    l = step(); l.backward(); rt.sync()          # 预热
    ACC.clear(); CNT.clear()
    t0 = time.time()
    l = step()
    l.backward()
    rt.sync()
    total = time.time() - t0
    print("S=%d 一步合计 %.3f s" % (S, total))
    for k in sorted(ACC, key=lambda x: -ACC[x]):
        print("  %-26s %7.1f ms  n=%d" % (k, ACC[k] * 1e3, CNT[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
