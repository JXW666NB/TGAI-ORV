# -*- coding: utf-8 -*-
"""统计一步里每个 GEMM 形状的调用次数与累计耗时，找出反向的小 GEMM 大头。"""
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE  # noqa: E402
from v3_orv_train import build, params_of  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    d, S, E, L, n_q, n_kv = 512, 2048, 8, 8, 8, 2
    d_k = d // n_q
    cfg = dict(d=d, d_ff=4 * d, E=E, layers=L, n_q=n_q, n_kv=n_kv, vocab=4096,
               seq=S, steps=1)
    rng = np.random.default_rng(0)
    P = build(cfg, rng)
    cos, sin = V.rope_cos_sin(S, d_k, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    ids = rng.integers(0, 4096, S).astype(np.int64)
    tgt = np.roll(ids, -1)

    stats = defaultdict(lambda: [0, 0.0])

    def wrap(name):
        orig = getattr(rt, name, None)
        if orig is None:
            return
        def f(*a, **kw):
            key = "%s%s" % (name, tuple(a[2:5]) if len(a) >= 5 else ())
            t0 = time.perf_counter()
            r = orig(*a, **kw)
            rt.sync()
            s = stats[key]
            s[0] += 1
            s[1] += time.perf_counter() - t0
            return r
        setattr(rt, name, f)

    for nm in ("_sgemm", "sgemm_strided_batched", "sgemm_nt_strided_batched",
               "sgemm_tn_strided_batched"):
        wrap(nm)

    # 再按**内核名**统计（含逐元素/scatter/gather），找出非 GEMM 的大头
    kstats = defaultdict(lambda: [0, 0.0])
    _orig_launch = rt.launch

    def _launch(name, grid, block, args, *a, **kw):
        t0 = time.perf_counter()
        r = _orig_launch(name, grid, block, args, *a, **kw)
        rt.sync()
        s = kstats[name]
        s[0] += 1
        s[1] += time.perf_counter() - t0
        return r

    rt.launch = _launch

    only_bwd = {"on": False}

    def fwd():
        x = P["emb"][ids].reshape(1, S, d)
        for Lw in P["layers"]:
            x = V.attention_orv(V.rmsnorm_orv(x, Lw["ln1"]), Lw["wq"], Lw["wk"],
                                Lw["wv"], Lw["wo"], cos, sin, d_k, n_q, n_kv)
            h = x
            hn = V.rmsnorm_orv(h, Lw["ln2"]).reshape(S, d)
            e, g, _ = MOE.route_topk(hn, Lw["router"], 2)
            mo = MOE.moe_topk_orv(hn, e, g, Lw["W1"], Lw["W2"], Lw["W3"])
            mo = mo + V.situ_glu_orv(hn, Lw["S1"], Lw["S2"], Lw["S3"])
            x = h + mo.reshape(1, S, d)
        lg = V.rmsnorm_orv(x, P["ln_f"]).reshape(S, d) @ P["head"].transpose(0, 1)
        return T.functional.cross_entropy(lg, tgt)

    fwd().backward()                     # 预热
    rt.sync()
    stats.clear()
    t0 = time.time()
    loss = fwd()
    fwd_t = time.time() - t0
    f0 = sum(v[1] for v in stats.values())
    stats.clear()
    kstats.clear()
    t0 = time.time()
    loss.backward()
    rt.sync()
    bwd_t = time.time() - t0
    f1 = sum(v[1] for v in stats.values())

    print("前向总 %.3fs，其中 GEMM %.3fs；反向总 %.3fs，其中 GEMM %.3fs"
          % (fwd_t, f0, bwd_t, f1))
    print("反向：按内核累计耗时 top12")
    for k, v in sorted(kstats.items(), key=lambda kv: -kv[1][1])[:12]:
        print("  %-34s n=%-6d %.3fs" % (k[:34], v[0], v[1]))
    print("反向：内核发射总数 %d" % sum(v[0] for v in kstats.values()))
    print("按累计耗时 top10（反向）：")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1][1])[:10]:
        print("  %-46s n=%-5d %.3fs" % (k[:46], v[0], v[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
