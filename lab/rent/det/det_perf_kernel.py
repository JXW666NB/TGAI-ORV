# -*- coding: utf-8 -*-
"""算子级性能：被改动的两条路径在**真实规模**下开/关的耗时。

必须真实规模：embedding 表 (96000,1024) = 393 MB、行数 2048/4096（真实 batch=1、
seq=2048 的 token 数）。用小张量验证过会被 24× 的差距打脸（历史记录）。

计时口径：`rt.event_create/record/elapsed_ms` —— **纯流内打点**，
不在每次调用前后 sync；先预热 3 次，再每轮 20 次取平均，共 3 轮取中位数。

跑法：D:\\orv_bench\\remote_py.bat D:\\orv_bench\\det_perf_kernel.py D:\\orv_det_work\\new_pkg_x
"""
import json
import statistics
import sys

PKG = sys.argv[1] if len(sys.argv) > 1 else ""
if PKG:
    sys.path.insert(0, PKG)

import numpy as np                                              # noqa: E402

import tgai_go.train as T                                       # noqa: E402

OUT = r"D:\orv_bench\det_perf_kernel.json"
SHARD = r"D:\orv_bench\data2048_v96000\shard_000.npz"


def timeit(rt, fn, warm=3, inner=20, rounds=3):
    for _ in range(warm):
        fn()
    rt.sync()
    med = []
    for _ in range(rounds):
        a = rt.event_create()
        b = rt.event_create()
        rt.event_record(a)
        for _ in range(inner):
            fn()
        rt.event_record(b)
        rt.event_sync(b)
        med.append(rt.event_elapsed_ms(a, b) / float(inner))
        rt.event_destroy(a)
        rt.event_destroy(b)
    return {"median_ms": round(statistics.median(med), 4),
            "min_ms": round(min(med), 4), "max_ms": round(max(med), 4),
            "rounds_ms": [round(x, 4) for x in med]}


def main():
    rt = T._cuda()
    rng = np.random.default_rng(5)
    V, D = 96000, 1024
    ids_all = np.load(SHARD)["input_ids"].reshape(-1).astype(np.int64)
    out = {"config": {"V": V, "D": D, "shard": SHARD,
                      "n_tokens_avail": int(ids_all.size)}}

    # ---- 1) embedding 反向：real ids，2048 token ----
    toks = np.asarray(ids_all[:2048], np.int64)
    out["config"]["uniq_ids_in_window"] = int(len(np.unique(toks)))
    idx = np.ascontiguousarray(toks, np.int32)
    g = rt.h2d(rng.standard_normal((idx.size, D)).astype(np.float32))
    ib = rt.h2d(idx, dtype=np.int32)
    old = getattr(rt, "deterministic", False)
    try:
        rt.deterministic = False
        off = timeit(rt, lambda: rt.scatter_add_rows(g, ib, V, D, int(idx.size)))
        rt.deterministic = True
        on = timeit(rt, lambda: rt.scatter_add_rows(g, ib, V, D, int(idx.size)))
    finally:
        rt.deterministic = old
    out["emb_scatter_2048x1024"] = {"off": off, "on": on}

    # ---- 2) MoE permute 的 gather 反向：2048 token × k=2 = 4096 行，目标 2048 行 ----
    tok = np.repeat(np.arange(2048), 2).astype(np.int32)
    gm = rt.h2d(rng.standard_normal((tok.size, D)).astype(np.float32))
    tb = rt.h2d(tok, dtype=np.int32)
    try:
        rt.deterministic = False
        off = timeit(rt, lambda: rt.scatter_add_rows(gm, tb, 2048, D, int(tok.size)))
        rt.deterministic = True
        on = timeit(rt, lambda: rt.scatter_add_rows(gm, tb, 2048, D, int(tok.size)))
    finally:
        rt.deterministic = old
    out["moe_gather_bwd_4096x1024"] = {"off": off, "on": on}

    # ---- 3) RMSNorm 反向：(2048,1024) ----
    x = rt.h2d(rng.standard_normal((2048, D)).astype(np.float32))
    dy = rt.h2d(rng.standard_normal((2048, D)).astype(np.float32))
    w = rt.h2d(np.ones(D, np.float32))
    try:
        rt.deterministic = False
        off = timeit(rt, lambda: rt.rmsnorm_bwd(dy, x, w, 2048, D, 1e-6))
        rt.deterministic = True
        on = timeit(rt, lambda: rt.rmsnorm_bwd(dy, x, w, 2048, D, 1e-6))
    finally:
        rt.deterministic = old
    out["rmsnorm_bwd_2048x1024"] = {"off": off, "on": on}

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for k, v in out.items():
        if k == "config":
            print("config: %s" % (v,))
            continue
        o, n = v["off"]["median_ms"], v["on"]["median_ms"]
        print("%-26s off %.4f ms (%.4f..%.4f)  on %.4f ms (%.4f..%.4f)  Δ %+.4f ms (%+.1f%%)"
              % (k, o, v["off"]["min_ms"], v["off"]["max_ms"],
                 n, v["on"]["min_ms"], v["on"]["max_ms"], n - o,
                 100.0 * (n - o) / o if o else 0.0))
    print("[det_perf_kernel] wrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
