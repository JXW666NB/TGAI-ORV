# -*- coding: utf-8 -*-
"""只测一件事：`scatter_rows_owned` 的 grid 块数对耗时的关系（真实规模，CUDA event）。

跑法：D:\\orv_bench\\remote_py.bat D:\\orv_bench\\det_blocks_only.py D:\\orv_det_work\\new_pkg_x
"""
import ctypes
import json
import statistics
import sys

PKG = sys.argv[1] if len(sys.argv) > 1 else ""
if PKG:
    sys.path.insert(0, PKG)

import numpy as np                                              # noqa: E402

import tgai_go.train as T                                       # noqa: E402

OUT = r"D:\orv_bench\det_blocks_only.json"
CI = ctypes.c_int
SHARD = r"D:\orv_bench\data2048_v96000\shard_000.npz"


def timeit(rt, fn, warm=3, inner=20, rounds=3):
    for _ in range(warm):
        fn()
    rt.sync()
    med = []
    for _ in range(rounds):
        a, b = rt.event_create(), rt.event_create()
        rt.event_record(a)
        for _ in range(inner):
            fn()
        rt.event_record(b)
        rt.event_sync(b)
        med.append(rt.event_elapsed_ms(a, b) / float(inner))
        rt.event_destroy(a)
        rt.event_destroy(b)
    return round(statistics.median(med), 4)


def main():
    rt = T._cuda()
    rng = np.random.default_rng(5)
    D = 1024
    ids = np.load(SHARD)["input_ids"].reshape(-1).astype(np.int64)
    res = {}
    for tag, idx, n_src in (
            ("emb_2048x1024_nsrc96000", np.ascontiguousarray(ids[:2048], np.int32), 96000),
            ("moe_4096x1024_nsrc2048",
             np.repeat(np.arange(2048), 2).astype(np.int32), 2048)):
        n_rows = int(idx.size)
        g = rt.h2d(rng.standard_normal((n_rows, D)).astype(np.float32))
        ib = rt.h2d(idx, dtype=np.int32)
        ob = rt.zeros((n_src, D), dtype=np.float32)
        row = {}
        for blocks in (16, 32, 64, 128, 256):
            def go(blocks=blocks):
                rt.launch("scatter_rows_owned", (blocks, 1, 1), (256, 1, 1), [
                    rt._ptr(g), rt._ptr(ib), rt._ptr(ob),
                    (n_rows, CI), (D, CI), (n_src, CI)])
            row["owned_%d" % blocks] = timeit(rt, go)
        grid, block = rt._ewise_grid(n_rows * D)
        row["atomic"] = timeit(rt, lambda: rt.launch(
            "scatter_add_rows", grid, block,
            [rt._ptr(g), rt._ptr(ib), rt._ptr(ob), (n_rows, CI), (D, CI)]))
        row["zeros"] = timeit(rt, lambda: rt.zeros((n_src, D), dtype=np.float32))
        res[tag] = row
        print("%-26s %s" % (tag, row), flush=True)
        del g, ib, ob
        rt.sync()
        rt.pool_trim() if hasattr(rt, "pool_trim") else None
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[det_blocks_only] wrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
