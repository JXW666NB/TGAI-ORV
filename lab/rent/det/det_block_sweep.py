# -*- coding: utf-8 -*-
"""`scatter_rows_owned` 的 grid 块数扫描（CUDA event 计时，真实规模）与 det_v1 的 rms 代价。

被改的只是"哪个 block 拥有哪个目标行"，每个输出元素仍然**按源行号升序**累加
⇒ 结果与块数**逐位无关**；块数只影响"每块扫一遍 n_rows"的扫描开销。

跑法：D:\\orv_bench\\remote_py.bat D:\\orv_bench\\det_block_sweep.py D:\\orv_det_work\\new_pkg_x
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

OUT = r"D:\orv_bench\det_block_sweep.json"
SHARD = r"D:\orv_bench\data2048_v96000\shard_000.npz"
CI = ctypes.c_int


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
    out = {}
    cases = {
        "emb_2048x1024_nsrc96000": (
            np.ascontiguousarray(ids[:2048], np.int32), 2048, 96000),
        "moe_4096x1024_nsrc2048": (
            np.repeat(np.arange(2048), 2).astype(np.int32), 2048, 2048),
    }
    for tag, (idx, n_src, n_rows_src) in cases.items():
        n_rows = int(idx.size)
        g = rt.h2d(rng.standard_normal((n_rows, D)).astype(np.float32))
        ib = rt.h2d(idx, dtype=np.int32)
        outbuf = rt.zeros((n_src, D), dtype=np.float32)
        zero_ms = timeit(rt, lambda: rt.zeros((n_src, D), dtype=np.float32))
        row = {"n_rows": n_rows, "n_src": n_src, "D": D,
               "zeros_ms": zero_ms, "blocks": {}}
        for blocks in (16, 32, 48, 64, 128, 256):
            def go(blocks=blocks):
                rt.launch("scatter_rows_owned", (blocks, 1, 1), (256, 1, 1), [
                    rt._ptr(g), rt._ptr(ib), rt._ptr(outbuf),
                    (n_rows, CI), (D, CI), (n_src, CI)])
            row["blocks"][blocks] = timeit(rt, go)
        # atomic 版对照（同样不含 zeros）
        def atomic():
            grid, block = rt._ewise_grid(n_rows * D)
            rt.launch("scatter_add_rows", grid, block, [
                rt._ptr(g), rt._ptr(ib), rt._ptr(outbuf), (n_rows, CI), (D, CI)])
        row["atomic_kernel_ms"] = timeit(rt, atomic)
        out[tag] = row
        print("%s: zeros=%.4f atomic=%.4f  blocks=%s"
              % (tag, zero_ms, row["atomic_kernel_ms"],
                 {k: v for k, v in sorted(row["blocks"].items())}), flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[det_block_sweep] wrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
