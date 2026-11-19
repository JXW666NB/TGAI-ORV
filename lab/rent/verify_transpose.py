# -*- coding: utf-8 -*-
"""验证分块转置：正确性（含非 32 倍数形状）+ 新老内核 A/B 速度。

正确性判据：与 numpy 的 `d2h(buf).T` 逐位一致（max_abs == 0）。
速度判据：同一块数据，朴素内核 vs 分块内核的 GB/s。
"""
import ctypes
import json
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()

report = {}

# ---------- ① 正确性 ----------
shapes = [(64, 64), (100, 37), (2048, 1024), (1, 17), (33, 65), (1024, 3328)]
worst = 0.0
bad = []
for rows, cols in shapes:
    src = np.random.default_rng(0).normal(size=(rows, cols)).astype(np.float32)
    buf = rt.h2d(src)
    out = rt.transpose2d(buf)
    got = np.asarray(rt.d2h(out), np.float32).reshape(cols, rows)
    diff = float(np.abs(got - src.T).max())
    worst = max(worst, diff)
    if diff != 0.0:
        bad.append([rows, cols, diff])
report["correctness_max_abs"] = worst
report["correctness_bad"] = bad

# ---------- ② 速度 A/B ----------
rows, cols = 2048, 1024
src = np.random.default_rng(1).normal(size=(rows, cols)).astype(np.float32)
buf = rt.h2d(src)
out = rt.empty((cols, rows))
n = rows * cols
gb = n * 4 * 2 / 1e9            # 读 + 写


def bench_naive(rep=50):
    grid, block = rt._ewise_grid(n)
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(rep):
        rt.launch("transpose2d", grid, block, [
            rt._ptr(buf), rt._ptr(out),
            (rows, ctypes.c_int), (cols, ctypes.c_int)])
    rt.sync()
    return (time.perf_counter() - t0) / rep


def bench_tiled(rep=50):
    tx = (cols + 31) // 32
    ty = (rows + 31) // 32
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(rep):
        rt.launch("transpose2d_tiled", (tx * ty, 1, 1), (256, 1, 1), [
            rt._ptr(buf), rt._ptr(out),
            (rows, ctypes.c_int), (cols, ctypes.c_int),
            (tx, ctypes.c_int)])
    rt.sync()
    return (time.perf_counter() - t0) / rep


t_old = bench_naive()
t_new = bench_tiled()
report["naive_ms"] = round(t_old * 1e3, 3)
report["tiled_ms"] = round(t_new * 1e3, 3)
report["naive_gbs"] = round(gb / t_old, 1)
report["tiled_gbs"] = round(gb / t_new, 1)
report["speedup"] = round(t_old / t_new, 2) if t_new else None
print("[transpose] " + json.dumps(report, ensure_ascii=False))
