# -*- coding: utf-8 -*-
"""cuda.py 新增 event 计时 API 的最小自检（在真 GPU 上跑）。

跑法（远端）：
    D:\\orv_pip_0962\\Scripts\\python.exe D:\\orv_bench\\selftest_event.py

它只依赖 `tgai_go.train`：
  1. `rt.event_selftest()` 自带判据（数值/量级/event 总和 vs 墙钟）；
  2. 再手工做一次「N 次 add 的 event 总和 vs sync 前后墙钟」对照，并把
     「sync 包裹法」逐次计时的结果一并打出来做反面对照 —— 这就是历史坑的量化证据。
"""
import json
import sys
import time

import numpy as np

import tgai_go.train as T

rt = T._cuda()


def main():
    print("version:", getattr(T, "__version__", "?"))
    print("device :", rt.device_caps())

    # ---- 1) 库自带自检 ----
    res = rt.event_selftest(iters=100, elems=(1 << 20))
    print("SELFTEST " + json.dumps(res, sort_keys=True))

    # ---- 2) 手工对照：event 总和 vs 墙钟 ----
    elems = 1 << 20
    a = rt.full((elems,), 1.0)
    b = rt.full((elems,), 2.0)
    for _ in range(5):
        rt.add(a, b)
    rt.sync()

    n = 200
    evs = [rt.event_create() for _ in range(2 * n)]
    t0 = time.perf_counter()
    for i in range(n):
        rt.event_record(evs[2 * i])
        rt.add(a, b)
        rt.event_record(evs[2 * i + 1])
    rt.sync()
    wall = (time.perf_counter() - t0) * 1000.0
    tot = 0.0
    for i in range(n):
        tot += rt.event_elapsed_ms(evs[2 * i], evs[2 * i + 1])
    for ev in evs:
        rt.event_destroy(ev)
    print("BURST " + json.dumps({
        "n": n, "event_sum_ms": tot, "wall_ms": wall,
        "avg_event_ms": tot / n, "ratio": (tot / wall) if wall else None}))

    # ---- 3) 反面对照：sync 包裹法逐次计时（历史坑）----
    m = 50
    t1 = time.perf_counter()
    for _ in range(m):
        rt.sync()
        t_s = time.perf_counter()
        rt.add(a, b)
        rt.sync()
        t_e = time.perf_counter()
    per_sync_ms = ((time.perf_counter() - t1) / m) * 1000.0
    print("SYNCWRAP " + json.dumps({
        "n": m, "per_call_ms": per_sync_ms,
        "inflation_x": (per_sync_ms / (tot / n)) if tot else None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
