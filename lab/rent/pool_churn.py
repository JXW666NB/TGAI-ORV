# -*- coding: utf-8 -*-
"""独立交叉校验：**不挂任何 event 计时**，只统计「每步的显存分配次数与主机耗时」。

为什么需要它：`prof_event_kernels.py` 用 event 打点，虽然不阻塞流，但仍会改变
时序。要把「分配器抖动」这个结论坐实，必须有**换了探针也能复现**的证据。

这里只做三件事：
  1. 给 `cuMemAlloc` / `cuMemFree` / `cuMemGetInfo(_v2)` 包一层计数器
     （只读 `time.perf_counter()`，不记 event、不同步流）；
  2. 用 `forward_loss` 进入点切分「步」；
  3. 每步报一次：池 hits/misses/trims 增量、三个驱动调用的次数与主机毫秒。

跑法（远端）：
    D:\\orv_pip_0962\\Scripts\\python.exe D:\\orv_bench\\pool_churn.py --data ... --seq 2048 ...
"""
import atexit
import json
import os
import sys
import time

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_pretrain as TR

RT = T._cuda()
OUT = os.environ.get("CHURN_OUT", r"D:\orv_bench\pool_churn.json")
SKIP = int(os.environ.get("CHURN_SKIP", "3"))

NAMES = ("cuMemAlloc", "cuMemFree", "cuMemGetInfo_v2", "cuMemGetInfo")
ACC = {n: [0, 0.0] for n in NAMES}
_rows = []
_prev = [None]
_last_t = [None]


class _Count(object):
    __slots__ = ("fn", "name")

    def __init__(self, fn, name):
        self.fn = fn
        self.name = name

    def __call__(self, *a):
        t0 = time.perf_counter()
        rc = self.fn(*a)
        dt = (time.perf_counter() - t0) * 1000.0
        st = ACC[self.name]
        st[0] += 1
        st[1] += dt
        return rc

    @property
    def argtypes(self):
        return self.fn.argtypes

    @argtypes.setter
    def argtypes(self, v):
        self.fn.argtypes = v

    @property
    def restype(self):
        return self.fn.restype

    @restype.setter
    def restype(self, v):
        self.fn.restype = v


def _install():
    for nm in NAMES:
        fn = getattr(RT.cu, nm, None)
        if fn is not None:
            setattr(RT.cu, nm, _Count(fn, nm))


def _snap():
    st = RT.pool_stats()
    return (int(st["hits"]), int(st["misses"]), int(st["trims"]),
            int(st["live_bytes"]), int(st["cached_bytes"]))


def _close_step():
    """把「上一个 forward_loss 以来」的增量结算成一行。"""
    cur = _snap()
    now = time.perf_counter()
    dt = (now - _last_t[0]) if _last_t[0] is not None else None
    acc = {n: list(v) for n, v in ACC.items()}
    p = _prev[0]
    _prev[0] = cur
    _last_t[0] = now
    for v in ACC.values():
        v[0] = 0
        v[1] = 0.0
    if p is None or dt is None:
        return
    _rows.append({
        "dt": dt,
        "dhits": cur[0] - p[0], "dmisses": cur[1] - p[1], "dtrims": cur[2] - p[2],
        "live_gib": cur[3] / float(1 << 30), "cached_gib": cur[4] / float(1 << 30),
        "alloc_n": acc["cuMemAlloc"][0], "alloc_ms": acc["cuMemAlloc"][1],
        "free_n": acc["cuMemFree"][0], "free_ms": acc["cuMemFree"][1],
        "info_n": acc["cuMemGetInfo_v2"][0] + acc["cuMemGetInfo"][0],
        "info_ms": acc["cuMemGetInfo_v2"][1] + acc["cuMemGetInfo"][1],
    })


_orig_fl = TR.forward_loss


def _fl(*a, **kw):
    if _prev[0] is not None:
        _close_step()
    else:
        _prev[0] = _snap()
        _last_t[0] = time.perf_counter()
        for v in ACC.values():
            v[0] = 0
            v[1] = 0.0
    return _orig_fl(*a, **kw)


TR.forward_loss = _fl


def _report():
    rows = _rows[SKIP:]
    if not rows:
        print("[churn] 没有样本", flush=True)
        return
    n = float(len(rows))

    def avg(k):
        return sum(r[k] for r in rows) / n

    res = {
        "steps": int(len(rows)),
        "wall_s": avg("dt"),
        "hits": avg("dhits"), "misses": avg("dmisses"), "trims": avg("dtrims"),
        "hit_rate": (avg("dhits") / (avg("dhits") + avg("dmisses"))
                     if (avg("dhits") + avg("dmisses")) else 0.0),
        "live_gib": avg("live_gib"), "cached_gib": avg("cached_gib"),
        "cuMemAlloc_n": avg("alloc_n"), "cuMemAlloc_ms": avg("alloc_ms"),
        "cuMemFree_n": avg("free_n"), "cuMemFree_ms": avg("free_ms"),
        "cuMemGetInfo_n": avg("info_n"), "cuMemGetInfo_ms": avg("info_ms"),
        "alloc_free_ms": avg("alloc_ms") + avg("free_ms") + avg("info_ms"),
    }
    res["alloc_free_pct_of_step"] = (100.0 * res["alloc_free_ms"]
                                     / (res["wall_s"] * 1000.0))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[churn] " + json.dumps(res, sort_keys=True), flush=True)


atexit.register(_report)

if __name__ == "__main__":
    _install()
    TR.main(sys.argv[1:])
