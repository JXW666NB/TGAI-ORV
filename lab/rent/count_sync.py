# -*- coding: utf-8 -*-
"""数一数：每步有多少次主机同步(sync)、多少次 d2h、多少次内核发射。

假设（要验证）：MoE 每层有一次主机往返。若 sync 次数随层数固定、而总耗时随 seq 涨，
说明"每次往返的代价"随 token 变多而变贵（GPU 队列更长 ⇒ 排空更慢），
这正是长序列 forward 涨 4.9 倍的一种可能机制。
"""
import atexit
import json
import sys
import time

import tgai_go.train as T
from tgai_go.train import v3_pretrain as TR

rt = T._cuda()
cnt = {"sync": 0, "d2h": 0, "h2d": 0, "launch": 0, "sync_s": 0.0,
       "h2d_bytes": 0, "d2h_bytes": 0}

_sync = rt.sync
_d2h = rt.d2h
_h2d = rt.h2d
_launch = getattr(rt, "launch", None)


def sync_timed(*a, **kw):
    cnt["sync"] += 1
    t0 = time.perf_counter()
    out = _sync(*a, **kw)
    cnt["sync_s"] += time.perf_counter() - t0
    return out


def d2h_timed(*a, **kw):
    cnt["d2h"] += 1
    try:                       # 统计传输字节（a[0] 是 GpuBuf）
        cnt["d2h_bytes"] += int(getattr(a[0], "n", 0)) * 4
    except Exception:
        pass
    return _d2h(*a, **kw)


def h2d_timed(*a, **kw):
    cnt["h2d"] += 1
    try:                       # 统计传输字节（a[0] 通常是 numpy 数组）
        cnt["h2d_bytes"] += int(getattr(a[0], "nbytes", 0))
    except Exception:
        pass
    return _h2d(*a, **kw)


def launch_timed(*a, **kw):
    cnt["launch"] += 1
    return _launch(*a, **kw)


rt.sync = sync_timed
rt.d2h = d2h_timed
rt.h2d = h2d_timed
if _launch is not None:
    rt.launch = launch_timed


def _report():
    print("[counts] " + json.dumps(cnt), flush=True)


atexit.register(_report)

if __name__ == "__main__":
    TR.main(sys.argv[1:])
