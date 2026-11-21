# -*- coding: utf-8 -*-
"""Host-time attribution probe (ASCII only).

Wrap hot host-side functions with perf_counter_ns and report, per step:
calls, total ms, us/call.  Reports a nested breakdown (outer includes inner).
"""
import atexit
import collections
import json
import os
import sys
import time

MODE = os.environ.get("PROBE_MODE", "host")
OUT = os.environ.get("PROBE_OUT", "/root/probe_host.json")

import tgai_go.train as T          # noqa: E402
from tgai_go.train import v3_pretrain as VP   # noqa: E402

rt = T._cuda()
pc = time.perf_counter_ns

STAT = collections.Counter()
NS = collections.Counter()
marks = []
steps = [0]

orig_clip = T.clip_grad_norm_


def _snap():
    marks.append({"t": time.perf_counter(),
                  "n": dict(STAT), "ns": dict(NS),
                  "gemm_n": STAT.get("sgemm_f16", 0),
                  "cache": rt.f16_cache_stats() if hasattr(rt, "f16_cache_stats") else {},
                  "pool": {k: rt.pool_stats()[k] for k in ("hits", "misses")}})


def clip(*a, **kw):
    _snap()
    return orig_clip(*a, **kw)


T.clip_grad_norm_ = clip


def wrap(obj, name, label=None):
    label = label or name
    fn = getattr(obj, name)

    def inner(*a, **kw):
        t0 = pc()
        r = fn(*a, **kw)
        NS[label] += pc() - t0
        STAT[label] += 1
        return r
    setattr(obj, name, inner)


if MODE == "host":
    wrap(rt, "free")
    wrap(rt, "_f16_cache_drop_ptr")
    wrap(rt, "_sgemm_f16")
    wrap(rt, "_cast_f16_cached")
    wrap(rt, "_cast_flat_ptr_into")
    wrap(rt, "empty")
    wrap(rt, "launch")

_orig_sync = rt.sync
sync_wait = []


def sync():
    t0 = pc()
    r = _orig_sync()
    sync_wait.append((pc() - t0) / 1e9)
    return r


rt.sync = sync


def dump():
    try:
        d = {"mode": MODE, "marks": marks, "sync_wait": sync_wait,
             "host_s": {k: v / 1e9 for k, v in NS.items()},
             "calls": dict(STAT)}
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        sys.stderr.write("[probe] wrote %s\n" % OUT)
    except Exception as e:                  # noqa: BLE001
        sys.stderr.write("[probe] dump failed %r\n" % (e,))


atexit.register(dump)
raise SystemExit(VP.main(sys.argv[1:]))
