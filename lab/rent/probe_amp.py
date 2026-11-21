# -*- coding: utf-8 -*-
"""AMP instrumentation probe for ORV v3_pretrain (ASCII only).

Env:
  PROBE_MODE=count|time|off   count also records kernel-name counters
  PROBE_NO_CAST=1             suppress every f32->f16 cast launch (timing only,
                              numerics become garbage; shapes are unchanged)
  PROBE_OUT=/root/probe.json

Usage: PROBE_MODE=time python probe_amp.py <v3_pretrain args...>
"""
import atexit
import collections
import json
import os
import sys
import time

import numpy as np   # noqa: E402

MODE = os.environ.get("PROBE_MODE", "time")
NO_CAST = os.environ.get("PROBE_NO_CAST", "0") == "1"
OUT = os.environ.get("PROBE_OUT", "/root/probe_amp.json")

import tgai_go.train as T          # noqa: E402
from tgai_go.train import v3_pretrain as VP   # noqa: E402

rt = T._cuda()

counts = collections.Counter()
cast_keys = collections.Counter()      # (src_ptr, count) -> n
cast_by_count = collections.Counter()  # element count -> n
cast_pinned = [0, 0]                   # [pinned, unpinned]
gemm_ops = collections.Counter()       # (m, n, k, a_pin, b_pin) -> n
gemm_n = [0]
marks = []
sync_wait = []
chit = [0]
cmiss = [0]

orig_clip = T.clip_grad_norm_


def _snap(tag):
    ps = rt.pool_stats()
    marks.append({
        "tag": tag,
        "t": time.perf_counter(),
        "counts": dict(counts) if MODE == "count" else {},
        "cast_keys": dict(cast_keys) if MODE == "count" else {},
        "cast_pinned": list(cast_pinned),
        "chit": chit[0], "cmiss": cmiss[0],
        "gemm_n": gemm_n[0],
        "pool_misses": ps["misses"],
        "pool_hits": ps["hits"],
        "pool_live": ps["live_bytes"],
        "pool_cached": ps["cached_bytes"],
        "cache": rt.f16_cache_stats() if hasattr(rt, "f16_cache_stats") else {},
    })


def clip(*a, **kw):
    _snap("pre_clip")
    return orig_clip(*a, **kw)


T.clip_grad_norm_ = clip

# ---- cast suppression (independent of MODE) ----
if NO_CAST:
    _orig_cast_into = rt._cast_flat_ptr_into
    _orig_cast_flat = rt._cast_flat_ptr
    _orig_cast_buf_into = rt.cast_f32_to_f16_into

    def cast_into_noop(src_ptr, count, out):
        return out

    def cast_flat_noop(src_ptr, count, to_f16):
        if not to_f16:
            return _orig_cast_flat(src_ptr, count, to_f16)
        return rt.empty((int(count),), dtype=np.float16)

    def cast_buf_into_noop(src, dst):
        return dst

    rt._cast_flat_ptr_into = cast_into_noop
    rt._cast_flat_ptr = cast_flat_noop
    rt.cast_f32_to_f16_into = cast_buf_into_noop

if MODE == "count":
    orig_launch = rt.launch
    orig_sgemm_f16 = rt._sgemm_f16
    orig_cast_into_c = rt._cast_flat_ptr_into
    orig_cast_flat_c = rt._cast_flat_ptr
    orig_cast_buf_into_c = rt.cast_f32_to_f16_into

    def launch(name, grid, block, args, shared=0, module=None):
        counts[name] += 1
        return orig_launch(name, grid, block, args, shared=shared, module=module)

    def _rec(src_ptr, count):
        cast_keys[(int(src_ptr), int(count))] += 1
        cast_by_count[int(count)] += 1
        pinned = getattr(rt, "_f16_pinned", None) or ()
        cast_pinned[0 if int(src_ptr) in pinned else 1] += 1

    def sgemm_f16(transa, transb, m, n, k, A, lda, B, ldb, C, ldc):
        pinned = getattr(rt, "_f16_pinned", None) or ()
        ap = 1 if int(A) in pinned else 0
        bp = 1 if int(B) in pinned else 0
        gemm_ops[(int(m), int(n), int(k), ap, bp)] += 1
        gemm_n[0] += 1
        return orig_sgemm_f16(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)

    def cast_into(src_ptr, count, out):
        _rec(src_ptr, count)
        return orig_cast_into_c(src_ptr, count, out)

    def cast_flat(src_ptr, count, to_f16):
        if to_f16:
            _rec(src_ptr, count)
        return orig_cast_flat_c(src_ptr, count, to_f16)

    def cast_buf_into(src, dst):
        _rec(src.ptr, src.n)
        return orig_cast_buf_into_c(src, dst)

    orig_cached = rt._cast_f16_cached
    orig_scratch = rt._f16_scratch

    def cast_cached(src_ptr, count):
        cache = getattr(rt, "_f16_cache", None) or {}
        if (int(src_ptr), int(count)) in cache:
            chit[0] += 1
        else:
            cmiss[0] += 1
        return orig_cached(src_ptr, count)

    def scratch(slot, n):
        out = orig_scratch(slot, n)
        return out

    rt._cast_f16_cached = cast_cached
    rt.launch = launch
    rt._sgemm_f16 = sgemm_f16
    if not NO_CAST:
        rt._cast_flat_ptr_into = cast_into
        rt._cast_flat_ptr = cast_flat
        rt.cast_f32_to_f16_into = cast_buf_into

_orig_sync = rt.sync


def sync():
    t0 = time.perf_counter()
    r = _orig_sync()
    sync_wait.append(time.perf_counter() - t0)
    return r


rt.sync = sync


def _repeat_hist():
    h = collections.Counter()
    for (ptr, cnt), n in cast_keys.items():
        h[n] += 1
    return sorted(h.items())


def dump():
    try:
        d = {
            "mode": MODE, "no_cast": NO_CAST,
            "counts": dict(counts),
            "gemm_n": gemm_n[0],
            "gemm_ops": [[list(k), v] for k, v in
                         sorted(gemm_ops.items(), key=lambda kv: -kv[1])],
            "n_cast_keys": len(cast_keys),
            "n_cast_calls": sum(cast_keys.values()),
            "cast_pinned": cast_pinned,
            "cast_by_count": sorted(cast_by_count.items(), key=lambda kv: -kv[1])[:40],
            "repeat_hist": _repeat_hist(),
            "marks": marks,
            "sync_wait": sync_wait,
            "cache_state": rt.f16_cache_stats() if hasattr(rt, "f16_cache_stats") else {},
            "pool": rt.pool_stats(),
        }
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        sys.stderr.write("[probe] wrote %s\n" % OUT)
    except Exception as e:                      # noqa: BLE001
        sys.stderr.write("[probe] dump failed: %r\n" % (e,))


atexit.register(dump)

raise SystemExit(VP.main(sys.argv[1:]))
