# -*- coding: utf-8 -*-
"""Why are weight shadows re-cast more often than they exist? (ASCII only)

Counts per step: _cast_f16_cached calls / hits / misses, f16_cache_clear calls,
_f16_cache_drop_ptr calls that actually removed a key, and the distinct (ptr,count)
keys that were cast.
"""
import atexit
import collections
import json
import os
import sys

OUT = os.environ.get("PROBE_OUT", "/root/probe_why.json")

import tgai_go.train as T          # noqa: E402
from tgai_go.train import v3_pretrain as VP   # noqa: E402

rt = T._cuda()
C = collections.Counter()
marks = []
keys_seen = collections.Counter()
orig_clip = T.clip_grad_norm_


def _snap():
    marks.append({"n": dict(C), "keys": len(keys_seen)})


def clip(*a, **kw):
    _snap()
    return orig_clip(*a, **kw)


T.clip_grad_norm_ = clip

_cached = rt._cast_f16_cached
_clear = rt.f16_cache_clear
_drop = rt._f16_cache_drop_ptr
_free = rt.free


def cast_cached(src_ptr, count):
    cache = getattr(rt, "_f16_cache", None) or {}
    k = (int(src_ptr), int(count))
    if k in cache:
        C["cached_hit"] += 1
    else:
        C["cached_miss"] += 1
        keys_seen[k] += 1
    return _cached(src_ptr, count)


def clear():
    C["clear_calls"] += 1
    C["clear_entries"] += len(getattr(rt, "_f16_cache", None) or {})
    return _clear()


def drop(ptr):
    idx = getattr(rt, "_f16_index", None) or {}
    had = int(ptr or 0) in idx
    if had:
        C["drop_removed"] += 1
        C["drop_removed_keys"] += len(idx.get(int(ptr), ()) or ())
    else:
        C["drop_noop"] += 1
    r = _drop(ptr)
    if had:
        C["drop_was_pinned_free"] += 1
    return r


def free(ptr, token=None):
    C["free_calls"] += 1
    return _free(ptr, token)


rt._cast_f16_cached = cast_cached
rt.f16_cache_clear = clear
rt._f16_cache_drop_ptr = drop
rt.free = free


def dump():
    try:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({"counts": dict(C), "marks": marks,
                       "distinct_keys": len(keys_seen),
                       "top_keys": [[list(k), v] for k, v in
                                    sorted(keys_seen.items(), key=lambda kv: -kv[1])[:12]]},
                      f, ensure_ascii=False, indent=1)
        sys.stderr.write("[probe] wrote %s\n" % OUT)
    except Exception as e:      # noqa: BLE001
        sys.stderr.write("[probe] dump failed %r\n" % (e,))


atexit.register(dump)
raise SystemExit(VP.main(sys.argv[1:]))
