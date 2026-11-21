# -*- coding: utf-8 -*-
"""Attribute every fp16 cast to the model code path that issues it (ASCII only).

For each `_cast_flat_ptr_into` / `_cast_flat_ptr(to_f16)` call we walk the stack and
record the nearest 3 frames that live in the model (not in cuda.py), plus the
source pointer and element count.  Aggregated per step, this says exactly which
activation is being re-cast and how many times.
"""
import atexit
import collections
import json
import os
import sys
import time

MODEL_HINTS = ("v3_moe", "v3_csa", "v3_pretrain", "v3_ops", "v3_layer", "v3_flash",
               "nn_ops", "indexing", "v3_scatter", "v3_masks", "amp_mlp", "the__init__")


def _is_model(fn):
    b = os.path.basename(fn)
    return ("tgai_go" in fn and "cuda.py" not in b
            and (b.endswith("__init__.py") or any(h in b for h in
                 ("v3_", "nn_ops", "indexing", "amp_mlp"))))


def site(depth=14):
    f = sys._getframe(1)
    out = []
    n = 0
    while f is not None and n < depth:
        fn = f.f_code.co_filename
        if _is_model(fn):
            out.append("%s:%d:%s" % (os.path.basename(fn), f.f_lineno,
                                     f.f_code.co_name))
            if len(out) >= 3:
                break
        f = f.f_back
        n += 1
    return " <- ".join(out) if out else "?"


MODE = os.environ.get("PROBE_MODE", "sites")
OUT = os.environ.get("PROBE_OUT", "/root/probe_sites.json")

import tgai_go.train as T          # noqa: E402
from tgai_go.train import v3_pretrain as VP   # noqa: E402

rt = T._cuda()
sites = collections.Counter()          # (site, count, pinned) -> n
by_site = collections.Counter()        # site -> casts
per_step = []
marks = []
cur = [collections.Counter(), collections.Counter()]   # [site->n], [ptr->n]

orig_clip = T.clip_grad_norm_


def _snap():
    marks.append({"t": time.perf_counter(),
                  "by_site": dict(cur[0]),
                  "casts": sum(cur[0].values()),
                  "gemm": sum(cur[1].values())})
    cur[0] = collections.Counter()
    cur[1] = collections.Counter()


def clip(*a, **kw):
    _snap()
    return orig_clip(*a, **kw)


T.clip_grad_norm_ = clip

_into = rt._cast_flat_ptr_into
_flat = rt._cast_flat_ptr
_sg = rt._sgemm_f16
pinned_ref = [set()]


def cast_into(src_ptr, count, out):
    s = site()
    p = int(src_ptr)
    tag = "P" if p in (getattr(rt, "_f16_pinned", None) or ()) else "A"
    sites[(s, int(count), tag)] += 1
    cur[0][s] += 1
    cur[1][p] += 1
    return _into(src_ptr, count, out)


def cast_flat(src_ptr, count, to_f16):
    if to_f16:
        s = site()
        p = int(src_ptr)
        tag = "P" if p in (getattr(rt, "_f16_pinned", None) or ()) else "A"
        sites[(s, int(count), tag)] += 1
        cur[0][s] += 1
        cur[1][p] += 1
    return _flat(src_ptr, count, to_f16)


rt._cast_flat_ptr_into = cast_into
rt._cast_flat_ptr = cast_flat


def dump():
    try:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({
                "sites": [[list(k), v] for k, v in
                          sorted(sites.items(), key=lambda kv: -kv[1])],
                "marks": marks,
            }, f, ensure_ascii=False, indent=1)
        sys.stderr.write("[probe] wrote %s\n" % OUT)
    except Exception as e:                      # noqa: BLE001
        sys.stderr.write("[probe] dump failed %r\n" % (e,))


atexit.register(dump)
raise SystemExit(VP.main(sys.argv[1:]))
