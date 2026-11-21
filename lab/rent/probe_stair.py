# -*- coding: utf-8 -*-
"""Staircase ablation: skip the f32->f16 cast launch for a controlled fraction of
fp16 GEMMs, and measure step time vs number of casts actually issued (ASCII only).

PROBE_SKIP=k  ->  for every 8 consecutive fp16 GEMMs, the first k get NO cast launch
                  (numerics become garbage; shapes and kernel count change only by
                  the skipped casts).  k=0 is the real thing, k=8 is "no casts".
"""
import atexit
import json
import os
import sys

SKIP = int(os.environ.get("PROBE_SKIP", "0"))
OUT = os.environ.get("PROBE_OUT", "/root/stair.json")

import numpy as np   # noqa: E402

import tgai_go.train as T          # noqa: E402
from tgai_go.train import v3_pretrain as VP   # noqa: E402

rt = T._cuda()
state = {"i": 0, "skipping": False, "casts": 0, "gemms": 0}

_into = rt._cast_flat_ptr_into
_flat = rt._cast_flat_ptr
_sg = rt._sgemm_f16
_scratch = rt._f16_scratch


def cast_into(src_ptr, count, out):
    if state["skipping"]:
        return out
    state["casts"] += 1
    return _into(src_ptr, count, out)


def cast_flat(src_ptr, count, to_f16):
    if state["skipping"] and to_f16:
        return rt.empty((int(count),), dtype=np.float16)
    if to_f16:
        state["casts"] += 1
    return _flat(src_ptr, count, to_f16)


def sgemm_f16(transa, transb, m, n, k, A, lda, B, ldb, C, ldc):
    i = state["i"]
    state["i"] = i + 1
    state["gemms"] += 1
    state["skipping"] = SKIP > 0 and (i % 8) < SKIP
    try:
        return _sg(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)
    finally:
        state["skipping"] = False


rt._cast_flat_ptr_into = cast_into
rt._cast_flat_ptr = cast_flat
rt._sgemm_f16 = sgemm_f16


def dump():
    try:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({"skip": SKIP, "casts": state["casts"],
                       "gemms": state["gemms"]}, f)
    except Exception:      # noqa: BLE001
        pass


atexit.register(dump)
raise SystemExit(VP.main(sys.argv[1:]))
