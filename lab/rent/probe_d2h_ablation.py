# -*- coding: utf-8 -*-
"""Ablation: how much step time do the blocking D2H copies actually cost?

Runtime-only patch of `RT.d2h`; python/tgai_go is untouched.

ABLATE=none     : real copies (control, should reproduce 0.382 s)
ABLATE=route    : the 12 int32 (N,k) routing read-backs return the *real* idx
                  recorded in a previous run (moe_route_eids.npz) without a copy
ABLATE=gatebwd  : the 12 float32 (N*k,) reads in `_array_index_cuda.bw`
                  (host fallback for `gate[order]`) return ones without a copy
ABLATE=both     : both of the above
ABLATE=scalar   : also fake the loss/gnorm scalar reads (upper bound; clip is
                  fed a fixed gnorm=1.0 so the run stays finite)

Why this is a valid upper bound: the D2H copy is replaced by an equally-shaped
host array, so *all* downstream host control flow, H2D uploads and kernel shapes
stay exactly as before; only the blocking copy+stream sync disappears.
"""
import os
import sys

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_pretrain as TR

RT = T._cuda()
_ARGV = list(sys.argv[1:])
if "--ablate" in _ARGV:
    _i = _ARGV.index("--ablate")
    os.environ["ABLATE"] = _ARGV[_i + 1]
    del _ARGV[_i:_i + 2]
    sys.argv = [sys.argv[0]] + _ARGV
MODE = os.environ.get("ABLATE", "none").strip().lower()
NPZ = os.environ.get("ABLATE_NPZ", r"D:\orv_bench\moe_route_eids.npz")

_EIDS = {}
if MODE in ("route", "both", "scalar", "all") and os.path.exists(NPZ):
    with np.load(NPZ) as z:
        for k in z.files:
            if k.startswith("s2_l"):
                _EIDS[int(k.split("_l")[1])] = z[k]

_layer = [0]
_scal = [0]
_bypassed = {"route": 0, "gatebwd": 0, "scalar": 0}
_real = {"route": 0, "gatebwd": 0, "scalar": 0}
_orig_d2h = RT.d2h
_orig_fwd = TR.forward_loss


def _d2h(buf):
    shape = tuple(int(s) for s in getattr(buf, "shape", ()))
    dt = str(getattr(buf, "dtype", "?"))
    # --- routing idx: int32 (N, k) ---
    if MODE in ("route", "both", "scalar", "all") and dt == "int32" and len(shape) == 2:
        arr = _EIDS.get(_layer[0] % max(len(_EIDS), 1)) if _EIDS else None
        if arr is not None and tuple(arr.shape) == shape:
            _layer[0] += 1
            _bypassed["route"] += 1
            return np.ascontiguousarray(arr)
    # --- gate[order] backward host fallback: float32 (N*k,) ---
    if MODE in ("gatebwd", "both", "scalar", "all") and dt == "float32" \
            and len(shape) == 1 and shape[0] == 4096:
        _bypassed["gatebwd"] += 1
        return np.ones(shape, np.float32)
    # --- loss / gnorm scalars ---
    if MODE in ("scalar", "all") and dt == "float32" and shape in ((), (1,)):
        _bypassed["scalar"] += 1
        return np.ones(shape, np.float32)
    # --- loss only: the first float32 scalar read of each step is the loss
    #     (v3_pretrain.py:1303); the second is gnorm after clip ---
    if MODE == "loss" and dt == "float32" and shape in ((), (1,)) and _scal[0] == 0:
        _scal[0] += 1
        _bypassed["scalar"] += 1
        return np.ones(shape, np.float32)
    if dt == "float32" and shape in ((), (1,)):
        _scal[0] += 1
    if dt == "int32" and len(shape) == 2:
        _real["route"] += 1
    elif dt == "float32" and len(shape) == 1 and shape[0] == 4096:
        _real["gatebwd"] += 1
    elif dt == "float32" and shape in ((), (1,)):
        _real["scalar"] += 1
    return _orig_d2h(buf)


def _fwd(*a, **kw):
    _layer[0] = 0
    _scal[0] = 0
    return _orig_fwd(*a, **kw)


RT.d2h = _d2h
TR.forward_loss = _fwd

if __name__ == "__main__":
    print("[ablate] mode=%s  routing layers=%d" % (MODE, len(_EIDS)), flush=True)
    try:
        TR.main(_ARGV)
    finally:
        print("[ablate] bypassed=%r real=%r" % (_bypassed, _real), flush=True)
