# -*- coding: utf-8 -*-
"""Where do the blocking D2H copies in the MoE routing actually come from?

Runs the *unmodified* 20-step baseline (same flags as bench_stable.py) and only
monkey-patches three callables at runtime:

  RT.d2h            -> bytes / dtype / host-blocking ms / caller frames (read-only)
  MOE.route_topk    -> per-layer expert histogram (host array already produced by
                       the existing D2H, so no extra copy)
  TR.forward_loss   -> step boundary + wall clock

Nothing under python/tgai_go is edited.  Prints ASCII only; writes JSON + NPZ.

    python -X utf8 probe_moe_d2h.py --data D:\\orv_bench\\data2048_v96000 ...

Output (default D:\\orv_bench\\moe_d2h.json):
  * per-step table: d2h count / ms / bytes, route_ms, step_s
  * per-caller tag: n_per_step, ms_per_step, bytes, shape samples
  * per-call records for step 2 and step 18 (1-based)
  * per-layer expert histograms for all steps (min/median/max/CV per layer)
  * NPZ: eids[12, N, k] for step 2 and step 18 (real routing distribution
    for the offline micro-benchmark)
"""
import json
import os
import sys
import time

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_moe as MOE
from tgai_go.train import v3_pretrain as TR

RT = T._cuda()
OUT = os.environ.get("PROBE_OUT", r"D:\orv_bench\moe_d2h.json")
NPZ = os.environ.get("PROBE_NPZ", r"D:\orv_bench\moe_route_eids.npz")
SAVE_STEPS = {1, 17}          # 0-based -> "step 2" and "step 18" 1-based

STEP = [-1]
_t_step0 = [0.0]
_layer = [0]
_steps = []                   # per-step rows
D2H_TAG = {}                  # tag -> dict(n, ms, bytes, shapes set, sample)
BY_SITE = {}                  # effective call site -> dict(n, ms, bytes, shapes)
D2H_CALLS = {}                # step -> list of per-call dicts
ROUTE = {}                    # step -> list of per-layer dict
EIDS = {}                     # (step, layer) -> int32 [N,k]

_cur = {}


def _reset_step():
    _cur.clear()
    _cur.update({"n": 0, "ms": 0.0, "bytes": 0, "route_n": 0, "route_ms": 0.0,
                 "calls": []})
    _layer[0] = 0


_reset_step()

_orig_d2h = RT.d2h


def _frames(depth=4):
    out = []
    f = sys._getframe(1)
    for _ in range(depth):
        if f is None:
            break
        out.append("%s:%s:%d" % (os.path.basename(f.f_code.co_filename),
                                 f.f_code.co_name, f.f_lineno))
        f = f.f_back
    return out


def _d2h(buf):
    fr = _frames(4)
    t0 = time.perf_counter()
    out = _orig_d2h(buf)
    dt = (time.perf_counter() - t0) * 1e3
    nbytes = int(getattr(buf, "nbytes", 0))
    shape = tuple(int(s) for s in getattr(buf, "shape", ()))
    dt_name = str(getattr(buf, "dtype", "?"))
    # classify by the deepest frame that is not the to_numpy() adapter
    tag = "other"
    for s in fr:
        if "v3_moe.py:route_topk" in s:
            tag = "moe.route_topk(idx_buf.to_numpy)"
            break
        if "v3_moe.py:route_top1" in s:
            tag = "moe.route_top1(s_all.to_numpy)"
            break
        if "v3_pretrain.py" in s and "forward_loss" in s:
            tag = "loss_scalar"
            break
        if "cuda.py:_numpy_exact_gemm" in s or "numpy_exact" in s:
            tag = "other.gemm_probe"
            break
    if tag == "other" and any("clip_grad" in s for s in fr):
        tag = "clip_grad_norm_"
    if tag == "other" and any("int8" in s.lower() for s in fr):
        tag = "optimizer.int8"
    rec = D2H_TAG.setdefault(tag, {"n": 0, "ms": 0.0, "bytes": 0,
                                   "shapes": {}, "dtypes": {}, "frames": {}})
    rec["n"] += 1
    rec["ms"] += dt
    rec["bytes"] += nbytes
    rec["shapes"]["x".join(str(x) for x in shape)] = \
        rec["shapes"].get("x".join(str(x) for x in shape), 0) + 1
    rec["dtypes"][dt_name] = rec["dtypes"].get(dt_name, 0) + 1
    key = fr[0] + " <- " + fr[1]
    rec["frames"][key] = rec["frames"].get(key, 0) + 1
    site = fr[2] if len(fr) > 2 else fr[-1]
    srec = BY_SITE.setdefault(site, {"n": 0, "ms": 0.0, "bytes": 0, "shapes": {}})
    srec["n"] += 1
    srec["ms"] += dt
    srec["bytes"] += nbytes
    sk = "x".join(str(x) for x in shape) + ":" + dt_name
    srec["shapes"][sk] = srec["shapes"].get(sk, 0) + 1
    _cur["n"] += 1
    _cur["ms"] += dt
    _cur["bytes"] += nbytes
    _cur["calls"].append({"tag": tag, "bytes": nbytes, "ms": dt,
                          "shape": list(shape), "dtype": dt_name, "frames": fr})
    return out


_orig_route = MOE.route_topk


def _route_topk(hn, router_w, k=2, bias=None):
    t0 = time.perf_counter()
    eids, gates, s_all = _orig_route(hn, router_w, k=k, bias=bias)
    dt = (time.perf_counter() - t0) * 1e3
    layer = _layer[0]
    _layer[0] += 1
    n, kk = int(eids.shape[0]), int(eids.shape[1])
    E = int(router_w.shape[0]) if len(router_w.shape) == 2 else -1
    flat = np.asarray(eids, np.int64).reshape(-1)
    counts = np.bincount(flat, minlength=max(E, 0)).astype(np.int64)
    c = counts.astype(np.float64)
    mean = float(c.mean()) if c.size else 0.0
    sd = float(c.std()) if c.size else 0.0
    rec = {"layer": layer, "n": n, "k": kk, "E": E,
           "counts": [int(x) for x in counts],
           "min": int(c.min()) if c.size else 0,
           "median": float(np.median(c)) if c.size else 0.0,
           "max": int(c.max()) if c.size else 0,
           "cv": (sd / mean) if mean > 0 else 0.0,
           "route_ms": dt}
    ROUTE.setdefault(STEP[0], []).append(rec)
    _cur["route_n"] += 1
    _cur["route_ms"] += dt
    if STEP[0] in SAVE_STEPS:
        EIDS[(STEP[0], layer)] = np.asarray(eids, np.int32)
    return eids, gates, s_all


_orig_fwd = TR.forward_loss


def _forward_loss(*a, **kw):
    if STEP[0] >= 0:
        _close_step()
    STEP[0] += 1
    _reset_step()
    _t_step0[0] = time.perf_counter()
    return _orig_fwd(*a, **kw)


def _close_step():
    wall = time.perf_counter() - _t_step0[0]
    row = {"step1": STEP[0] + 1, "step0": STEP[0], "wall_s": wall,
           "d2h_n": _cur["n"], "d2h_ms": _cur["ms"], "d2h_bytes": _cur["bytes"],
           "route_n": _cur["route_n"], "route_ms": _cur["route_ms"]}
    _steps.append(row)
    if STEP[0] in SAVE_STEPS:
        D2H_CALLS[STEP[0]] = list(_cur["calls"])


RT.d2h = _d2h
MOE.route_topk = _route_topk
TR.forward_loss = _forward_loss


def _report():
    if STEP[0] >= 0:
        _close_step()
    nsteps = len(_steps)
    tags = {}
    for tag, r in D2H_TAG.items():
        tags[tag] = {"n_per_step": r["n"] / float(max(nsteps, 1)),
                     "ms_per_step": r["ms"] / float(max(nsteps, 1)),
                     "ms_per_call": (r["ms"] / r["n"]) if r["n"] else 0.0,
                     "bytes_per_step": r["bytes"] / float(max(nsteps, 1)),
                     "shapes": r["shapes"], "dtypes": r["dtypes"],
                     "frames": dict(sorted(r["frames"].items(),
                                           key=lambda kv: -kv[1])[:4])}
    sites = {}
    for site, r in BY_SITE.items():
        sites[site] = {"n_per_step": r["n"] / float(max(nsteps, 1)),
                       "ms_per_step": r["ms"] / float(max(nsteps, 1)),
                       "ms_per_call": (r["ms"] / r["n"]) if r["n"] else 0.0,
                       "bytes_per_step": r["bytes"] / float(max(nsteps, 1)),
                       "shapes": r["shapes"]}
    saved = {}
    for st in sorted(SAVE_STEPS):
        recs = ROUTE.get(st, [])
        saved["step%d" % (st + 1)] = {
            "layers": len(recs),
            "route_ms_per_layer": [round(r["route_ms"], 3) for r in recs],
            "counts": [r["counts"] for r in recs],
            "min": [r["min"] for r in recs],
            "median": [r["median"] for r in recs],
            "max": [r["max"] for r in recs],
            "cv": [round(r["cv"], 4) for r in recs],
            "d2h_calls": D2H_CALLS.get(st, []),
        }
    # per-layer aggregate across all steps
    agg = []
    for layer in range(12):
        cc = [ROUTE[st][layer]["counts"] for st in sorted(ROUTE)
              if len(ROUTE[st]) > layer]
        arr = np.asarray(cc, np.float64)
        if arr.size:
            agg.append({"layer": layer,
                        "counts_mean": [round(float(x), 1) for x in arr.mean(0)],
                        "min": int(arr.min()),
                        "median": float(np.median(arr)),
                        "max": int(arr.max()),
                        "cv": [round(float(x), 4) for x in
                               (arr.std(0) / np.maximum(arr.mean(0), 1e-9))]})
    res = {"measured_steps": nsteps,
           "steps": _steps,
           "d2h_tags": tags,
           "d2h_sites": sites,
           "route_saved": saved,
           "route_layers": agg,
           "env": {"cwd": os.getcwd(), "argv": sys.argv,
                   "orv_version": getattr(__import__("tgai_go"), "__version__", "?")}}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=True, indent=1)
    if EIDS:
        np.savez_compressed(NPZ, **{"s%d_l%d" % (st + 1, ly): v
                                    for (st, ly), v in EIDS.items()})
    print("[probe] steps=%d  d2h/step=%.1f  d2h_ms/step=%.2f  route_ms/step=%.2f"
          % (nsteps, tags.get("moe.route_topk(idx_buf.to_numpy)", {}).get(
                 "n_per_step", 0.0),
             sum(r["d2h_ms"] for r in _steps) / max(nsteps, 1),
             sum(r["route_ms"] for r in _steps) / max(nsteps, 1)))
    for k, v in sorted(tags.items(), key=lambda kv: -kv[1]["ms_per_step"]):
        print("  %-42s n=%6.2f/step  %8.2f ms/step  %8.3f ms/call  %d B"
              % (k, v["n_per_step"], v["ms_per_step"], v["ms_per_call"],
                 v["bytes_per_step"] / max(v["n_per_step"], 1e-9)))
    print("[probe] wrote", OUT, NPZ)


import atexit  # noqa: E402

atexit.register(_report)

if __name__ == "__main__":
    TR.main(sys.argv[1:])
