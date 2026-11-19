# -*- coding: utf-8 -*-
"""GPU-time micro-benchmark: current MoE dispatch vs padded-dense (capacity) dispatch.

Scale is the real one (seq2048: N=2048 tokens, k=2 -> 4096 pairs, d=1024,
d_ff=3328, E=4 routed experts) and the routing distribution is the one dumped
from a real 20-step run (moe_route_eids.npz: s2_l0..11 / s18_l0..11).

Measured with CUDA events, one pair of events around each timed block and a
single stream sync per iteration -- never "sync before + sync after a single
kernel" (that inflates small kernels; see lab/rent/selftest_event.py).

Blocks measured
  A. current dispatch  = v3_moe.moe_topk_orv (argsort on host, per-expert
     variable-shape gather_rows + 3 fp16 GEMMs + row_scale + atomic scatter)
  B. padded-dense      = fixed C = ceil(N*k/E*pad) rows per expert:
     index_select(C rows) + 3 fp16 GEMMs + gate/mask multiply + atomic scatter
  C. same GEMMs but fp32 strided-batched bmm ([E,C,d] @ [E,d,d_ff]) -- what the
     runtime's 3-D matmul does today (no fp16 batched path exists)
  G. GEMM-only: fp16 2-D vs fp32 bmm at fixed m and at the real per-expert m

The internal D2H copies are stubbed out for the *GPU-time* measurement (their
host-blocking cost was measured separately: probe_moe_d2h.py).  Stubbing keeps
every kernel shape/launch identical, so GPU time is unaffected.

Writes bench_moe_dispatch.json; prints an ASCII table.
"""
import json
import math
import os
import sys

import numpy as np

import tgai_go.train as T
from tgai_go.train import indexing as IDX
from tgai_go.train import v3_moe as MOE
from tgai_go.train import v3_ops as V
from tgai_go.train import v3_scatter as S

OUT = os.environ.get("BENCH_OUT", r"D:\orv_bench\bench_moe_dispatch.json")
NPZ = os.environ.get("BENCH_NPZ", r"D:\orv_bench\moe_route_eids.npz")
STEPS = ("s2", "s18")
PADS = (1.0, 1.25, 1.5)
WARMUP = 3
ITERS = 10

N, K, D, DFF, E = 2048, 2, 1024, 3328, 4


# ── event timing helpers ────────────────────────────────────────────────────
class Ev:
    def __init__(self, rt, n=64):
        self.rt = rt
        import ctypes
        self.ct = ctypes
        self.pool = []
        for _ in range(n):
            ev = ctypes.c_void_p()
            rt._chk(rt.cu.cuEventCreate(ctypes.byref(ev), ctypes.c_uint(0)),
                    "cuEventCreate")
            self.pool.append(ev)

    def rec(self):
        import ctypes
        ev = self.pool.pop()
        self.rt._chk(self.rt.cu.cuEventRecord(ev, self.rt._stream), "cuEventRecord")
        return ev

    def give(self, ev):
        self.pool.append(ev)

    def ms(self, a, b):
        import ctypes
        v = ctypes.c_float()
        self.rt._chk(self.rt.cu.cuEventElapsedTime(ctypes.byref(v), a, b),
                     "cuEventElapsedTime")
        return float(v.value)


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    T.amp_on(dtype="fp16", weight_cache=True)
    st = T.amp_state()
    fp16 = bool(st.get("fp16"))
    print("[bench] N=%d k=%d d=%d d_ff=%d E=%d  amp_fp16=%s  (rt.fp16_gemm=%r is a "
          "dev flag, NOT the amp path)" % (N, K, D, DFF, E, fp16,
                                           getattr(rt, "fp16_gemm", None)), flush=True)

    rng = np.random.default_rng(1234)
    # x is [N, D] (one row per token), exactly like the production model.
    # The dispatch gathers x[tok] with tok = pair // K internally.
    x_np = rng.normal(0, 0.3, (N, D)).astype(np.float32)
    gate_np = rng.uniform(0.1, 1.0, (N * K,)).astype(np.float32)
    # production shapes: W1/W2 = (d_ff, d), W3 = (d, d_ff)   (v3_train_cli:375)
    W = [[T.tensor(rng.normal(0, 0.02, (DFF, D)).astype(np.float32))
          for _ in range(E)],
         [T.tensor(rng.normal(0, 0.02, (DFF, D)).astype(np.float32))
          for _ in range(E)],
         [T.tensor(rng.normal(0, 0.02, (D, DFF)).astype(np.float32))
          for _ in range(E)]]
    for wl in W:
        for w in wl:
            w.requires_grad = True
    rt.f16_cache_pin([int(w._gpu.ptr) for wl in W for w in wl])

    x = T.tensor(x_np)
    x.requires_grad = True
    gates = T.tensor(gate_np.reshape(N, K))
    gates.requires_grad = True

    # ---- stub the D2H during timing (GPU-time only; host blocking measured
    #      separately in probe_moe_d2h.py).  Every kernel launch is unchanged. ----
    _orig_d2h = rt.d2h
    STUB = [True]

    def _stub(buf):
        if not STUB[0]:
            return _orig_d2h(buf)
        return np.zeros(tuple(int(s) for s in buf.shape), buf.dtype)

    rt.d2h = _stub

    ev = Ev(rt, 256)

    def time_block(fn, iters=ITERS, warm=WARMUP):
        for _ in range(warm):
            fn()
        rt.sync()
        a = ev.rec()
        for _ in range(iters):
            fn()
        b = ev.rec()
        rt.sync()
        ms = ev.ms(a, b)
        ev.give(a)
        ev.give(b)
        return ms / float(iters)

    def zero_grads():
        x.grad = None
        gates.grad = None
        for wl in W:
            for w in wl:
                w.grad = None

    # ---- routing distributions ----
    dists = {}
    with np.load(NPZ) as z:
        for tag in STEPS:
            layers = []
            for ly in range(12):
                key = "%s_l%d" % (tag, ly)
                if key in z.files:
                    layers.append(np.asarray(z[key], np.int64).reshape(N, K))
            dists[tag] = layers
    print("[bench] loaded routing: %s" % {k: len(v) for k, v in dists.items()},
          flush=True)

    def padded_plan(eids, pad):
        """host-side plan for the padded path (a device implementation would
        produce this with rt.moe_permute via atomic slot reservation)."""
        C = int(math.ceil(N * K / float(E) * pad))
        flat = eids.reshape(-1)
        idxs, toks, gws, cnts = [], [], [], []
        for e in range(E):
            pos = np.flatnonzero(flat == e)
            cnts.append(int(pos.size))
            arr = np.zeros(C, np.int32)          # pair ids, 0-padded
            used = pos[:C]
            arr[:used.size] = used
            tok_arr = (arr // K).astype(np.int32)  # token ids for x[tok] / scatter
            idxs.append(tok_arr)
            toks.append(tok_arr)
            gw = np.zeros(C, np.float32)
            gw[:used.size] = gate_np[used]
            gws.append(gw)
        return C, idxs, toks, gws, cnts

    # ================= A: current dispatch =================
    def fwd_A(eids):
        out = None
        for eids_l in eids:
            y = MOE.moe_topk_orv(x, eids_l, gates, W[0], W[1], W[2])
            out = y if out is None else out + y
        return out

    def run_A(eids):
        zero_grads()
        return fwd_A(eids)

    def run_A_bwd(eids):
        zero_grads()
        out = fwd_A(eids)
        out.backward()
        return out

    # ================= B: padded dense =================
    def make_fwd_B(plan_layers, gw_t):
        def fwd():
            acc = None
            for li, (idxs, toks) in enumerate(plan_layers):
                part = None
                for e in range(E):
                    rows = IDX.index_select(x, 0, idxs[e])
                    y = V.situ_glu_orv(rows, W[0][e], W[1][e], W[2][e])
                    y = y * gw_t[li][e].reshape(-1, 1)
                    z = S.scatter_add(y, toks[e], N)
                    part = z if part is None else part + z
                acc = part if acc is None else acc + part
            return acc
        return fwd

    def run_B(plan_layers, gw_t, bwd=False):
        zero_grads()
        out = make_fwd_B(plan_layers, gw_t)()
        if bwd:
            out.backward()
        return out

    # ================= C: fp32 strided-batched GEMM only =================
    def gemm_c_only(C):
        xh = rng.normal(0, 0.3, (E, C, D)).astype(np.float32)
        Wg = rng.normal(0, 0.02, (E, D, DFF)).astype(np.float32)
        Wd = rng.normal(0, 0.02, (E, DFF, D)).astype(np.float32)
        xt = T.tensor(xh)
        wg = T.tensor(Wg)
        wd = T.tensor(Wd)

        def f():
            g = T.bmm(xt, wg)
            u = T.bmm(xt, wg)
            h = g * u
            return T.bmm(h, wd)
        return f

    def gemm_f16_only(C):
        xh = rng.normal(0, 0.3, (C, D)).astype(np.float32)
        xt = T.tensor(xh)

        def f():
            for e in range(E):
                g = xt @ W[0][e].transpose(0, 1)
                u = xt @ W[1][e].transpose(0, 1)
                h = g * u
                y = h @ W[2][e].transpose(0, 1)
        return f

    res = {"config": {"N": N, "k": K, "d": D, "d_ff": DFF, "E": E,
                      "fp16_gemm": fp16, "iters": ITERS, "warmup": WARMUP},
           "notes": ["D2H stubbed inside the timed blocks (GPU time only)",
                     "routing distribution from a real 20-step run"],
           "dispatch": {}, "gemm": {}, "correctness": {}}

    # ---------- dispatch timing ----------
    for tag in STEPS:
        layers = dists.get(tag) or []
        if not layers:
            continue
        key = "%s_A" % tag
        ms_f = time_block(lambda: run_A(layers))
        ms_b = time_block(lambda: run_A_bwd(layers))
        res["dispatch"][key] = {"fwd_ms_per_12_layers": round(ms_f, 3),
                                "fwd_ms_per_layer": round(ms_f / 12.0, 3),
                                "fwd_bwd_ms_per_12_layers": round(ms_b, 3),
                                "fwd_bwd_ms_per_layer": round(ms_b / 12.0, 3)}
        print("[bench] A(%s) fwd %.2f ms/12L (%.3f/layer)  fwd+bwd %.2f ms"
              % (tag, ms_f, ms_f / 12.0, ms_b), flush=True)

        for pad in PADS:
            plans, gws, cs, drops = [], [], [], 0
            for eids_l in layers:
                C, idxs, toks, gwl, cnts = padded_plan(eids_l, pad)
                cs.append(C)
                drops += sum(max(0, c - C) for c in cnts)
                gp = [T.tensor(g.astype(np.float32)) for g in gwl]
                for t in gp:
                    t.requires_grad = False
                plans.append((idxs, toks))
                gws.append(gp)
            ms_f2 = time_block(lambda: run_B(plans, gws))
            ms_b2 = time_block(lambda: run_B(plans, gws, bwd=True))
            res["dispatch"]["%s_B_pad%.2f" % (tag, pad)] = {
                "C": cs[0], "rows_per_expert": cs[0],
                "dropped_pairs_per_12L": int(drops),
                "fwd_ms_per_12_layers": round(ms_f2, 3),
                "fwd_ms_per_layer": round(ms_f2 / 12.0, 3),
                "fwd_bwd_ms_per_12_layers": round(ms_b2, 3),
                "fwd_bwd_ms_per_layer": round(ms_b2 / 12.0, 3)}
            print("[bench] B(%s,pad=%.2f,C=%d,drop=%d) fwd %.2f ms  fwd+bwd %.2f ms"
                  % (tag, pad, cs[0], drops, ms_f2, ms_b2), flush=True)

    # ---------- GEMM-only ----------
    for C in (1024, 1280, 1536):
        ms16 = time_block(gemm_f16_only(C))
        ms32 = time_block(gemm_c_only(C))
        res["gemm"]["m%d_fp16_2d_12L" % C] = round(ms16, 3)
        res["gemm"]["m%d_fp32_bmm_12L" % C] = round(ms32, 3)
        print("[bench] GEMM-only rows=%d: 4x(fp16 2-D x3) %.2f ms/12L   "
              "fp32 bmm x3 %.2f ms/12L" % (C, ms16, ms32), flush=True)

    # ---------- correctness (forward, real distribution, step18) ----------
    STUB[0] = False
    layers = dists.get("s18") or []
    if layers:
        eids_l = layers[0]
        out_A = MOE.moe_topk_orv(x, eids_l, gates, W[0], W[1], W[2])
        ref = np.asarray(out_A.to_numpy(), np.float32)
        for pad in PADS:
            C, idxs, toks, gwl, cnts = padded_plan(eids_l, pad)
            gp = [T.tensor(g) for g in gwl]
            acc = None
            for e in range(E):
                rows = IDX.index_select(x, 0, idxs[e])
                y = V.situ_glu_orv(rows, W[0][e], W[1][e], W[2][e])
                y = y * gp[e].reshape(-1, 1)
                z = S.scatter_add(y, toks[e], N)
                acc = z if acc is None else acc + z
            got = np.asarray(acc.to_numpy(), np.float32)
            denom = np.maximum(np.abs(ref), 1e-6)
            res["correctness"]["pad%.2f" % pad] = {
                "C": int(C), "drops": int(sum(max(0, c - C) for c in cnts)),
                "max_abs": float(np.abs(got - ref).max()),
                "max_rel": float((np.abs(got - ref) / denom).max()),
                "bitwise_equal": bool(np.array_equal(got, ref))}
            print("[bench] corr pad=%.2f C=%d drop=%d max|d|=%.3e bitwise=%s"
                  % (pad, C, sum(max(0, c - C) for c in cnts),
                     float(np.abs(got - ref).max()),
                     np.array_equal(got, ref)), flush=True)

    # ---------- extreme distributions (drop risk) ----------
    extremes = {}
    one_hot = np.zeros((N, K), np.int64)                     # every token -> e0
    one_hot[:, 1] = 1
    skew = np.zeros((N, K), np.int64)                        # 70/20/10/0 split
    third = int(N * 0.7)
    skew[:third] = [0, 1]
    skew[third:third + int(N * 0.2)] = [1, 2]
    skew[third + int(N * 0.2):] = [2, 3]
    empt = (np.arange(N * K).reshape(N, K) % (E - 1)).astype(np.int64)  # e3 empty
    for name, eids in (("one_hot_e0", one_hot), ("skew_70_20_10", skew),
                       ("one_expert_empty", empt)):
        flat = eids.reshape(-1)
        cnts = [int((flat == e).sum()) for e in range(E)]
        row = {"counts": cnts, "mean": N * K / float(E)}
        for pad in PADS:
            C = int(math.ceil(N * K / float(E) * pad))
            row["pad%.2f" % pad] = {"C": C,
                                    "dropped": int(sum(max(0, c - C) for c in cnts))}
        extremes[name] = row
        print("[bench] extreme %-18s counts=%s" % (name, cnts), flush=True)
    res["extremes"] = extremes

    rt.d2h = _orig_d2h
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=True, indent=1)
    print("[bench] wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
