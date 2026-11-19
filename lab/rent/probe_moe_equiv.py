# -*- coding: utf-8 -*-
"""Sanity check the padded-dense plumbing against a numpy reference.

Tiny case (E=2, small D/DFF, no fp16) so the numpy reference is meaningful:
  ref[t] = sum_s situ_glu(x[t], W[eids[t,s]]) * gate[t,s]
Path B (padded-dense) must reproduce it to ~1e-10 (fp32 GEMM is bitwise the
same numpy path at these shapes).  Path A (current dispatch) is compared too.

Run:  python -X utf8 probe_moe_equiv.py
"""
import math

import numpy as np

import tgai_go.train as T
from tgai_go.train import indexing as IDX
from tgai_go.train import v3_moe as MOE
from tgai_go.train import v3_ops as V
from tgai_go.train import v3_scatter as S


def build(N, D, DFF, E, seed=0):
    rng = np.random.default_rng(seed)
    xh = rng.normal(0, .3, (N, D)).astype(np.float32)          # [N, D] token-major
    gh = rng.uniform(.1, 1., (N, 2)).astype(np.float32)        # [N, k=2]
    eids = rng.integers(0, E, (N, 2)).astype(np.int64)
    Wn = [[rng.normal(0, .02, (DFF, D)).astype(np.float32) for _ in range(E)],
          [rng.normal(0, .02, (DFF, D)).astype(np.float32) for _ in range(E)],
          [rng.normal(0, .02, (D, DFF)).astype(np.float32) for _ in range(E)]]
    W = [[T.tensor(w) for w in wl] for wl in Wn]
    return xh, gh, eids, Wn, W, T.tensor(xh), T.tensor(gh)


def np_ref(xh, gh, eids, Wn, N):
    out = np.zeros((N, xh.shape[1]), np.float32)
    for t in range(N):
        for s in range(eids.shape[1]):
            e = int(eids[t, s])
            xr = xh[t:t + 1]
            g = xr @ Wn[0][e].T
            u = xr @ Wn[1][e].T
            gate = (4.0 * np.tanh(g / 4.0)) * (0.5 * (1.0 + np.tanh(0.5 * g)))
            up = 25.0 * np.tanh(u / 25.0)
            out[t] += ((gate * up) @ Wn[2][e].T)[0] * gh[t, s]
    return out


def run_B(x, eids, gh, W, N, K, E, D, pad):
    C = int(math.ceil(N * K / float(E) * pad))
    flat = eids.reshape(-1)
    acc = None
    for e in range(E):
        pos = np.flatnonzero(flat == e)
        arr = np.zeros(C, np.int32)
        used = pos[:C]
        arr[:used.size] = used
        tok = (arr // K).astype(np.int32)
        gw = np.zeros(C, np.float32)
        gw[:used.size] = gh.reshape(-1)[used]
        rows = IDX.index_select(x, 0, tok)
        y = V.situ_glu_orv(rows, W[0][e], W[1][e], W[2][e])
        y = y * T.tensor(gw).reshape(-1, 1)
        z = S.scatter_add(y, tok, N)
        acc = z if acc is None else acc + z
    return np.asarray(acc.to_numpy())


def main():
    T.set_default_device("cuda")
    T.amp_on(dtype="fp16", weight_cache=True)

    N, D, DFF, E = 8, 16, 24, 2
    xh, gh, eids, Wn, W, x, gates = build(N, D, DFF, E, seed=1)
    ref = np_ref(xh, gh, eids, Wn, N)
    a = np.asarray(MOE.moe_topk_orv(x, eids, gates, W[0], W[1], W[2]).to_numpy())
    print("[tiny] N=%d D=%d DFF=%d E=%d k=2" % (N, D, DFF, E))
    print("[tiny] max|A-ref| = %.3e   max|ref| = %.3e" % (np.abs(a - ref).max(),
                                                          np.abs(ref).max()))
    for pad in (1.0, 1.5):
        b = run_B(x, eids, gh, W, N, 2, E, D, pad)
        print("[tiny] pad=%.2f  max|B-ref| = %.3e  max|A-B| = %.3e"
              % (pad, np.abs(b - ref).max(), np.abs(a - b).max()))

    # real scale: A vs B only (numpy reference would be a 30 s fp32 loop)
    N2, K2, D2, DFF2, E2 = 2048, 2, 1024, 3328, 4
    xh2, gh2, eids2, Wn2, W2, x2, g2 = build(N2, D2, DFF2, E2, seed=3)
    a2 = np.asarray(MOE.moe_topk_orv(x2, eids2, g2, W2[0], W2[1], W2[2]).to_numpy())
    for pad in (1.0, 1.25, 1.5):
        b2 = run_B(x2, eids2, gh2, W2, N2, K2, E2, D2, pad)
        d = np.abs(a2 - b2)
        C = int(math.ceil(N2 * K2 / float(E2) * pad))
        drop = sum(max(0, int((eids2.reshape(-1) == e).sum()) - C) for e in range(E2))
        print("[real] pad=%.2f C=%4d drop=%4d  max|A-B|=%.3e  mean|A-B|=%.3e  "
              "|A|max=%.3e" % (pad, C, drop, d.max(), d.mean(), np.abs(a2).max()))


if __name__ == "__main__":
    main()
