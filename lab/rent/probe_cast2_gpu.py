# -*- coding: utf-8 -*-
"""Bit-exactness of the fused 2-segment cast (ASCII only).

Proves `_cast2_flat_ptr_into` == two `_cast_flat_ptr_into` calls, elementwise,
including an odd length and two different lengths.
"""
import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
rng = np.random.default_rng(0)
fails = []


def check(name, cond, extra=""):
    print("  %-46s %s %s" % (name, "PASS" if cond else "FAIL", "" if cond else extra))
    if not cond:
        fails.append(name)


def one(n0, n1):
    x0 = rng.normal(0, 3, n0).astype(np.float32)
    x1 = rng.normal(0, 5, n1).astype(np.float32)
    a0 = rt.h2d(x0.reshape(1, n0))
    a1 = rt.h2d(x1.reshape(1, n1))
    r0 = rt.empty((n0,), dtype=np.float16)
    r1 = rt.empty((n1,), dtype=np.float16)
    rt._cast_flat_ptr_into(a0.ptr, n0, r0)
    rt._cast_flat_ptr_into(a1.ptr, n1, r1)
    s0 = rt.d2h(r0).copy()
    s1 = rt.d2h(r1).copy()
    f0 = rt.empty((n0,), dtype=np.float16)
    f1 = rt.empty((n1,), dtype=np.float16)
    # poison then fused
    rt._cast_flat_ptr_into(rt.h2d(np.zeros((1, n0), np.float32)).ptr, n0, f0)
    rt._cast_flat_ptr_into(rt.h2d(np.zeros((1, n1), np.float32)).ptr, n1, f1)
    rt._cast2_flat_ptr_into(a0.ptr, n0, f0, a1.ptr, n1, f1)
    g0 = rt.d2h(f0).copy()
    g1 = rt.d2h(f1).copy()
    check("fused == 2 separate (n0=%d n1=%d) A" % (n0, n1),
          s0.tobytes() == g0.tobytes(),
          "first diff at %r" % (np.flatnonzero(s0 != g0)[:4],))
    check("fused == 2 separate (n0=%d n1=%d) B" % (n0, n1),
          s1.tobytes() == g1.tobytes(),
          "first diff at %r" % (np.flatnonzero(s1 != g1)[:4],))
    # parity with a pure-numpy RNE fp16 conversion
    ref0 = x0.astype(np.float16)
    check("matches numpy float16 (A)", s0.tobytes() == ref0.tobytes())


for n0, n1 in ((4096, 4096), (4097, 1023), (1, 262144), (524288, 1703936),
               (262144, 2048)):
    one(n0, n1)
print("RESULT %s" % ("ALL PASS" if not fails else "FAIL: %r" % (fails,)))
raise SystemExit(1 if fails else 0)
