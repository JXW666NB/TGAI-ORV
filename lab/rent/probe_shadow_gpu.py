# -*- coding: utf-8 -*-
"""GPU-only behavioural checks for the fp16 shadow cache (standalone, no pytest).

Mirrors tests/test_fp16_shadow_cache.py::test_shadow_is_recast_after_clear and
::test_drop_ptr_invalidates_recycled_shadow so they can also be run on a remote
GPU box that has no pytest installed.
"""
import ctypes
import sys

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
fails = []


def check(name, cond, msg=""):
    print("  %-46s %s %s" % (name, "PASS" if cond else "FAIL", "" if cond else msg))
    if not cond:
        fails.append(name)


n = 4096
a = rt.h2d(np.full((1, n), 1.0, np.float32))
rt.f16_cache_pin([a.ptr])
rt.f16_cache_enable(True)
try:
    b1 = rt._cast_f16_cached(a.ptr, n)
    check("hit returns same shadow", rt._cast_f16_cached(a.ptr, n) is b1)
    got1 = np.asarray(rt.d2h(rt.cast_f16_to_f32(b1)), np.float64).reshape(-1)
    check("first shadow == 1.0", bool(np.all(got1 == 1.0)), repr(got1[:4]))

    host = np.full(n, 2.0, np.float32)
    rt.h2d_into(a, ctypes.c_void_p(host.ctypes.data), n * 4)
    rt.f16_cache_clear()
    b2 = rt._cast_f16_cached(a.ptr, n)
    got2 = np.asarray(rt.d2h(rt.cast_f16_to_f32(b2)), np.float64).reshape(-1)
    check("shadow re-cast after clear (==2.0)", bool(np.all(got2 == 2.0)), repr(got2[:4]))

    rt.f16_cache_clear()
    rt._f16_cache_drop_ptr(a.ptr)
    b3 = rt._cast_f16_cached(a.ptr, n)
    check("drop_ptr invalidates shadow", b3 is not b2)

    # 连跑两次 cast 必须逐位相同（cast 本身是确定性的，没有 atomicAdd）
    b4 = rt._cast_f16_cached(a.ptr, n)
    v1 = rt.d2h(b3).copy()
    rt.f16_cache_clear()
    rt._f16_cache_drop_ptr(a.ptr)
    b5 = rt._cast_f16_cached(a.ptr, n)
    v2 = rt.d2h(b5).copy()
    check("repeated casts are bit-identical", bool(v1.tobytes() == v2.tobytes()))
finally:
    rt.f16_cache_pin([])
    rt.f16_cache_enable(False)

print("RESULT %s" % ("ALL PASS" if not fails else "FAIL: %r" % (fails,)))
sys.exit(1 if fails else 0)
