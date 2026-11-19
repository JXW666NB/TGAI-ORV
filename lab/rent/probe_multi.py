# -*- coding: utf-8 -*-
"""最小用例隔离多张量内核：先 2 个张量，再 321 个。"""
import json

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
out = {}

a = np.array([3.0, 4.0], np.float32)          # 平方和 = 25
b = np.array([1.0, 2.0, 2.0], np.float32)     # 平方和 = 9
ga, gb = rt.h2d(a), rt.h2d(b)

try:
    ptrs, sizes = rt._ptrs_and_sizes([ga, gb])
    out["h2d_ptrs_ok"] = True
    out["ptrs_n"] = int(ptrs.n)
    out["sizes_n"] = int(sizes.n)
    out["ptrs_dtype"] = str(ptrs.dtype)
except Exception as e:                        # noqa: BLE001
    out["h2d_ptrs_ok"] = "FAIL: %r" % (e,)

try:
    out["sumsq_multi"] = float(rt.sumsq_multi_total([ga, gb]))
    out["expect"] = 34.0
except Exception as e:                        # noqa: BLE001
    out["sumsq_multi"] = "FAIL: %r" % (e,)

try:
    rt.scale_multi([ga, gb], 2.0)
    rt.sync()
    out["scale_a"] = np.asarray(rt.d2h(ga), np.float32).tolist()
    out["scale_b"] = np.asarray(rt.d2h(gb), np.float32).tolist()
except Exception as e:                        # noqa: BLE001
    out["scale_multi"] = "FAIL: %r" % (e,)

print("[probe] " + json.dumps(out))
