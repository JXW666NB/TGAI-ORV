# -*- coding: utf-8 -*-
"""对拍 Polaris 补丁：`_polar_express_np` 的 Python 逐元素平方和 → np.dot。

判据：
  1) 数值：与原始实现在同一输入上 max|Δ| <= 1e-6（NS 会把 1 ulp 放大，故给一点余量）
  2) 速度：至少 10× 以上（1.05M 参数模型的 CPU opt.step 从 620~813 ms 降下来）
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


orig = load("polaris_orig", os.path.join(HERE, "polaris_orig.py"))
patched = load("polaris_patched", os.path.join(HERE, "polaris_patched.py"))

rng = np.random.default_rng(0)
rows = []
for shape in [(256, 1024), (512, 512), (1024, 1024), (64, 4096)]:
    x = rng.normal(size=shape).astype(np.float32)

    t0 = time.perf_counter()
    a = orig._polar_express_np(x, steps=5)
    t_orig = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    b = patched._polar_express_np(x, steps=5)
    t_patch = (time.perf_counter() - t0) * 1e3

    d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
    rel = d / max(1e-30, float(np.abs(a.astype(np.float64)).max()))
    rows.append({"shape": list(shape), "orig_ms": round(t_orig, 2),
                 "patched_ms": round(t_patch, 2),
                 "speedup": round(t_orig / max(1e-9, t_patch), 1),
                 "max_abs_diff": d, "rel": rel})
    print("%-12s orig %8.2f ms -> patched %7.2f ms  (%.1fx)  max|Δ|=%.3e rel=%.2e"
          % (str(shape), t_orig, t_patch, t_orig / max(1e-9, t_patch), d, rel))

print()
print("[polaris_patch] " + json.dumps(rows, ensure_ascii=False))
