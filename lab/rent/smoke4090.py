# -*- coding: utf-8 -*-
"""4090 上 ORV 的 GPU 冒烟：能不能起 runtime、能不能算。"""
import os
import sys

import numpy as np

# 让 ctypes 能 dlopen torch 自带的 cuBLAS / NVRTC / cudart
import glob

torch_libs = []
try:
    import torch
    base = os.path.dirname(torch.__file__)
    for pat in ("nvidia/*/lib", "nvidia/*/bin"):
        torch_libs += glob.glob(os.path.join(base, pat))
except Exception:                                        # noqa: BLE001
    pass
if torch_libs:
    os.environ["LD_LIBRARY_PATH"] = ":".join(torch_libs + [os.environ.get("LD_LIBRARY_PATH", "")])
    print("lib dirs:", len(torch_libs))

import tgai_go.train as T

rt = None
for nm in ("cuda", "_cuda"):
    f = getattr(T, nm, None)
    if f is None:
        continue
    try:
        rt = f()
        print("runtime via", nm, "->", type(rt).__name__)
        break
    except Exception as e:                               # noqa: BLE001
        print(nm, "失败:", type(e).__name__, str(e)[:200])

if rt is None:
    print("!! 拿不到 CUDA runtime")
    sys.exit(1)

a = T.tensor(np.random.rand(64, 64).astype(np.float32))
b = T.tensor(np.random.rand(64, 64).astype(np.float32))
c = a @ b
print("matmul ok")

try:
    st = rt.pool_stats()
    print("pool:", {k: st[k] for k in list(st)[:4]})
except Exception as e:                                   # noqa: BLE001
    print("pool_stats 不可用:", type(e).__name__)

print("dev name:", getattr(rt, "device_name", lambda: "?")())
