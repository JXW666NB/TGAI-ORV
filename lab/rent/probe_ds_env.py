# -*- coding: utf-8 -*-
"""环境/API 事实核对（为 dS fp16 缩放实验做准备）。

只读：打印版本、layer_kinds、相关内核名、_bmm_op 是否已有 batch==1 fp16 路由、
以及 absmax/scale/cast 相关 API 是否存在。
跑法（远端）：D:\\orv_pip_0962\\Scripts\\python.exe -X utf8 D:\\orv_bench\\probe_ds_env.py
"""
import inspect
import os
import sys

WORK = r"D:\orv_bench"
os.chdir(WORK)
sys.path.insert(0, WORK)

import numpy as np                                              # noqa: E402

import tgai_go                                                  # noqa: E402
import tgai_go.train as T                                       # noqa: E402
from tgai_go.train import cuda as _cuda_mod_unused              # noqa: F401
from tgai_go.train import v3_pretrain as TR                     # noqa: E402
from tgai_go.train import v3_train_cli as CLI                   # noqa: E402
from tgai_go.train import v3_csa as C                           # noqa: E402

print("== version ==")
for expr in ("tgai_go.__version__",):
    try:
        print(expr, "=", eval(expr))
    except Exception as e:                                      # noqa: BLE001
        print(expr, "ERR", repr(e))
try:
    import importlib.metadata as md
    print("pip tgai-orv =", md.version("tgai-orv"))
except Exception as e:                                          # noqa: BLE001
    print("pip version ERR", repr(e))

print("\n== layer_kinds (csa=4, layers=12) ==")
cfg = {"layers": 12, "csa": 4}
try:
    ks = CLI.layer_kinds(cfg)
    print("kinds =", ks)
    print("csa idx =", [i for i, k in enumerate(ks) if k == "csa"])
    print("global idx =", [i for i, k in enumerate(ks) if k != "csa"])
except Exception as e:                                          # noqa: BLE001
    print("layer_kinds ERR", repr(e))

print("\n== _bmm_op has batch==1 fp16 route? ==")
src = inspect.getsource(T._cuda().__class__)
i = src.find("def _bmm_op")
print(src[i:i + 1400])

print("\n== relevant runtime methods ==")
rt = T._cuda()
names = sorted(n for n in dir(rt) if any(
    s in n.lower() for s in ("absmax", "abs_max", "amax", "scale", "cast", "max",
                             "reduce", "sumsq", "quant")))
print(names)

print("\n== T.help(s=...) ==")
for q in ("absmax", "amax", "cast", "scale", "softmax_bwd", "attn_mask"):
    try:
        print("---- help(%s) ----" % q)
        T.help(s=q)
    except Exception as e:                                      # noqa: BLE001
        print("help ERR", repr(e))

print("\n== fp16 state api ==")
for n in ("_fp16_state", "_fp16_fail", "f16_cache_stats", "cast_f32_to_f16"):
    print(n, hasattr(rt, n))

print("\n== csa attn chunk default ==", C._ATTN_CQ)
print("== sgemm_strided_batched sig ==")
print(inspect.signature(rt.sgemm_strided_batched))
print(inspect.signature(rt.sgemm_nt_strided_batched))
print(inspect.signature(rt.sgemm_tn_strided_batched))
