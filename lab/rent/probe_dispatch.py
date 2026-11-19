# -*- coding: utf-8 -*-
"""量化 MoE 分发/置换类 kernel 的时间（seq2048）。

做法：包装这些只被 MoE 用到的运行时方法，前后 sync 计时：
  index_gather8 / gather_rows / scatter_add_rows / index_scatter_add8 /
  cat_place / cat_extract / permute8_v4 / transpose2d
逐次 sync 会引入固定开销，所以把结果当作**上界**看，并同时给出调用次数。
"""
import atexit
import json
import sys
import time

import tgai_go.train as T
from tgai_go.train import v3_pretrain as TR

rt = T._cuda()
NAMES = ["index_gather8", "gather_rows", "scatter_add_rows", "index_scatter_add8",
         "cat_place", "cat_extract", "permute8_v4", "transpose2d", "situ_glu_fwd",
         "situ_glu_bwd", "mul_bcast8", "add_bcast8", "ewise_add", "ewise_scale_v4"]
acc = {}

for nm in NAMES:
    fn = getattr(rt, nm, None)
    if fn is None:
        continue
    acc[nm] = {"n": 0, "s": 0.0}

    def make(f, key):
        def wrapper(*a, **kw):
            rt._sync_raw() if hasattr(rt, "_sync_raw") else None
            t0 = time.perf_counter()
            out = f(*a, **kw)
            try:
                rt.sync()
            except Exception:
                pass
            acc[key]["s"] += time.perf_counter() - t0
            acc[key]["n"] += 1
            return out
        return wrapper

    setattr(rt, nm, make(fn, nm))


def _report():
    out = {k: {"n": v["n"], "ms": round(v["s"] * 1000.0, 1)} for k, v in acc.items()}
    print("[dispatch] " + json.dumps(out, ensure_ascii=False), flush=True)


atexit.register(_report)

if __name__ == "__main__":
    TR.main(sys.argv[1:])
