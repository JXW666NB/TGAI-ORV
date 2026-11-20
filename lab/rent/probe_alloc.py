# -*- coding: utf-8 -*-
"""量分配器churn：每步的 pool miss / alloc / free / cuMemGetInfo 次数。

用法： python probe_alloc.py off|gemm
"""
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "off"

sys.argv = [
    "v3_pretrain",
    "--data", "/root/bench/data2048_v96000", "--vocab", "96000",
    "--seq", "512", "--d", "1024", "--layers", "12", "--E", "4",
    "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
    "--opt", "int8gpu", "--amp", MODE, "--batch", "1",
    "--steps", "4", "--log-every", "1", "--no-final-save",
]

import tgai_go.train as T                                    # noqa: E402

rt = T._cuda()

cnt = {"alloc": 0, "free": 0, "getinfo": 0, "trim": 0}

for nm, key in (("_alloc", "alloc"), ("free", "free"),
                ("free_device_bytes", "getinfo"), ("_shed", "trim")):
    fn = getattr(rt, nm, None)
    if fn is None:
        continue

    def make(f, k):
        def wrap(*a, **kw):
            cnt[k] += 1
            return f(*a, **kw)
        return wrap

    setattr(rt, nm, make(fn, key))

st0 = dict(rt.pool_stats())

from tgai_go.train import v3_pretrain as VP                  # noqa: E402

try:
    VP.main()
except SystemExit:
    pass

st1 = dict(rt.pool_stats())
steps = 4
print("\n[alloc-probe] amp=%s  步数=%d" % (MODE, steps))
print("[alloc-probe] 每步统计：")
for k in ("alloc", "free", "getinfo", "trim"):
    if cnt[k]:
        print("   %-8s %7d 次/步" % (k, cnt[k] / steps))
print("[alloc-probe] pool_stats 前后：")
for k in st0:
    if k in ("misses", "hits", "hit_rate", "occupied_bytes",
             "live_bytes", "cached_bytes", "peak_live_bytes"):
        try:
            a, b = st0.get(k), st1.get(k)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                print("   %-16s %s -> %s   (Δ %s)" % (k, a, b, b - a))
        except Exception:                                # noqa: BLE001
            pass
