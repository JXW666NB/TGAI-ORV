# -*- coding: utf-8 -*-
"""验证 reset_peak() 与 resolve_optimizer 的内置解析。"""
import json

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()

out = {}
out["peak_before"] = int(rt.pool_stats()["peak_live_bytes"])
rt.reset_peak()
out["peak_after_reset"] = int(rt.pool_stats()["peak_live_bytes"])

x = rt.empty((1 << 20,))          # 4 MiB
rt.sync()
out["peak_after_alloc"] = int(rt.pool_stats()["peak_live_bytes"])
rt.free(int(x.ptr))
rt.sync()
rt.reset_peak()
out["peak_after_second_reset"] = int(rt.pool_stats()["peak_live_bytes"])
out["hits_after_reset"] = int(rt.pool_stats()["hits"])

out["resolved"] = {n: getattr(T.plugin.resolve_optimizer(n), "__name__", None)
                   for n in ("AdamW", "SGD", "Adafactor", "LAMB", "Lion", "Muon",
                             "Polaris", "Accumulator")}
print("[verify] " + json.dumps(out, ensure_ascii=False))
