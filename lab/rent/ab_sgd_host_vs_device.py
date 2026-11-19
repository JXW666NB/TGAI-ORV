# -*- coding: utf-8 -*-
"""SGD 修前/修后对照：同一份 9.4M 参数，强制走 numpy 路径 vs 设备路径。"""
import json
import time

import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
rng = np.random.default_rng(0)
sizes = [256 * 1024] * 30 + [1024 * 512] + [128 * 1024] * 8
tot = sum(sizes)

res = {}
for tag in ("host", "device"):
    big = []
    for n in sizes:
        p = T.tensor(rng.normal(0, 0.01, n).astype(np.float32), requires_grad=True)
        p.grad = T.tensor(rng.normal(0, 1.0, n).astype(np.float32))
        big.append(p)
    o = T.optim.SGD(big, lr=1e-3, momentum=0.9)
    o._dev = (tag == "device")          # 强制切换实现
    o.step()                            # 先走一步把状态热起来
    rt.sync()
    ts = []
    for _ in range(2 if tag == "host" else 3):
        rt.sync(); t0 = time.perf_counter(); o.step(); rt.sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    res[tag] = round(ts[0], 2)
    print("%-7s %.2f ms" % (tag, ts[0]), flush=True)

print()
print("[sgd_ab] " + json.dumps({"params": tot, **res, "speedup": round(res["host"] / max(1e-9, res["device"]), 1)}))
