# -*- coding: utf-8 -*-
"""查 data2048_v96000 的分片里有没有 seg，值是什么；以及 CSA plan 是否退化。"""
import json
import os
import sys

WORK = r"D:\orv_bench"
os.chdir(WORK)
sys.path.insert(0, WORK)

import numpy as np                                              # noqa: E402

print("== shards ==")
sh = os.path.join(WORK, "data2048_v96000")
for root, _dirs, files in os.walk(sh):
    for f in sorted(files)[:3]:
        p = os.path.join(root, f)
        print("--", p, os.path.getsize(p))
        try:
            z = np.load(p, allow_pickle=True)
            print("   keys =", list(z.keys()))
            for k in z.keys():
                a = z[k]
                try:
                    print("   %-8s shape=%r dtype=%s min=%s max=%s nuniq=%d"
                          % (k, a.shape, a.dtype, a.min() if a.size else None,
                             a.max() if a.size else None,
                             len(np.unique(a)) if a.size and a.size < 5e6 else -1))
                except Exception as e:                           # noqa: BLE001
                    print("   %-8s %r" % (k, e))
        except Exception as e:                                   # noqa: BLE001
            print("   ERR", repr(e))
    break

print("\n== data.py: how seg is produced ==")
import tgai_go.train.data as D                                  # noqa: E402
import inspect
src = inspect.getsource(D)
for i, line in enumerate(src.splitlines(), 1):
    if "seg" in line:
        print("%4d %s" % (i, line))
