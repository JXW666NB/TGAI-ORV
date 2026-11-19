# -*- coding: utf-8 -*-
"""在真实训练进程里打印池上限与分配器统计（跑完再报）。

用法：python probe_cap.py --data ... --seq 2048 ...（参数原样传给训练器）
"""
import atexit
import json
import sys

import tgai_go.train as T
from tgai_go.train import v3_pretrain as TR

rt = T._cuda()


def _report():
    info = {}
    for k in ("_pool_cap", "_pool_cached_bytes", "_pool_misses", "_pool_hits",
              "_pool_cap_adapted", "_dev_total_bytes"):
        v = getattr(rt, k, None)
        if isinstance(v, int) and k in ("_pool_cap", "_pool_cached_bytes",
                                        "_dev_total_bytes"):
            info[k] = round(v / (1024 ** 3), 2)
        else:
            info[k] = v
    print("[pool] " + json.dumps(info), flush=True)


atexit.register(_report)

if __name__ == "__main__":
    TR.main(sys.argv[1:])
