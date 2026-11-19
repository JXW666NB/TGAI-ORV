# -*- coding: utf-8 -*-
"""因果性实验（只在 lab 侧改一个普通属性，不动运行时）：显存池上限 → 步时。

背景（实测）：seq2048 一步里 `cuMemAlloc` 约 2015 次、`cuMemFree` 约 2016 次，
CPU 侧合计约 369 ms（占 0.78 s 步时的 47%）。池的 `_pool_cap` 默认 12 GiB，
而 seq2048 的 cached 正好长期顶在 12 GiB 上 ⇒ 每次归还都触发 trim → cuMemFree。

如果「分配器抖动」真的在关键路径上，那么把 cap 压小（归还更早、抖动更凶）
步时应该**变慢**；这就是因果，而不是相关。

跑法（远端）：
    set CHURN_CAP_GB=4
    D:\\orv_pip_0962\\Scripts\\python.exe D:\\orv_bench\\pool_cap_ab.py --data ... --seq 2048 ...
"""
import json
import os
import sys

import tgai_go.train as T
from tgai_go.train import v3_pretrain as TR

RT = T._cuda()
_cap = os.environ.get("CHURN_CAP_GB", "").strip()
if _cap:
    RT._pool_cap = int(float(_cap) * (1024 ** 3))
    print("[ab] pool_cap = %.2f GiB（默认 12 GiB）" % (RT._pool_cap / float(1 << 30)),
          flush=True)

if __name__ == "__main__":
    TR.main(sys.argv[1:])
