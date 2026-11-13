# -*- coding: utf-8 -*-
"""最小判别：分配 1 GB → 删除 → empty_cache，ORV 的 live 是否回到基线。

若归不回去，则此前"no_grad 不省内存"的结论是分配器假象，不是自动微分的问题。
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402


def mib(b):
    return b / 2**20


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    rt.empty_cache()
    base = rt.pool_stats()
    print("基线    live=%.0f peak_live=%.0f peak=%.0f"
          % (mib(base["live_bytes"]), mib(base["peak_live_bytes"]),
             mib(base["peak_bytes"])))

    # ① 一个大张量
    big = T.tensor(np.zeros((256 * 1024 * 1024,), np.float32))   # 1 GiB
    s1 = rt.pool_stats()
    print("分配后  live=%.0f（应 +1024）" % mib(s1["live_bytes"]))

    del big
    s2 = rt.pool_stats()
    print("删除后  live=%.0f（未回收属正常，看下一行）" % mib(s2["live_bytes"]))

    rt.empty_cache()
    s3 = rt.pool_stats()
    print("回收后  live=%.0f（应回到基线附近）" % mib(s3["live_bytes"]))

    # ② 多个中等张量，模拟"每层中间量"
    keep = []
    for i in range(5):
        keep.append(T.tensor(np.zeros((32 * 1024 * 1024,), np.float32)))  # 128 MiB
    s4 = rt.pool_stats()
    print("5×128MiB 后 live=%.0f（应 +640）" % mib(s4["live_bytes"]))
    keep.clear()
    rt.empty_cache()
    s5 = rt.pool_stats()
    print("清空回收后 live=%.0f" % mib(s5["live_bytes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
