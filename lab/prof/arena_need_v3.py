# -*- coding: utf-8 -*-
"""量一件事：捕获一整步需要多大 arena。

`_arena_alloc` 是 bump-only（捕获期内释放的块不复用），
所以 arena 至少要装下**一整步所有分配的字节总和**。
这个数远大于 `peak_live`，是 CUDA Graph 能否整步捕获的决定性约束。

做法：给 `_alloc` 挂钩累计请求字节，跑 N 步真训练，除以步数。
"""
import argparse
import sys

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_pretrain as P

T.set_default_device("cuda")
rt = T._cuda()
CLS = type(rt)

TOT = [0]
CNT = [0]
TOT_ALL = [0]
CNT_ALL = [0]

_orig_empty = CLS.empty


def empty_hook(self, shape, dtype=None):
    """捕获期内 `_pool_frozen()` 为真 ⇒ **所有** empty 都走 arena。
    所以真实的 arena 需求是"全部请求字节之和"，不是只有池未命中的那部分。"""
    try:
        n = 1
        for d in shape:
            n *= int(d)
        TOT_ALL[0] += n * 4
        CNT_ALL[0] += 1
    except Exception:
        pass
    return _orig_empty(self, shape, dtype)


CLS.empty = empty_hook
_orig = CLS._alloc


def alloc_hook(self, nbytes):
    TOT[0] += int(nbytes)
    CNT[0] += 1
    return _orig(self, nbytes)


CLS._alloc = alloc_hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--d_ff", type=int, default=2816)
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()

    data = (r"D:\TGAI-V3-1.5B\01_corpus\packed512\s512" if a.seq <= 512
            else r"D:\TGAI-V3-1.5B\01_corpus\packed_1024")
    sys.argv = ["v3_pretrain", "--data", data, "--vocab", "96000",
                "--seq", str(a.seq), "--d", "1024", "--layers", str(a.layers),
                "--E", "4", "--n_q", "16", "--d_ff", str(a.d_ff),
                "--top_k", "2", "--csa", "4", "--steps", str(a.steps),
                "--log-every", "100", "--no-final-save"]
    try:
        P.main()
    except SystemExit:
        pass

    ps = rt.pool_stats()
    gib = float(1 << 30)
    per_step = TOT[0] / max(a.steps, 1)
    per_cnt = CNT[0] / max(a.steps, 1)
    print("\n===== 捕获 arena 需求 ======")
    print("  层数=%d seq=%d d_ff=%d  步数=%d" % (a.layers, a.seq, a.d_ff, a.steps))
    print("  每步分配次数      : %10.0f" % per_cnt)
    print("  每步分配字节总和  : %10.2f GiB   <== arena 至少要这么大" % (per_step / gib))
    print("  每步峰值活占用    : %10.2f GiB   （对比：图能省的是 CPU，不是这个）"
          % (ps["peak_live_bytes"] / gib))
    print("  倍数              : %10.1fx" % (per_step / max(ps["peak_live_bytes"], 1)))
    print("  设备总量          : %10.2f GiB" % (rt._device_total_bytes() / gib))
    print("  ---- 全部分配请求（捕获期真实 arena 需求）----")
    print("  每步请求次数      : %10.0f" % (CNT_ALL[0] / max(a.steps, 1)))
    print("  每步请求字节总和  : %10.2f GiB  （bump-only arena 至少要这么大）"
          % (TOT_ALL[0] / max(a.steps, 1) / gib))
    print("  相对峰值活占用    : %10.1fx" % (TOT_ALL[0] / max(ps["peak_live_bytes"], 1)))


if __name__ == "__main__":
    main()
