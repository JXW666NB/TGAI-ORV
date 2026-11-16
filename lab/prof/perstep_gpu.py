# -*- coding: utf-8 -*-
"""按步统计 GPU 时间：找出「快步」与「慢步」到底差在哪个 op。

之前把 1.5B 的波动归因为「MoE 负载不均放大瘦长 GEMM 低效」——
微基准（skew_cost.py）否证了它：成本对 n 是线性的（截距 0.017 ms + 0.000617 ms/token），
同样总 token 数下均衡与不均总耗时相同。所以必须重新定位。

本脚本对每一步用 CUDA event 统计各内核 GPU 耗时，逐步打印 top 项，直接对比快/慢步。
"""
import argparse
import collections
import ctypes
import sys

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_pretrain as P

T.set_default_device("cuda")
rt = T._cuda()
cu = rt.cu
STREAM = rt._stream

CUR = {"on": False, "agg": None}
PAIRS = []
POOL = []


def _ev():
    if POOL:
        return POOL.pop()
    e = ctypes.c_void_p()
    cu.cuEventCreate(ctypes.byref(e), 0)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--d_ff", type=int, default=3328)
    ap.add_argument("--steps", type=int, default=10)
    a = ap.parse_args()

    orig_launch = rt.launch

    def launch(name, grid, block, args=None, *ar, **kw):
        if not CUR["on"]:
            return orig_launch(name, grid, block, args, *ar, **kw)
        e0, e1 = _ev(), _ev()
        cu.cuEventRecord(e0, STREAM)
        r = orig_launch(name, grid, block, args, *ar, **kw)
        cu.cuEventRecord(e1, STREAM)
        PAIRS.append((str(name), e0, e1))
        return r

    rt.launch = launch

    # 每步边界：AdamW.step
    from tgai_go.train import optim as _om
    AW = getattr(_om, "AdamW")
    ostep = AW.step
    step_times = []

    def step(self, *ar, **kw):
        if CUR["on"]:
            rt.sync()
            agg = CUR["agg"]
            for nm, e0, e1 in PAIRS:
                ms = ctypes.c_float(0.0)
                if cu.cuEventElapsedTime(ctypes.byref(ms), e0, e1) == 0:
                    agg[nm] += float(ms.value)
                POOL.append(e0)
                POOL.append(e1)
            PAIRS.clear()
            tot = sum(agg.values())
            top = agg.most_common(4)
            idx = len(step_times)
            print("STEP %2d gpu=%.1f ms | %s" %
                  (idx, tot, "  ".join("%s:%.2f" % (k, v) for k, v in top)), flush=True)
            step_times.append(tot)
            CUR["agg"] = collections.Counter()
        return ostep(self, *ar, **kw)

    AW.step = step

    # 第二步起开始计时（第一步含 JIT/池增长）
    orig_fwd = P.forward_loss

    def counted(*ar, **kw):
        return orig_fwd(*ar, **kw)

    sys.argv = ["v3_pretrain",
                "--data", r"D:\TGAI-V3-1.5B\01_corpus\packed512\s512",
                "--vocab", "96000", "--seq", "512", "--d", "1024",
                "--layers", str(a.layers), "--E", "4", "--n_q", "16",
                "--d_ff", str(a.d_ff), "--top_k", "2", "--csa", "4",
                "--steps", str(a.steps), "--log-every", "100", "--no-final-save"]
    CUR["on"] = True
    CUR["agg"] = collections.Counter()
    try:
        P.main()
    except SystemExit:
        pass

    print("\nGPU 时间（逐步）:", " ".join("%.1f" % t for t in step_times))


if __name__ == "__main__":
    main()
