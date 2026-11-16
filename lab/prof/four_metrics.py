# -*- coding: utf-8 -*-
"""逐步四量对照：墙钟 / CPU 时间 / 分配次数 / 内核派发次数。

1.5B 的波动已确认在 CPU 侧（GPU 只涨 23 ms，墙钟涨 600 ms）。
本脚本逐步打印这四个量，看步时到底跟着谁走 —— 不再靠猜。
"""
import argparse
import sys
import time

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_pretrain as P

T.set_default_device("cuda")
rt = T._cuda()
CLS = type(rt)

CUR = {"alloc": 0, "launch": 0, "on": False}
ROWS = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--d_ff", type=int, default=3328)
    ap.add_argument("--steps", type=int, default=12)
    a = ap.parse_args()

    o_empty = CLS.empty
    o_launch = CLS.launch

    def empty(self, shape, dtype=None, **kw):
        if CUR["on"]:
            CUR["alloc"] += 1
        return o_empty(self, shape, dtype, **kw)

    def launch(self, name, grid, block, args=None, *ar, **kw):
        if CUR["on"]:
            CUR["launch"] += 1
        return o_launch(self, name, grid, block, args, *ar, **kw)

    CLS.empty = empty
    CLS.launch = launch

    from tgai_go.train import optim as _om
    AW = getattr(_om, "AdamW")
    ostep = AW.step
    prev = {"t": None, "cpu": None}

    def step(self, *ar, **kw):
        if CUR["on"]:
            rt.sync()
            now_w = time.perf_counter()
            now_c = time.process_time()
            if prev["t"] is not None:
                ROWS.append((now_w - prev["t"], now_c - prev["cpu"],
                             CUR["alloc"], CUR["launch"]))
            prev["t"] = now_w
            prev["cpu"] = now_c
            CUR["alloc"] = 0
            CUR["launch"] = 0
        return ostep(self, *ar, **kw)

    AW.step = step

    sys.argv = ["v3_pretrain",
                "--data", r"D:\TGAI-V3-1.5B\01_corpus\packed512\s512",
                "--vocab", "96000", "--seq", "512", "--d", "1024",
                "--layers", str(a.layers), "--E", "4", "--n_q", "16",
                "--d_ff", str(a.d_ff), "--top_k", "2", "--csa", "4",
                "--steps", str(a.steps), "--log-every", "100", "--no-final-save"]
    CUR["on"] = True
    prev["t"] = time.perf_counter()
    prev["cpu"] = time.process_time()
    try:
        P.main()
    except SystemExit:
        pass

    print("\n step |  墙钟 s |  CPU s  | CPU占比 | 分配次数 | 内核数")
    for i, (w, c, al, la) in enumerate(ROWS):
        print(" %4d | %7.3f | %7.3f |  %5.1f%% | %8d | %6d"
              % (i, w, c, c / w * 100.0, al, la))
    if ROWS:
        W = np.asarray([r[0] for r in ROWS])
        A = np.asarray([r[2] for r in ROWS])
        L = np.asarray([r[3] for r in ROWS])
        C = np.asarray([r[1] for r in ROWS])
        print("\n相关性 vs 墙钟：分配次数 r=%+.3f   内核数 r=%+.3f   CPU 时间 r=%+.3f"
              % (np.corrcoef(A, W)[0, 1], np.corrcoef(L, W)[0, 1],
                 np.corrcoef(C, W)[0, 1]))


if __name__ == "__main__":
    main()
