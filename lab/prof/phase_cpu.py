# -*- coding: utf-8 -*-
"""按阶段拆 CPU 时间（process_time），定位 1.5B 波动的阶段来源。

已知：CPU 占用 ~100%（墙钟≈CPU）、分配数/内核数逐 step 恒定、热降频已否证。
所以波动必然落在某个阶段的 Python 工作量上。这里把每步拆成
「fwd+bwd」与「opt」两段，用 process_time 计（不受 GPU 等待影响）。
"""
import argparse
import sys
import time

import tgai_go.train as T
from tgai_go.train import v3_pretrain as P

T.set_default_device("cuda")
rt = T._cuda()

MARK = {"fwd": None, "opt": None}
ROWS = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--d_ff", type=int, default=3328)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--nogc", action="store_true", help="关闭 Python GC，验证 GC 是否为波动来源")
    a = ap.parse_args()

    if a.nogc:
        import gc
        gc.disable()
        print("[gc] 已关闭")
    o_fwd = P.forward_loss

    def forward_loss(*ar, **kw):
        now = time.process_time()
        if MARK["opt"] is not None:
            ROWS.append((None, now - MARK["opt"], None))
        MARK["fwd"] = now
        return o_fwd(*ar, **kw)

    P.forward_loss = forward_loss

    from tgai_go.train import optim as _om
    AW = getattr(_om, "AdamW")
    ostep = AW.step

    def step(self, *ar, **kw):
        now = time.process_time()
        if MARK["fwd"] is not None and ROWS and ROWS[-1][0] is None:
            fb = now - MARK["fwd"]
            ROWS[-1] = (fb, ROWS[-1][1], None)
        MARK["opt"] = now
        r = ostep(self, *ar, **kw)
        return r

    AW.step = step

    sys.argv = ["v3_pretrain",
                "--data", r"D:\TGAI-V3-1.5B\01_corpus\packed512\s512",
                "--vocab", "96000", "--seq", "512", "--d", "1024",
                "--layers", str(a.layers), "--E", "4", "--n_q", "16",
                "--d_ff", str(a.d_ff), "--top_k", "2", "--csa", "4",
                "--steps", str(a.steps), "--log-every", "100", "--no-final-save"]
    try:
        P.main()
    except SystemExit:
        pass

    print("\n step | fwd+bwd CPU s | opt CPU s | 合计")
    for i, (fb, op, _) in enumerate(ROWS):
        if fb is None:
            continue
        print(" %4d | %13.3f | %9.3f | %6.3f" % (i, fb, op, fb + op))


if __name__ == "__main__":
    main()
