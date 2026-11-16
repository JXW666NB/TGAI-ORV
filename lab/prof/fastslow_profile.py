# -*- coding: utf-8 -*-
"""对「快步」与「慢步」分别 cProfile，逐函数对比，找出变慢的具体函数。

已知：1.5B 在 step 4 出现干净台阶（fwd+bwd 的 CPU 时间 0.49 s -> 0.84 s）；
已否证热降频、GC、分配次数、内核数、MoE 分位路由。
本脚本直接给出「慢步里哪几个函数变慢」。
"""
import argparse
import cProfile
import io
import pstats
import sys

import tgai_go.train as T
from tgai_go.train import v3_pretrain as P

T.set_default_device("cuda")
rt = T._cuda()

PROF = {"pr": None, "idx": 0, "buckets": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--d_ff", type=int, default=3328)
    ap.add_argument("--steps", type=int, default=10)
    a = ap.parse_args()

    from tgai_go.train import optim as _om
    AW = getattr(_om, "AdamW")
    ostep = AW.step

    def step(self, *ar, **kw):
        idx = PROF["idx"]
        PROF["idx"] += 1
        pr = PROF["pr"]
        if pr is not None:
            pr.disable()
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
            bucket = "fast" if idx <= 4 else "slow"
            PROF["buckets"].setdefault(bucket, []).append(s)
        if idx >= 1:                      # 跳过第 0 步（JIT/池增长）
            PROF["pr"] = cProfile.Profile()
            PROF["pr"].enable()
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

    # 汇总两个桶里各函数的累计 tottime
    def agg(bucket):
        tot = {}
        stats = PROF["buckets"].get(bucket) or []
        for s in stats:
            st = pstats.Stats(*(s.getvalue(),)) if False else None
        return None

    # 简化：直接打印两个桶里第一份 profile 的 top 12
    for bucket in ("fast", "slow"):
        ss = PROF["buckets"].get(bucket) or []
        print("\n=== %s（%d 个 step）===" % (bucket, len(ss)))
        if not ss:
            continue
        txt = ss[-1].getvalue().splitlines()
        started = False
        shown = 0
        for line in txt:
            if line.strip().startswith("ncalls"):
                started = True
                print("  " + line.strip()[:100])
                continue
            if started and line.strip():
                print("  " + line.strip()[:110])
                shown += 1
                if shown >= 12:
                    break


if __name__ == "__main__":
    main()
