# -*- coding: utf-8 -*-
"""按固定口径跑一次 event 归因（参数：seq skip outfile [tag] [leaves]）。

远端用法：
    python D:\\orv_bench\\run_prof_one.py 2048 3 D:\\orv_bench\\prof_2048_new.json seq2048
    python D:\\orv_bench\\run_prof_one.py 2048 999999 D:\\orv_bench\\prof_2048_vanilla.json vanilla
    python D:\\orv_bench\\run_prof_one.py 2048 3 D:\\orv_bench\\prof_2048_phases.json phases 0

`skip=999999` 时 profiler 永不进入计时窗口（不包 launch、不发 event），
于是同进程的墙钟就等于**无打点基线** —— 用它减去带打点的墙钟，即为打点开销。
`leaves=0` 时只打四段 event（每步 8 个），不逐 kernel 打点 ⇒ phase 的
inclusive 时间是干净的（不含探针 CPU 开销）。
"""
import os
import sys

WORK = r"D:\orv_bench"
os.chdir(WORK)
sys.path.insert(0, WORK)

seq = int(sys.argv[1])
skip = int(sys.argv[2])
out = sys.argv[3]
tag = sys.argv[4] if len(sys.argv) > 4 else ("seq%d" % seq)
leaves = sys.argv[5] if len(sys.argv) > 5 else "1"

os.environ["PROF_SKIP"] = str(skip)
os.environ["PROF_OUT"] = out
os.environ["PROF_TAG"] = tag
os.environ["PROF_LEAVES"] = leaves

import prof_event_kernels as P                                    # noqa: E402

argv = ["--data", os.path.join(WORK, "data2048_v96000"), "--vocab", "96000",
        "--seq", str(seq), "--d", "1024", "--layers", "12", "--E", "4",
        "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
        "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
        "--steps", "20", "--log-every", "99", "--no-final-save"]

P.TR.main(argv)
