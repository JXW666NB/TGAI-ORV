# -*- coding: utf-8 -*-
"""量 _alloc / free 的累计耗时占比。"""
import sys, time
MODE = sys.argv[1] if len(sys.argv) > 1 else "off"
sys.argv = ["v3_pretrain", "--data", "/root/bench/data2048_v96000", "--vocab", "96000",
            "--seq", "512", "--d", "1024", "--layers", "12", "--E", "4",
            "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
            "--opt", "int8gpu", "--amp", MODE, "--batch", "1",
            "--steps", "4", "--log-every", "1", "--no-final-save"]
import tgai_go.train as T
rt = T._cuda()
acc = {"alloc_n": 0, "alloc_s": 0.0, "free_n": 0, "free_s": 0.0}
fa, ff = rt._alloc, rt.free
def alloc(*a, **k):
    t0 = time.perf_counter()
    try: return fa(*a, **k)
    finally:
        acc["alloc_s"] += time.perf_counter() - t0
        acc["alloc_n"] += 1
def free(*a, **k):
    t0 = time.perf_counter()
    try: return ff(*a, **k)
    finally:
        acc["free_s"] += time.perf_counter() - t0
        acc["free_n"] += 1
rt._alloc, rt.free = alloc, free
from tgai_go.train import v3_pretrain as VP
t0 = time.perf_counter()
try: VP.main()
except SystemExit: pass
wall = time.perf_counter() - t0
steps = 4
print("\n[time-probe] amp=%s" % MODE)
print("  _alloc : %7.1f 次/步, 累计 %8.1f ms/步 (%.2f us/次)" % (acc["alloc_n"]/steps, acc["alloc_s"]/steps*1e3, acc["alloc_s"]/max(1,acc["alloc_n"])*1e6))
print("  free   : %7.1f 次/步, 累计 %8.1f ms/步 (%.2f us/次)" % (acc["free_n"]/steps, acc["free_s"]/steps*1e3, acc["free_s"]/max(1,acc["free_n"])*1e6))
print("  两者合计 %.1f ms/步" % ((acc["alloc_s"]+acc["free_s"])/steps*1e3))