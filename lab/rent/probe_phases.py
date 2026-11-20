# -*- coding: utf-8 -*-
"""按步拆解：发射次数/时间、alloc 次数/时间、free 次数/时间、wall。
冷启动（构建期）单独一行，不污染稳态。"""
import builtins, sys, time
MODE = sys.argv[1] if len(sys.argv) > 1 else "off"
sys.argv = ["v3_pretrain", "--data", "/root/bench/data2048_v96000", "--vocab", "96000",
            "--seq", "512", "--d", "1024", "--layers", "12", "--E", "4",
            "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
            "--opt", "int8gpu", "--amp", MODE, "--batch", "1",
            "--steps", "6", "--log-every", "1", "--no-final-save"]
import tgai_go.train as T
rt = T._cuda()
acc = {"ln": 0, "ls": 0.0, "an": 0, "as": 0.0, "fn": 0, "fs": 0.0}
fl, fa, ff = rt.launch, rt._alloc, rt.free
def launch(*a, **k):
    t0 = time.perf_counter()
    try: return fl(*a, **k)
    finally: acc["ls"] += time.perf_counter()-t0; acc["ln"] += 1
def alloc(*a, **k):
    t0 = time.perf_counter()
    try: return fa(*a, **k)
    finally: acc["as"] += time.perf_counter()-t0; acc["an"] += 1
def free(*a, **k):
    t0 = time.perf_counter()
    try: return ff(*a, **k)
    finally: acc["fs"] += time.perf_counter()-t0; acc["fn"] += 1
rt.launch, rt._alloc, rt.free = launch, alloc, free
rows, prev, tprev = [], dict(acc), time.perf_counter()
_orig = builtins.print
def spy(*a, **k):
    global prev, tprev
    _orig(*a, **k)
    s = a[0] if a else ""
    if isinstance(s, str) and s.startswith("step"):
        now, tw = dict(acc), time.perf_counter()
        d = {key: now[key]-prev[key] for key in acc}
        rows.append((s.split()[1], d, tw-tprev))
        prev, tprev = now, tw
builtins.print = spy
from tgai_go.train import v3_pretrain as VP
try: VP.main()
except SystemExit: pass
builtins.print = _orig
print("\n[phases] amp=%s  （每步：发射/alloc/free 的次数与累计耗时，单位 ms）" % MODE)
print("   step      wall   launch(n/ms)      alloc(n/ms)       free(n/ms)   launch均us")
for tag, d, w in rows:
    ln, ls = d["ln"], d["ls"]*1e3
    an, as_ = d["an"], d["as"]*1e3
    fn, fs = d["fn"], d["fs"]*1e3
    per = (ls/ln*1e3) if ln else 0
    print("   %-8s %6.1f   %6d/%7.1f   %6d/%7.1f   %6d/%6.1f   %8.2f"
          % (tag, w*1e3, ln, ls, an, as_, fn, fs, per))