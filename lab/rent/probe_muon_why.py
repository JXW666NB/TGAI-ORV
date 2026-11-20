# -*- coding: utf-8 -*-
"""Muon 为什么慢：按步统计发射次数/主机耗时/内核种类。

用法： python probe_muon_why.py [muon|adamw] [fast|parity]
"""
import builtins
import collections
import os
import sys
import time

OPT = sys.argv[1] if len(sys.argv) > 1 else "muon"
MODE = sys.argv[2] if len(sys.argv) > 2 else "fast"

if MODE == "parity":
    os.environ["TG_MUON_NS_PARITY"] = "1"

sys.argv = ["v3_pretrain", "--data", "/root/bench/data2048_v96000", "--vocab", "96000",
            "--seq", "512", "--d", "1024", "--layers", "12", "--E", "4",
            "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
            "--opt", OPT, "--amp", "off", "--batch", "1",
            "--steps", "4", "--log-every", "1", "--no-final-save"]

import tgai_go.train as T                                    # noqa: E402

rt = T._cuda()
acc = {"n": 0, "s": 0.0}
names = collections.Counter()
fl = rt.launch


def launch(name, *a, **k):
    t0 = time.perf_counter()
    try:
        return fl(name, *a, **k)
    finally:
        acc["s"] += time.perf_counter() - t0
        acc["n"] += 1
        names[name] += 1


rt.launch = launch

rows, prev, tprev = [], dict(acc), time.perf_counter()
_orig = builtins.print


def spy(*a, **k):
    global prev, tprev
    _orig(*a, **k)
    s = a[0] if a else ""
    if isinstance(s, str) and s.startswith("step"):
        now, tw = dict(acc), time.perf_counter()
        rows.append((s.split()[1], now["n"] - prev["n"],
                     (now["s"] - prev["s"]) * 1e3, (tw - tprev) * 1e3))
        prev.clear()
        prev.update(now)
        tprev = tw


builtins.print = spy
from tgai_go.train import v3_pretrain as VP                  # noqa: E402

try:
    VP.main()
except SystemExit:
    pass
builtins.print = _orig

print("\n[muon-why] opt=%s mode=%s" % (OPT, MODE))
print("   step   launches   launch_host_ms   step_wall_ms")
for tag, n, ms, w in rows:
    print("   %-7s %7d %14.1f %14.1f" % (tag, n, ms, w))
if len(rows) >= 2:
    n, ms, w = rows[-1][1], rows[-1][2], rows[-1][3]
    print("\n  稳态：%d 次发射 / %.1f ms 主机 / %.1f ms 墙钟（主机占 %.0f%%）"
          % (n, ms, w, 100.0 * ms / w))
    print("  平均每次发射 %.2f µs" % (ms * 1e3 / max(1, n)))
print("\n  内核种类前 12（全程）：")
for nm, c in names.most_common(12):
    print("    %-24s %6d" % (nm, c))
