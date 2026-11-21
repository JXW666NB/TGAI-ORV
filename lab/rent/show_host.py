# -*- coding: utf-8 -*-
"""Summarize probe_host.py dumps (ASCII only)."""
import json
import os
import sys

BASE = os.environ.get("PROBE_DIR", "/root")


def show(tag):
    p = os.path.join(BASE, "ph_%s.json" % tag)
    d = json.load(open(p, encoding="utf-8"))
    ms = d["marks"]
    ns = len(ms) - 1
    if ns <= 0:
        print("%s: no marks" % tag)
        return
    # use last 3 steps
    warm = min(3, ns - 1)
    a, b = ms[warm], ms[-1]
    k = len(ms) - 1 - warm
    print("=" * 74)
    print("amp=%s   over %d steps (marks %d..%d)" % (tag, k, warm, len(ms) - 1))
    print("  step wall (s): %.4f" % ((b["t"] - a["t"]) / k))
    print("  %-24s %10s %10s" % ("section", "calls/step", "ms/step"))
    for lab in sorted(b["ns"], key=lambda x: -(b["ns"][x] - a["ns"].get(x, 0))):
        dn = b["n"].get(lab, 0) - a["n"].get(lab, 0)
        dt = (b["ns"].get(lab, 0) - a["ns"].get(lab, 0)) / 1e6
        print("  %-24s %10.1f %10.2f" % (lab, dn / k, dt / k))
    print("  pool hits/step %d  misses/step %d"
          % ((b["pool"]["hits"] - a["pool"]["hits"]) / k,
             (b["pool"]["misses"] - a["pool"]["misses"]) / k))
    print("  cache at end: %r" % (b["cache"],))
    sw = d["sync_wait"]
    print("  sync n=%d  sum=%.3f s  mean=%.4f" % (len(sw), sum(sw), sum(sw) / max(1, len(sw))))


for t in sys.argv[1:] or ["gemm", "off"]:
    show(t)
