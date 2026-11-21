# -*- coding: utf-8 -*-
"""Summarize probe_cast_sites.py dumps (ASCII only)."""
import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/root/sites_gemm.json",
                   encoding="utf-8"))
marks = d["marks"]
k = max(1, len(marks) - 1)
print("=== per-step casts by model call site (%d steps) ===" % k)
agg = {}
for m in marks[1:]:
    for s, n in m["by_site"].items():
        agg[s] = agg.get(s, 0) + n
tot = sum(agg.values())
for s, n in sorted(agg.items(), key=lambda kv: -kv[1]):
    print("  %7.1f /step  %5.1f%%  %s" % (n / k, 100.0 * n / tot, s))
print("  total %.1f casts/step" % (tot / k))
print()
print("=== biggest (site, elements, pinned) buckets ===")
rows = [[list(ke), v] for ke, v in d["sites"]]
for ke, v in rows[:24]:
    print("  %7.1f /step  n=%-9d %s  %s" % (v / k, ke[1], ke[2], ke[0][:110]))
