# -*- coding: utf-8 -*-
"""Summarize probe_clear.py (ASCII only)."""
import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/root/probe_clear.json",
                   encoding="utf-8"))
print("total clears=%d  entries-sum=%d" % (d["counts"].get("calls", 0),
                                           d["counts"].get("entries", 0)))
print("by (site, entries-when-cleared):")
for k, v in d["sites"]:
    print("  %4d x  entries=%-4d  %s" % (v, k[1], k[0]))
print("first 40 clears in order (step_index, site, entries):")
for s in d["seq"][:40]:
    print("   step#%d  n=%d  %s" % (s[0], s[2], s[1]))
