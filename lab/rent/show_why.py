# -*- coding: utf-8 -*-
"""Summarize probe_why.py dump (ASCII only)."""
import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/root/why.json",
                   encoding="utf-8"))
c = d["counts"]
mk = d["marks"]
k = max(1, len(mk) - 1)
print("steps=%d  distinct (ptr,count) keys cast during run: %d"
      % (k, d["distinct_keys"]))
for key in ("cached_hit", "cached_miss", "clear_calls", "clear_entries",
            "drop_removed", "drop_noop", "drop_was_pinned_free", "free_calls"):
    v = c.get(key, 0)
    print("  %-22s %8.1f /step   (total %d)" % (key, v / k, v))
print("top re-cast keys (elements, times): %r"
      % ([(t[0][1], t[1]) for t in d["top_keys"]],))
