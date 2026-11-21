# -*- coding: utf-8 -*-
"""Analyze probe_amp.py dumps (ASCII only)."""
import collections
import json
import os
import sys

BASE = os.environ.get("PROBE_DIR", "/tmp")

def load(tag):
    return json.load(open(os.path.join(BASE, "probe_%s.json" % tag),
                          encoding="utf-8"))

def per_step(d):
    """Delta between consecutive clip markers = one full step."""
    ms = d["marks"]
    out = []
    for i in range(1, len(ms)):
        a, b = ms[i - 1], ms[i]
        dt = b["t"] - a["t"]
        cnt = {}
        for k, v in b["counts"].items():
            dv = v - a["counts"].get(k, 0)
            if dv:
                cnt[k] = dv
        out.append({"i": i, "dt": dt, "counts": cnt,
                    "gemm": b["gemm_n"] - a["gemm_n"],
                    "miss": b["pool_misses"] - a["pool_misses"],
                    "hit": b["pool_hits"] - a["pool_hits"],
                    "cache": b["cache"]})
    return out

def report(tag):
    d = load(tag)
    print("=" * 70)
    print("MODE=%s (amp=%s)" % (tag, "gemm" if "gemm" in tag else "off"))
    print("total kernel launches by name (top 25):")
    for k, v in sorted(d["counts"].items(), key=lambda kv: -kv[1])[:25]:
        print("   %-32s %8d" % (k, v))
    print("gemm_f16 calls total = %d" % d["gemm_n"])
    ps = per_step(d)
    steady = [p for p in ps if p["i"] >= 3]
    print("per-step (i>=3 avg over %d steps):" % len(steady))
    agg = collections.Counter()
    for p in steady:
        for k, v in p["counts"].items():
            agg[k] += v
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:15]:
        print("   %-32s %9.1f / step" % (k, v / max(1, len(steady))))
    print("   gemm_f16 calls/step          %9.1f" % (sum(p["gemm"] for p in steady) / max(1, len(steady))))
    print("   pool misses/step             %9.1f" % (sum(p["miss"] for p in steady) / max(1, len(steady))))
    print("   pool hits/step               %9.1f" % (sum(p["hit"] for p in steady) / max(1, len(steady))))
    print("   step dt (s)                  %s" % [round(p["dt"], 4) for p in ps])
    print("cache state at last mark: %r" % (d["marks"][-1]["cache"],))
    print("n_cast_keys=%d  n_cast_calls=%d" % (d["n_cast_keys"], d["n_cast_calls"]))
    print("repeat histogram (times cast -> #distinct(ptr,count)): %s" % (d["repeat_hist"],))
    print("cast_by_count top: %s" % (d["cast_by_count"][:10],))
    print("gemm_ops top (m,n,k,apin,bpin)->n:")
    for k, v in d["gemm_ops"][:20]:
        print("   %-40s %7d" % (tuple(k), v))
    print("pool: %r" % ({k: d["pool"][k] for k in
                        ("hits", "misses", "hit_rate", "peak_live_bytes", "trims")},))
    print("sync waits: n=%d mean=%.4f max=%.4f"
          % (len(d["sync_wait"]), sum(d["sync_wait"]) / max(1, len(d["sync_wait"])),
             max(d["sync_wait"] or [0])))
    return d, ps

for tag in sys.argv[1:] or ["gemm", "off"]:
    d, ps = report(tag)
    # per-step duplicate analysis (only meaningful in count mode)
    ms = d["marks"]
    if ms and ms[0].get("cast_keys"):
        print("--- per-step duplicate cast analysis ---")
        for i in range(2, len(ms)):
            a, b = ms[i - 1], ms[i]
            ka, kb = a.get("cast_keys") or {}, b.get("cast_keys") or {}
            delta = {}
            for k, v in kb.items():
                dv = v - ka.get(k, 0)
                if dv > 0:
                    delta[tuple(k) if isinstance(k, str) else k] = dv
            tot = sum(delta.values())
            dup = sum(v - 1 for v in delta.values())
            print("  step %d: calls=%d  distinct(ptr,count)=%d  duplicate=%d (%.1f%%)"
                  % (i - 1, tot, len(delta), dup, 100.0 * dup / max(1, tot)))
            print("        chit=%d cmiss=%d  gemm=%d"
                  % (b["chit"] - a["chit"], b["cmiss"] - a["cmiss"],
                     b["gemm_n"] - a["gemm_n"]))
            top = sorted(delta.items(), key=lambda kv: -kv[1])[:6]
            print("        top repeats: %r" % [(k[1], v) for k, v in top])
        break
