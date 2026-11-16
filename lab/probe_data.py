# -*- coding: utf-8 -*-
"""Probe the V100 for real V3 corpora / tokenized shards / tokenizer. ASCII-only output."""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOTS = [r"D:\TGAI-V3-1.5B", r"D:\tgai_corpus"]


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return "%.2f%s" % (n, u)
        n /= 1024.0
    return "%.2fPB" % (n,)


def scan(root, max_depth=3):
    pats = {}
    total = 0
    n = 0
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            ents = list(os.scandir(d))
        except Exception:
            continue
        for e in ents:
            try:
                if e.is_dir():
                    if depth < max_depth:
                        stack.append((e.path, depth + 1))
                    continue
                ext = os.path.splitext(e.name)[1].lower()
                sz = e.stat().st_size
            except Exception:
                continue
            total += sz
            n += 1
            key = (os.path.relpath(os.path.dirname(e.path), root), ext)
            rec = pats.setdefault(key, [0, 0, 0])
            rec[0] += 1
            rec[1] += sz
            if sz > rec[2]:
                rec[2] = sz
    return pats, total, n


for root in ROOTS:
    print("=" * 70)
    print("ROOT", root, "exists=", os.path.isdir(root))
    if not os.path.isdir(root):
        continue
    pats, total, n = scan(root)
    print("files=%d bytes=%d (%s)" % (n, total, human(total)))
    rows = sorted(pats.items(), key=lambda kv: -kv[1][1])[:30]
    for (rel, ext), (cnt, sz, mx) in rows:
        print("  %-56s %-6s n=%-6d %10s max=%s"
              % (rel[:56], ext, cnt, human(sz), human(mx)))

for sub in ("01_corpus", "02_tokenizer", "03_src", "04_runs"):
    p = os.path.join(r"D:\TGAI-V3-1.5B", sub)
    print("=" * 70)
    print("DIR", p, "exists=", os.path.isdir(p))
    if os.path.isdir(p):
        for e in sorted(os.listdir(p))[:40]:
            fp = os.path.join(p, e)
            try:
                sz = os.path.getsize(fp) if os.path.isfile(fp) else -1
            except Exception:
                sz = -2
            print("   %-46s %s" % (e[:46], human(sz) if sz >= 0 else "<dir>"))

print("=" * 70)
print("candidate packed shards (part-*.npz / *.npy / *.bin) under TGAI-V3-1.5B:")
hits = []
for pat in ("part-*.npz", "*.npy", "*.bin"):
    for p in glob.glob(os.path.join(r"D:\TGAI-V3-1.5B", "**", pat), recursive=True):
        if "\\03_src\\" in p:
            continue
        hits.append(p)
for p in sorted(hits)[:40]:
    print("  ", os.path.getsize(p), p)
print("  total hits", len(hits))

print("=" * 70)
print("clean jsonl sample:")
cl = glob.glob(os.path.join(r"D:\TGAI-V3-1.5B\01_corpus\clean", "*.jsonl"))
cl.sort()
print("  n_files", len(cl))
for p in cl[:3]:
    print("  ", os.path.getsize(p), p)
if cl:
    import io
    with io.open(cl[0], encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            try:
                o = json.loads(line)
                print("   keys=", sorted(o.keys()), " len(text)=", len(str(o.get("text", ""))))
                print("   head=", str(o.get("text", ""))[:160].replace("\n", "\\n"))
            except Exception as ex:
                print("   parse fail", ex, line[:100])

print("=" * 70)
print("tokenizer files:")
for d in (r"D:\TGAI-V3-1.5B\02_tokenizer",):
    if os.path.isdir(d):
        for root, dirs, files in os.walk(d):
            for fn in files[:20]:
                fp = os.path.join(root, fn)
                print("   ", os.path.getsize(fp), fp)
