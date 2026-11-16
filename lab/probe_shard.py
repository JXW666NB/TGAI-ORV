# -*- coding: utf-8 -*-
"""Inspect packed npz shards: keys, shapes, dtypes, id ranges, label stats, scan cost."""
import glob
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

CANDS = [
    r"D:\TGAI-V3-1.5B\01_corpus\packed\train",
    r"D:\TGAI-V3-1.5B\01_corpus\packed\val",
    r"D:\TGAI-V3-1.5B\01_corpus\packed_1024\train",
    r"D:\TGAI-V3-1.5B\01_corpus\packed512\s512",
    r"D:\TGAI-V3-1.5B\01_corpus\packed512\s1024",
    r"D:\TGAI-V3-1.5B\01_corpus\packed_seq\s512",
]

for d in CANDS:
    print("=" * 74)
    print("DIR", d, "exists=", os.path.isdir(d))
    if not os.path.isdir(d):
        continue
    fs = sorted(glob.glob(os.path.join(d, "*.npz")))
    print("  n_shards", len(fs))
    if not fs:
        continue
    p = fs[0]
    z = np.load(p)
    print("  first:", os.path.basename(p), os.path.getsize(p))
    for k in z.files:
        a = z[k]
        print("    %-10s shape=%-14s dtype=%s" % (k, a.shape, a.dtype))
    ids = z["input_ids"]
    print("  id min/max", int(ids.min()), int(ids.max()))
    if "labels" in z.files:
        lab = z["labels"]
        print("  labels: ==ids frac %.4f   -100 frac %.4f"
              % (float((lab == ids).mean()), float((lab == -100).mean())))
        print("  labels[0][:8]", lab[0][:8].tolist())
        print("  ids[0][:8]   ", ids[0][:8].tolist())
        # doc starts: positions where labels==-100 (excluding pad at tail)
        row = lab[0]
        neg = np.flatnonzero(row == -100)
        print("  row0 -100 positions[:12]", neg[:12].tolist(), "count", len(neg))
    if "seg_id" in z.files:
        seg = z["seg_id"]
        print("  seg: min/max %d/%d  row0 n_docs %d"
              % (int(seg.min()), int(seg.max()), int(seg[0].max()) + 1))
        print("  seg[0][:16]", seg[0][:16].tolist())
    print("  pad id 256 count frac %.4f" % float((ids == 256).mean()))

print("=" * 74)
print("header-only scan cost (all shards of packed_1024/train):")
d = r"D:\TGAI-V3-1.5B\01_corpus\packed_1024\train"
fs = sorted(glob.glob(os.path.join(d, "*.npz")))
t0 = time.time()
tot = 0
import zipfile
for p in fs[:200]:
    with zipfile.ZipFile(p) as zf:
        with zf.open("input_ids.npy") as f:
            np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format._read_array_header(f, (1, 0))
            tot += shape[0]
print("  200 shards header scan %.3f s  total_rows %d" % (time.time() - t0, tot))
print("  projected for %d shards: %.2f s" % (len(fs), (time.time() - t0) / 200.0 * len(fs)))

# full scan totals
t0 = time.time()
nrows = 0
shapes = set()
for p in fs:
    with zipfile.ZipFile(p) as zf:
        with zf.open("input_ids.npy") as f:
            np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format._read_array_header(f, (1, 0))
            nrows += shape[0]
            shapes.add(shape)
print("  FULL scan %d shards: %.2f s rows=%d shapes=%s"
      % (len(fs), time.time() - t0, nrows, sorted(shapes)))
print("  tokens = %d (x seq)" % nrows)
