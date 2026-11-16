# -*- coding: utf-8 -*-
"""Local CPU smoke test for the data layer of v3_pretrain (no CUDA needed)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
from tgai_go.train import v3_pretrain as M  # noqa: E402

tmp = tempfile.mkdtemp(prefix="v3p_")
print("tmp", tmp)


def make_npz(path, n_rows, width, vocab=96000, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab, (n_rows, width)).astype(np.int32)
    lab = ids.copy()
    seg = np.zeros((n_rows, width), np.int16)
    # doc starts at 0 and mid
    lab[:, 0] = -100
    lab[:, width // 2] = -100
    seg[:, width // 2:] = 1
    np.savez(path, input_ids=ids, labels=lab, seg_id=seg)


d = os.path.join(tmp, "shards")
os.makedirs(d)
for i in range(3):
    make_npz(os.path.join(d, "part-%05d.npz" % i), 7, 16, seed=i)

paths = M.expand_data_paths([d])
print("paths", len(paths))
corpus = M.PackedCorpus(paths, seq=8, seed=7, vocab=96000)
print("stats", corpus.stats())
print("windows", corpus.total_windows, "expect", 3 * 7 * 2)
hi = corpus.check_vocab()

# order determinism + resume
c1 = M.PackedCorpus(paths, seq=8, seed=7, vocab=96000)
first = [c1.take(1)[0][0, 0] for _ in range(10)]
c2 = M.PackedCorpus(paths, seq=8, seed=7, vocab=96000)
second = [c2.take(1)[0][0, 0] for _ in range(10)]
print("same seed order equal:", first == second)

c3 = M.PackedCorpus(paths, seq=8, seed=7, vocab=96000)
for _ in range(5):
    c3.take(1)
st = c3.state_dict()
c4 = M.PackedCorpus(paths, seq=8, seed=7, vocab=96000)
c4.load_state_dict(st)
cont = [c4.take(1)[0][0, 0] for _ in range(5)]
c5 = M.PackedCorpus(paths, seq=8, seed=7, vocab=96000)
ref = [c5.take(1)[0][0, 0] for _ in range(10)]
print("resume continues identically:", cont == ref[5:], cont, ref[5:])

# epoch rollover
c6 = M.PackedCorpus(paths, seq=8, seed=7, vocab=96000)
n = c6.total_windows
for _ in range(n + 2):
    c6.take(1)
print("after rollover state", c6.state_dict())

# vocab guard
c7 = M.PackedCorpus(paths, seq=8, seed=7, vocab=4096)
try:
    c7.check_vocab()
except SystemExit as e:
    print("vocab guard fired:", str(e).splitlines()[0][:90])

# npy 2-D + flat bin
np.save(os.path.join(d, "..", "rows.npy"), np.random.default_rng(1).integers(0, 500, (5, 8)).astype(np.int32))
flat = np.random.default_rng(2).integers(0, 500, 8 * 10 + 3).astype(np.uint32)
flat.tofile(os.path.join(d, "..", "flat.bin"))
p2 = M.expand_data_paths([os.path.join(d, "..", "rows.npy")])
c8 = M.PackedCorpus(p2, seq=8, seed=1, vocab=1000)
print("npy2d", c8.stats(), c8.take(1)[0].shape)
p3 = M.expand_data_paths([os.path.join(d, "..", "flat.bin")])
try:
    M.PackedCorpus(p3, seq=8, seed=1)
except SystemExit as e:
    print("bin dtype guard fired:", str(e).splitlines()[0][:80])
c9 = M.PackedCorpus(p3, seq=8, seed=1, bin_dtype="uint32", vocab=1000)
print("bin", c9.stats(), c9.take(1)[0].shape)

# labels/seg extraction correctness
c10 = M.PackedCorpus(paths, seq=8, seed=7, vocab=96000)
ids, lab, seg = c10.take(1)
print("ids", ids.shape, "lab", lab.shape, "seg", seg.shape, "lab0", lab[0].tolist())
print("seg", seg[0].tolist())
assert seg[0].tolist() == ([0] * 8) or seg[0].tolist() == ([0] * 4 + [1] * 4), seg

# schedule helper
class A:
    lr = 3e-4
    min_lr_ratio = 0.1
    warmup_steps = 0
    warmup_ratio = 0.1
print("sched cfg", M._sched_cfg(A(), 60))

# scheduler state roundtrip
import tgai_go.train as T
class DummyOpt:
    lr = 3e-4
o = DummyOpt()
s = T.optim.lr_scheduler.CosineLR(o, T_max=50, eta_min=3e-5, warmup_steps=5)
seq = [s.step() for _ in range(12)]
o2 = DummyOpt()
s2 = T.optim.lr_scheduler.CosineLR(o2, T_max=50, eta_min=3e-5, warmup_steps=5)
for _ in range(5):
    s2.step()
s2.load_state_dict({"_n": 5}) if False else s2.load_state_dict({"n": 5, "lr": seq[4], "base_lr": 3e-4})
seq2 = [s2.step() for _ in range(7)]
print("sched resume equal:", np.allclose(seq[5:], seq2), seq[5:8], seq2[:3])
print("ALL LOCAL SMOKE OK")
