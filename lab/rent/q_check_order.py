# -*- coding: utf-8 -*-
"""验证 `q_pyt.WindowStream` 与包内 `v3_pretrain.PackedCorpus` **取到的是同一批窗口**。

做法：两侧各取 N 个窗口的 `input_ids` 行，逐位比对（不比对内部计数器，比数据本身）。
在 ORV 的 python 下跑（既有 tgai_go 也有 q_pyt.py）。
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, r"D:\orv_bench")
sys.path.insert(0, r"D:\orv_pip_0962\Lib\site-packages")

from tgai_go.train import v3_pretrain as VP  # noqa: E402
import q_pyt  # noqa: E402

DATA = r"D:\orv_bench\data2048_real"
SEQ = 512
N = 80

paths = sorted(glob.glob(os.path.join(DATA, "*.npz")))
corpus = VP.PackedCorpus(paths, seq=SEQ, seed=42, vocab=96000, verbose=False)
mine = q_pyt.WindowStream(DATA, SEQ, seed=42)

bad = 0
for i in range(N):
    x_ref, y_ref, s_ref = corpus.take(1)
    x_my, y_my, s_my = mine.next_window()
    eqx = np.array_equal(np.asarray(x_ref[0]), x_my)
    eqy = np.array_equal(np.asarray(y_ref[0]), y_my)
    eqs = np.array_equal(np.asarray(s_ref[0]), s_my)
    if not (eqx and eqy and eqs):
        bad += 1
        if bad <= 3:
            print("MISMATCH @%d  x=%s y=%s s=%s  ref_head=%s my_head=%s"
                  % (i, eqx, eqy, eqs, np.asarray(x_ref[0])[:6], x_my[:6]))
print("[order] compared %d windows: mismatches=%d" % (N, bad))
print("[order] corpus state: epoch=%d rank=%d offset=%d consumed=%d"
      % (corpus.epoch, corpus.rank, corpus.offset, corpus.consumed))
print("[order] mine  state: epoch=%d rank=%d offset=%d consumed=%d"
      % (mine.epoch, mine.rank, mine.offset, mine.taken))
print("OK" if bad == 0 else "FAILED")
