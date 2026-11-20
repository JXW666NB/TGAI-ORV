# -*- coding: utf-8 -*-
"""在 4090 本地生成 packed 分片（与 data2048_v96000 同格式同形状）。

真实数据是 seq=2048 的打包窗口，键为 input_ids / labels / seg_id。
吞吐只取决于形状与 seq，token 内容不影响计时；为了让 CE 的有效 token 数与真实
数据接近，这里把每个文档首 token 的 label 置 -100（约 1.4% 位置）。
"""
import os

import numpy as np

OUT = "/root/bench/data2048_v96000"
N_SHARD = 8
N_WIN = 16
T = 2048
VOCAB = 96000
DOCLEN = 700          # 每 ~700 token 一个文档边界 ⇒ 约 1.4% 的 -100

os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261120)
made = 0
for s in range(N_SHARD):
    ids = rng.integers(1, VOCAB, size=(N_WIN, T), dtype=np.int32)
    lab = ids.copy()
    seg = np.zeros((N_WIN, T), dtype=np.int32)
    for w in range(N_WIN):
        d = 0
        pos = 0
        while pos < T:
            seg[w, pos:pos + DOCLEN] = d
            lab[w, pos] = -100          # 文档首 token 不参与损失（与真实打包一致）
            pos += DOCLEN
            d += 1
    p = os.path.join(OUT, "part-%05d.npz" % s)
    np.savez(p, input_ids=ids, labels=lab, seg_id=seg)
    made += 1

tot = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
print("生成 %d 个分片，合计 %.2f MB，窗口 %d×%d，vocab %d"
      % (made, tot / 2 ** 20, N_WIN, T, VOCAB))
