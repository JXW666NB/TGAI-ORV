# -*- coding: utf-8 -*-
"""验证假设：1.5B 的步时波动是否由 CSA 分块规模（随 batch 文档结构变化）决定。

做法：用与 v3_pretrain 完全相同的语料/种子复现前 N 个 batch，
对每个 batch 调 `v3_csa.make_plan(..., seg=seg)`，打印分块统计，
再和实测步时（同样是这批 batch）对照。

实测步时（1.5B / 24 层 / seq512，12 步）：
  1.068 0.648 0.547 0.649 0.631 0.857 1.150 1.097 1.164 0.943 1.038 0.765
"""
import argparse

import numpy as np

from tgai_go.train import v3_csa as C

TIMES = [1.068, 0.648, 0.547, 0.649, 0.631, 0.857, 1.150, 1.097, 1.164, 0.943,
         1.038, 0.765]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\TGAI-V3-1.5B\01_corpus\packed512\s512")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=12)
    a = ap.parse_args()

    from tgai_go.train import v3_pretrain as P

    paths = P.list_shards(a.data) if hasattr(P, "list_shards") else None
    corpus = P.PackedCorpus(paths, seq=a.seq, seed=42)
    csac = C.CSAConfig(d=1024, n_q=16, n_kv=16, compress=4, conv_kernel=3,
                       local_window=0) if hasattr(C, "CSAConfig") else None
    print("CSAConfig 可用:", csac is not None)

    print("\n step | 耗时 s | Tc | n_chunk.max | 总 chunk | 文档数 | 平均文档长")
    rows = []
    for i in range(a.steps):
        ids, labels, seg = corpus.next_batch(1)
        seg_np = np.asarray(seg)
        nch = np.asarray([int((r > 0).sum()) + 1 for r in seg_np.reshape(1, -1)]) \
            if seg_np.ndim == 2 else np.asarray([1])
        plan = C.make_plan(csac, 1, a.seq, seg=(seg if csac is not None else None))
        tc = getattr(plan, "Tc", None)
        ncmax = getattr(plan, "n_chunk", None)
        tot = int(np.asarray(ncmax).sum()) if ncmax is not None else -1
        nmax = int(np.asarray(ncmax).max()) if ncmax is not None else -1
        n_docs = int(len(np.unique(seg_np))) if seg_np is not None else -1
        avg_len = a.seq / max(n_docs, 1)
        t = TIMES[i] if i < len(TIMES) else float("nan")
        print(" %4d | %6.3f | %s | %11d | %8d | %6d | %8.1f"
              % (i, t, tc, nmax, tot, n_docs, avg_len))
        rows.append((t, nmax, tot, n_docs))

    arr = np.asarray([r[1:] for r in rows], dtype=np.float64)
    ts = np.asarray([r[0] for r in rows], dtype=np.float64)
    print("\n相关性（步时 vs 各统计量）:")
    names = ["n_chunk.max", "总chunk", "文档数"]
    for j, nm in enumerate(names):
        x = arr[:, j]
        if np.std(x) < 1e-9:
            print("  %-12s 常数，无法相关" % nm)
            continue
        c = float(np.corrcoef(x, ts)[0, 1])
        print("  %-12s r = %+.3f" % (nm, c))


if __name__ == "__main__":
    main()
