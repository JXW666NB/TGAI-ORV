# -*- coding: utf-8 -*-
"""验证 embedding 反向的 GPU scatter 与原来的 CPU np.add.at 数值一致。

`__getitem__` 的反向从 CPU 中转（np.zeros + np.add.at + h2d）换成
GPU `scatter_add_rows`。必须证明两者逐元素等价。
"""
import numpy as np

import tgai_go.train as T

T.set_default_device("cuda")
rt = T._cuda()
rng = np.random.default_rng(0)


def main():
    V, D, S = 5000, 128, 256
    ids = rng.integers(0, V, S).astype(np.int64)
    g = rng.normal(0, 1.0, (S, D)).astype(np.float32)      # 梯度 (S,D)

    # 参考：CPU 路径
    full = np.zeros((V, D), dtype=np.float32)
    np.add.at(full, ids, g)

    # GPU 路径
    id_buf = rt.h2d(ids.astype(np.int32), dtype=np.int32)
    gbuf = rt.h2d(g)
    dx = np.asarray(rt.d2h(rt.scatter_add_rows(gbuf, id_buf, V, D, S)), np.float32)
    rt.sync()

    d = np.abs(dx - full).max()
    rel = float(d / max(float(np.abs(full).max()), 1e-30))
    print("=== scatter_add_rows vs np.add.at（V=%d D=%d S=%d）===" % (V, D, S))
    print("  max_abs=%.3e  max_rel=%.3e" % (d, rel))
    print("  判定：%s" % ("PASS（逐元素等价）" if rel < 1e-6 else "FAIL"))

    # 重复索引（同一行被 scatter 多次）也要对
    ids2 = np.concatenate([ids[:50], ids[:50]]).astype(np.int64)
    g2 = rng.normal(0, 1.0, (100, D)).astype(np.float32)
    ref2 = np.zeros((V, D), dtype=np.float32)
    np.add.at(ref2, ids2, g2)
    dx2 = np.asarray(rt.d2h(rt.scatter_add_rows(
        rt.h2d(g2), rt.h2d(ids2.astype(np.int32), dtype=np.int32), V, D, 100)), np.float32)
    rt.sync()
    rel2 = float(np.abs(dx2 - ref2).max() / max(float(np.abs(ref2).max()), 1e-30))
    print("  重复索引 max_rel=%.3e  %s" % (rel2, "PASS" if rel2 < 1e-6 else "FAIL"))


if __name__ == "__main__":
    main()
