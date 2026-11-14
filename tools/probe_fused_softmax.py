# -*- coding: utf-8 -*-
"""验收重写后的融合 attn_mask_softmax：数值 + 与"掩码+通用softmax"的耗时对比。"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402


def ref_softmax(a):
    n = a.shape[-1]
    mask = np.triu(np.ones((n, n), np.float32), 1) > 0
    x = np.where(mask, -1e30, a)
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def main():
    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    for n, rows in ((64, 64), (512, 256), (2048, 128)):
        a = rng.normal(0, 1, (rows, n)).astype(np.float32)
        t = T.tensor(a.copy())
        p = T.functional.attn_mask_softmax(t, causal=True)
        q = np.asarray(p.to_numpy(), np.float32)
        err = float(np.abs(q - ref_softmax(a)).max())

        # 耗时：融合内核 vs 掩码+通用 softmax
        t1 = T.tensor(a.copy()); T.functional.attn_mask_softmax(t1, causal=True)
        T._cuda().empty_cache()
        s = time.time()
        for _ in range(5):
            tt = T.tensor(a.copy())
            T.functional.attn_mask_softmax(tt, causal=True)
        fused = (time.time() - s) / 5

        s = time.time()
        for _ in range(5):
            tt = T.tensor(a.copy())
            mask = T.tensor(np.where(np.triu(np.ones((n, n), np.float32), 1) > 0,
                                     -1e30, 0.0).astype(np.float32))
            T.functional.softmax(tt + mask, axis=-1)
        plain = (time.time() - s) / 5
        print("n=%-5d rows=%-4d 误差=%.2e  融合=%.4fs  掩码+softmax=%.4fs  快 %.2fx"
              % (n, rows, err, fused, plain, plain / max(fused, 1e-9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
