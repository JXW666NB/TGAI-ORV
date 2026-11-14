# -*- coding: utf-8 -*-
"""验收重写后的融合 attn_mask_softmax：数值 + 与"掩码+通用softmax"的耗时对比。

融合路径由 TG_SOFTMAX_CAUSAL=1 打开（_sdpa_materialized 内），本探针直接比较两条路。
"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import _softmax_causal_cuda  # noqa: E402


def ref_softmax(a):
    n = a.shape[-1]
    mask = np.triu(np.ones((n, n), np.float32), 1) > 0
    x = np.where(mask, -1e30, a)
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def main():
    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    for rows, n in ((256, 512), (512, 1024), (1024, 2048)):
        a = rng.normal(0, 1, (rows, n)).astype(np.float32)
        p = _softmax_causal_cuda(T.tensor(a.copy()))
        err = float(np.abs(np.asarray(p.to_numpy(), np.float32) - ref_softmax(a)).max())

        mask_np = np.where(np.triu(np.ones((n, n), np.float32), 1) > 0, -1e30, 0.0).astype(np.float32)
        mask_t = T.tensor(mask_np)

        _softmax_causal_cuda(T.tensor(a.copy()))
        T._cuda().empty_cache()
        s = time.time()
        for _ in range(5):
            _softmax_causal_cuda(T.tensor(a.copy()))
        fused = (time.time() - s) / 5

        s = time.time()
        for _ in range(5):
            tt = T.tensor(a.copy())
            T.functional.softmax(tt + mask_t, axis=-1)
        plain = (time.time() - s) / 5
        print("rows=%-5d n=%-5d 误差=%.2e  融合=%.4fs  掩码+softmax=%.4fs  快 %.2fx"
              % (rows, n, err, fused, plain, plain / max(fused, 1e-9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
