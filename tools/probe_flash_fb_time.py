# -*- coding: utf-8 -*-
"""flash 前向与反向分别耗时（S=2048, 块 512/512）。"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_flash_op as FO  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    B, H, S, D = 1, 8, 2048, 64
    rng = np.random.default_rng(0)
    q0 = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)
    k0 = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)
    v0 = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)

    # 前向
    t0 = time.time()
    for _ in range(3):
        q = T.tensor(q0); k = T.tensor(k0); v = T.tensor(v0)
        o = FO.flash_attn(q, k, v, causal=True)
        del q, k, v, o
    fwd = (time.time() - t0) / 3

    # 前向 + 反向
    t0 = time.time()
    for _ in range(3):
        q = T.tensor(q0, requires_grad=True)
        k = T.tensor(k0, requires_grad=True)
        v = T.tensor(v0, requires_grad=True)
        o = FO.flash_attn(q, k, v, causal=True)
        o.sum().backward()
        del q, k, v, o
    full = (time.time() - t0) / 3

    # 纯 matmul 参考：同样 FLOP 的一次 512x64 @ 64x512
    print("前向         %.3f s" % fwd)
    print("前向+反向    %.3f s（反向 %.3f s，占 %.0f%%）" % (full, full - fwd,
                                                    100 * (full - fwd) / full))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
