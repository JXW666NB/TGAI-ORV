# -*- coding: utf-8 -*-
"""单独测 flash 注意力算子的前向 / 前向+反向 峰值随 seq 的走势（每档独立进程内清池）。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_flash_op as FO  # noqa: E402


def main():
    import os
    os.environ["FLASHDBG"] = "1"
    os.environ["FLASHDBG2"] = "1"
    T.set_default_device("cuda")
    rt = T._cuda()
    B, H, D = 1, 8, 64
    rng = np.random.default_rng(0)
    for S in (512, 1024, 2048):
        q0 = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)
        k0 = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)
        v0 = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)

        rt.empty_cache()
        p0 = rt.pool_stats()["peak_live_bytes"]
        q = T.tensor(q0); k = T.tensor(k0); v = T.tensor(v0)
        out = FO.flash_attn(q, k, v, causal=True)
        p1 = rt.pool_stats()["peak_live_bytes"]
        del q, k, v, out
        rt.empty_cache()

        q = T.tensor(q0, requires_grad=True)
        k = T.tensor(k0, requires_grad=True)
        v = T.tensor(v0, requires_grad=True)
        out = FO.flash_attn(q, k, v, causal=True)
        p2 = rt.pool_stats()["peak_live_bytes"]
        out.sum().backward()
        p3 = rt.pool_stats()["peak_live_bytes"]
        if S == 2048:
            st = rt.pool_stats()
            print("  miss_top（尺寸字节, 次数）:", st["miss_top"][:6])
            print("  live=%.0f MiB  occupied=%.0f MiB"
                  % (st["live_bytes"] / 2**20, st["occupied_bytes"] / 2**20))
        print("S=%-5d 前向=%5.0f MiB  前向+反向=%6.0f MiB（反向额外 %5.0f）"
              % (S, (p1 - p0) / 2**20, (p3 - p0) / 2**20, (p3 - p2) / 2**20))
        del q, k, v, out
        rt.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
