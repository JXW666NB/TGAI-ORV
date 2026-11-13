# -*- coding: utf-8 -*-
"""测不同块大小下 flash 的前向+反向耗时与峰值（S=2048, B=1, H=8, D=64）。"""
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

    print("%-10s %10s %10s" % ("块大小", "前+反向秒", "峰值 MiB"))
    for cq, ck in ((256, 128), (256, 256), (512, 256), (512, 512)):
        rt.empty_cache()
        p0 = rt.pool_stats()["peak_live_bytes"]
        q = T.tensor(q0, requires_grad=True)
        k = T.tensor(k0, requires_grad=True)
        v = T.tensor(v0, requires_grad=True)
        t0 = time.time()
        out = FO.flash_attn(q, k, v, causal=True, cq=cq, ck=ck)
        out.sum().backward()
        dt = time.time() - t0
        pk = rt.pool_stats()["peak_live_bytes"]
        print("%-10s %10.3f %10.0f" % ("%d/%d" % (cq, ck), dt, (pk - p0) / 2**20))
        del q, k, v, out
        rt.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
