# -*- coding: utf-8 -*-
"""量分块在线 softmax 前向的显存与耗时（按 S 单独进程跑，避免峰值累计）。"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V  # noqa: E402
from tgai_go.train import v3_flash as F  # noqa: E402


def main():
    S = int(sys.argv[1])
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)
    B, d, nq, nkv = 1, 512, 8, 2
    dk = d // nq
    x = T.tensor(rng.normal(size=(B, S, d)).astype(np.float32))
    ws = [T.tensor(rng.normal(0, .02, sh).astype(np.float32)) for sh in
          ((nq * dk, d), (nkv * dk, d), (nkv * dk, d), (d, nq * dk))]
    cos, sin = V.rope_cos_sin(S, dk, 10000.0)

    before = rt.pool_stats()["peak_live_bytes"]
    t0 = time.time()
    F.attention_flash_forward(x, *ws, cos, sin, dk, nq, nkv, cq=64, ck=64)
    dt = time.time() - t0
    st = rt.pool_stats()
    print("flash S=%4d  time=%.2fs  peak_live=%.0f MiB (before %.0f)"
          % (S, dt, st["peak_live_bytes"] / 2**20, before / 2**20))

    # 对照：整体注意力（物化）在同配置下的峰值
    t0 = time.time()
    V.attention_orv(x, *ws, cos, sin, dk, nq, nkv, causal=True)
    dt2 = time.time() - t0
    st2 = rt.pool_stats()
    print("full  S=%4d  time=%.2fs  peak_live=%.0f MiB"
          % (S, dt2, st2["peak_live_bytes"] / 2**20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
