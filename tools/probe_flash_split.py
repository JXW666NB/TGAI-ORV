# -*- coding: utf-8 -*-
"""把 flash 前向拆开计时：块循环本体 vs flash_attn 的外层（permute/logse/包装）。"""
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
    BH, S, D = 8, 2048, 64
    rng = np.random.default_rng(0)
    q0 = rng.normal(0, .3, (BH, S, D)).astype(np.float32)
    q = T.tensor(q0); k = T.tensor(q0); v = T.tensor(q0)

    # 预热（首次含 NVRTC / 内核加载）
    FO._fwd_device(q._gpu, k._gpu, v._gpu, True, 512, 512)
    rt.sync()

    for tile in (512, 1024):
        t0 = time.time()
        for _ in range(3):
            out = FO._fwd_device(q._gpu, k._gpu, v._gpu, True, tile, tile)
        rt.sync()
        inner = (time.time() - t0) / 3

        t0 = time.time()
        for _ in range(3):
            o = FO.flash_attn(q.reshape(1, 1, S, D) if False else q, k, v, causal=True)
        rt.sync()
        outer = (time.time() - t0) / 3
        print("tile=%-5d 只看块循环 %.4f s   完整 flash_attn %.4f s" % (tile, inner, outer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
