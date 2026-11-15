# -*- coding: utf-8 -*-
"""在 CUDA 上定位 svs-ops 的数值偏差（cumsum / _rowwise_mm / cat 的转置）。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import plugin  # noqa: E402

plugin.load("examples/plugins/svs_audio_ops.py")
plugin.activate("svs-ops")
import examples.plugins.svs_audio_ops as M  # noqa: E402


def main():
    T.set_default_device("cuda")
    x = np.array([[1., 2., 3., 4.], [10., 20., 30., 40.]], np.float32)
    xo = T.from_numpy(x)
    print("x cuda:", xo.to_numpy().tolist())
    print("cumsum 我们:", T.svs.cumsum(xo, -1).to_numpy().tolist())
    print("cumsum 期望:", np.cumsum(x, -1).tolist())

    L = M._tri(4)
    print("L cuda:", L.to_numpy().tolist())
    print("L@x:", M._rowwise_mm(L, xo).to_numpy().tolist())

    # 单行、单个 matmul [4,4]@[4,1]
    col = T.matmul(L, T.from_numpy(x[0].reshape(4, 1)))
    print("matmul 单列:", col.to_numpy().ravel().tolist())

    # cat(dim=1) 两个 [4,1]
    c1 = T.matmul(L, T.from_numpy(x[0].reshape(4, 1)))
    c2 = T.matmul(L, T.from_numpy(x[1].reshape(4, 1)))
    cc = T.cat([c1, c2], dim=1)
    print("cat dim=1 形状:", cc.shape, "值:", cc.to_numpy().tolist())
    print("其转置:", cc.T.to_numpy().tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
