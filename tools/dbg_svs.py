# -*- coding: utf-8 -*-
"""在 CUDA 上定位 svs-ops 的数值偏差：三角矩阵乘 + cat/转置 是否与 numpy 一致。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import plugin  # noqa: E402

plugin.load("examples/plugins/svs_audio_ops.py")
plugin.activate("svs-ops")


def tri(n):
    return T.from_numpy(np.tril(np.ones((n, n), np.float32)))


def rowwise(mat, x):
    n = int(mat.shape[1])
    m = int(mat.shape[0])
    lead = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
    flat = x.reshape(lead, n)
    cols = [T.matmul(mat, flat[i].reshape(n, 1)) for i in range(lead)]
    y = T.cat(cols, dim=1).T
    return y.reshape(tuple(list(x.shape[:-1]) + [m]))


def main():
    T.set_default_device("cuda")
    x = np.array([[1., 2., 3., 4.], [10., 20., 30., 40.]], np.float32)
    xo = T.from_numpy(x)
    print("x   :", xo.to_numpy().tolist())
    print("期望:", np.cumsum(x, -1).tolist())
    print("插件 cumsum:", T.svs.cumsum(xo, -1).to_numpy().tolist())
    print("手写 rowwise:", rowwise(tri(4), xo).to_numpy().tolist())

    L = tri(4)
    print("L  :", L.to_numpy().tolist())
    col = T.matmul(L, T.from_numpy(x[0].reshape(4, 1)))
    print("L@col0:", col.to_numpy().ravel().tolist())
    c1 = T.matmul(L, T.from_numpy(x[0].reshape(4, 1)))
    c2 = T.matmul(L, T.from_numpy(x[1].reshape(4, 1)))
    cc = T.cat([c1, c2], dim=1)
    print("cat dim=1 形状:", cc.shape, "值:", cc.to_numpy().tolist())
    print("cat 转置:", cc.T.to_numpy().tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
