# -*- coding: utf-8 -*-
"""最小用例：2 个块，打印在线 softmax 的每个中间量，定位 NaN 来源。

只测 `T.cat(...).max(axis=-1)` 这个"逐元素取大"的组合是否与 numpy 一致。
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")

import tgai_go.train as T  # noqa: E402


def np_(t):
    r = t.to_numpy() if hasattr(t, "to_numpy") else t
    return np.asarray(r, np.float32)


def main():
    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    rows, cols = 4, 3
    a = rng.normal(size=(2, rows)).astype(np.float32)      # 运行最大值 m
    b = rng.normal(size=(2, rows)).astype(np.float32)      # 本块行最大值

    at, bt = T.tensor(a), T.tensor(b)

    # ① 直接 cat + max
    cat = T.cat([at.reshape(2, rows, 1), bt.reshape(2, rows, 1)], dim=2)
    print("cat shape", tuple(cat.shape), "值一致:", np.allclose(np_(cat),
          np.stack([a, b], axis=-1), atol=1e-6))
    mm = cat.max(axis=-1)
    print("max shape", tuple(mm.shape), "期望形状 (2, %d)" % rows)
    print("max 值一致:", np.allclose(np_(mm), np.maximum(a, b), atol=1e-6))
    print("  got ", np_(mm).ravel()[:6])
    print("  want", np.maximum(a, b).ravel()[:6])

    # ② 单独测 sb.max(axis=-1) 的形状语义
    sb = T.tensor(rng.normal(size=(2, rows, cols)).astype(np.float32))
    mr = sb.max(axis=-1)
    print("sb.max shape", tuple(mr.shape), "期望 (2, %d)" % rows)

    # ③ 测 (-1e30 起始) 与 log
    x = T.tensor(np.array([[-1e30, 0.0, 1.0]], np.float32))
    print("exp 后:", np_(x.exp()).ravel())
    l = T.tensor(np.array([[0.0, 2.0, 3.0]], np.float32))
    print("log(0) →", np_(l.log()).ravel())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
