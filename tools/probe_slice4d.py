# -*- coding: utf-8 -*-
"""验 ORV 的四维切片是否与 numpy 一致（NaN 的最大嫌疑）。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")

import tgai_go.train as T  # noqa: E402


def np_(t):
    r = t.to_numpy() if hasattr(t, "to_numpy") else t
    return np.asarray(r, np.float32)


def check(name, arr, sl, sl_np):
    t = T.tensor(arr)
    try:
        got = np_(t[sl])
    except Exception as e:
        print("%-22s ERR %s: %s" % (name, type(e).__name__, str(e)[:80]))
        return
    want = arr[sl_np]
    ok = got.shape == want.shape and np.allclose(got, want)
    print("%-22s shape=%s want=%s 一致=%s" % (name, got.shape, want.shape, ok))


def main():
    T.set_default_device("cuda")
    a4 = np.arange(4 * 8 * 16 * 4, dtype=np.float32).reshape(4, 8, 16, 4)
    check("4D [:, :, 8:16, :]", a4, (slice(None), slice(None), slice(8, 16), slice(None)),
          (slice(None), slice(None), slice(8, 16), slice(None)))
    check("4D [:, :, 0:8, :]", a4, (slice(None), slice(None), slice(0, 8), slice(None)),
          (slice(None), slice(None), slice(0, 8), slice(None)))
    check("4D [:, 1:2]", a4, (slice(None), slice(1, 2)),
          (slice(None), slice(1, 2)))

    a3 = np.arange(2 * 8 * 4, dtype=np.float32).reshape(2, 8, 4)
    check("3D [:, 0:3, :]", a3, (slice(None), slice(0, 3), slice(None)),
          (slice(None), slice(0, 3), slice(None)))

    # 变形 + 转置组合（flash 里实际用的那串）
    q = T.tensor(a4)
    sub = q[:, :, 8:16, :]
    r = sub.reshape(4 * 8, 8, 4)
    print("reshape 后 shape", tuple(r.shape), "期望 (32, 8, 4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
