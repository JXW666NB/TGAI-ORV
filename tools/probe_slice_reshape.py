# -*- coding: utf-8 -*-
"""决定性验证：在**切片视图**上 reshape，数值是否仍然正确（形状对但值错是最隐蔽的坑）。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")

import tgai_go.train as T  # noqa: E402


def np_(t):
    r = t.to_numpy() if hasattr(t, "to_numpy") else t
    return np.asarray(r, np.float32)


def main():
    T.set_default_device("cuda")
    B, H, S, D = 2, 3, 16, 4
    a = np.arange(B * H * S * D, dtype=np.float32).reshape(B, H, S, D)

    for (lo, hi) in ((0, 8), (8, 16)):
        sub_np = a[:, :, lo:hi, :]
        want = sub_np.reshape(B * H, hi - lo, D)
        got = np_(T.tensor(a)[:, :, lo:hi, :].reshape(B * H, hi - lo, D))
        print("slice[%2d:%2d].reshape shape=%s 值一致=%s max_err=%.1f"
              % (lo, hi, got.shape, np.allclose(got, want, atol=1e-6),
                 float(np.abs(got - want).max())))

    # 转置后的切片：flash 里 q 是 transpose(1,2) 得到的，非连续视图
    q = T.tensor(a).transpose(1, 2)              # (B,S,H,D)
    sub_np = a.transpose(0, 2, 1, 3)[:, 4:12, :, :]
    want = sub_np.reshape(B * S, 8, D) if False else sub_np
    got = np_(q[:, 4:12, :, :])
    print("transpose 后切片 shape=%s 值一致=%s"
          % (got.shape, np.allclose(got, sub_np, atol=1e-6)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
