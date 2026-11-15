# -*- coding: utf-8 -*-
"""确认 ORV 的 reshape 在非连续（转置）张量上是否给出错误数据。

期望（PyTorch 语义）：reshape 在无法保持 view 时应**复制**，结果仍等于逻辑上的形状重排。
如果 ORV 直接按原缓冲重排，就是静默错误。
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402


def main():
    T.set_default_device("cuda")
    a = np.arange(8, dtype=np.float32).reshape(4, 2)
    x = T.from_numpy(a)
    print("x      :", x.to_numpy().tolist())
    xt = x.T
    print("x.T    :", xt.to_numpy().tolist(), "形状", xt.shape)
    want = np.asarray(xt.to_numpy(), np.float32).reshape(8)
    print("期望 reshape(8):", want.tolist())
    print("ORV reshape(8) :", xt.reshape(8).to_numpy().tolist())
    got2 = xt.contiguous().reshape(8)
    print("先 contiguous :", got2.to_numpy().tolist())
    print("contiguous 是否正确:", bool(np.allclose(got2.to_numpy(), want)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
