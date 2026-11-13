# -*- coding: utf-8 -*-
"""设备侧块化版：数值对拍 + 计时（与主机版、物化版比）。"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V  # noqa: E402
from tgai_go.train import v3_flash as FH  # noqa: E402
from tgai_go.train import v3_flash_dev as FD  # noqa: E402


def main():
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    B, d, nq, nkv = 1, 512, 8, 2
    dk = d // nq
    x = T.tensor(rng.normal(size=(B, S, d)).astype(np.float32))
    ws = [T.tensor(rng.normal(0, .02, sh).astype(np.float32)) for sh in
          ((nq * dk, d), (nkv * dk, d), (nkv * dk, d), (d, nq * dk))]
    cos, sin = V.rope_cos_sin(S, dk, 10000.0)

    ref = np.asarray(V.attention_orv(x, *ws, cos, sin, dk, nq, nkv, causal=True).to_numpy(),
                     np.float32)

    for name, fn in (("host", FH.attention_flash_forward),
                     ("device", FD.attention_flash_fwd_device)):
        t0 = time.time()
        got, _ = fn(x, *ws, cos, sin, dk, nq, nkv, cq=64, ck=64)
        dt = time.time() - t0
        print("%-7s S=%-5d time=%.3fs  max_err=%.3e"
              % (name, S, dt, float(np.abs(got - ref).max())))
    t0 = time.time()
    V.attention_orv(x, *ws, cos, sin, dk, nq, nkv, causal=True)
    print("full    S=%-5d time=%.3fs" % (S, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
