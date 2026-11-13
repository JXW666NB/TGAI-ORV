# -*- coding: utf-8 -*-
"""验分块在线 softmax 前向与整体注意力数值一致（V100）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V  # noqa: E402
from tgai_go.train import v3_flash as F  # noqa: E402


def main():
    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    B, S, d, nq, nkv = 1, 512, 512, 8, 2
    dk = d // nq
    x = T.tensor(rng.normal(size=(B, S, d)).astype(np.float32))
    ws = [T.tensor(rng.normal(0, .02, sh).astype(np.float32)) for sh in
          ((nq * dk, d), (nkv * dk, d), (nkv * dk, d), (d, nq * dk))]
    cos, sin = V.rope_cos_sin(S, dk, 10000.0)

    ref = np.asarray(V.attention_orv(x, *ws, cos, sin, dk, nq, nkv, causal=True).to_numpy(),
                     np.float32)
    for cq in (128, 256):
        got, lse = F.attention_flash_forward(x, *ws, cos, sin, dk, nq, nkv, cq=cq, ck=cq)
        print("flash cq=ck=%-4d max_err=%.3e  lse_shape=%s"
              % (cq, float(np.abs(got - ref).max()), lse.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
