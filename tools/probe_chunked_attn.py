# -*- coding: utf-8 -*-
"""验分块注意力与整体注意力数值一致（V100）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V  # noqa: E402
from tgai_go.train import v3_attn_chunked as C  # noqa: E402


def np_(t):
    r = t.to_numpy() if hasattr(t, "to_numpy") else t
    return np.asarray(r, np.float32)


def main():
    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    B, S, d, n_q, n_kv = 1, 512, 512, 8, 2
    d_k = d // n_q
    x = T.tensor(rng.normal(size=(B, S, d)).astype(np.float32), requires_grad=True)
    ws = [T.tensor(rng.normal(0, .02, sh).astype(np.float32)) for sh in
          ((n_q * d_k, d), (n_kv * d_k, d), (n_kv * d_k, d), (d, n_q * d_k))]
    cos, sin = V.rope_cos_sin(S, d_k, 10000.0)

    full = V.attention_orv(x, *ws, cos, sin, d_k, n_q, n_kv, causal=True)
    ref = np_(full)

    for chunk in (128, 256):
        got = C.attention_orv_chunked(x, *ws, cos, sin, d_k, n_q, n_kv, chunk=chunk)
        err = float(np.abs(np_(got) - ref).max())
        print("chunk=%-4d max_err=%.3e" % (chunk, err))

    # 反向也要一致：对 sum 求梯度
    full.sum().backward()
    g1 = np_(x.grad)
    x3 = T.tensor(np.asarray(x.to_numpy(), np.float32), requires_grad=True)
    out3 = C.attention_orv_chunked(x3, *ws, cos, sin, d_k, n_q, n_kv, chunk=256)
    out3.sum().backward()
    g3 = np_(x3.grad)
    print("反向 max_err=%.3e" % float(np.abs(g1 - g3).max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
