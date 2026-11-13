# -*- coding: utf-8 -*-
"""验 `flash_attn` 作为 autograd 节点：输出与梯度都要对上物化 SDPA。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_flash_op as F  # noqa: E402


def np_(t):
    r = t.to_numpy() if hasattr(t, "to_numpy") else t
    return np.asarray(r, np.float32)


def main():
    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    B, H, S, D = 1, 2, 32, 8
    q = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)
    k = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)
    v = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)
    dO = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)

    for causal in (False, True):
        # 参照：物化 SDPA
        qr = T.tensor(q, requires_grad=True)
        kr = T.tensor(k, requires_grad=True)
        vr = T.tensor(v, requires_grad=True)
        ref = T.functional.scaled_dot_product_attention(qr, kr, vr, causal=causal)
        (ref * T.tensor(dO)).sum().backward()
        rq, rk, rv = np_(qr.grad), np_(kr.grad), np_(vr.grad)
        ro = np_(ref)

        # 被测：块化 flash_attn
        qf = T.tensor(q, requires_grad=True)
        kf = T.tensor(k, requires_grad=True)
        vf = T.tensor(v, requires_grad=True)
        out = F.flash_attn(qf, kf, vf, causal=causal, cq=8, ck=8)
        (out * T.tensor(dO)).sum().backward()
        fq, fk, fv = np_(qf.grad), np_(kf.grad), np_(vf.grad)

        print("causal=%-5s out=%.2e  dq=%.2e dk=%.2e dv=%.2e"
              % (causal, float(np.abs(np_(out) - ro).max()),
                 float(np.abs(fq - rq).max()), float(np.abs(fk - rk).max()),
                 float(np.abs(fv - rv).max())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
