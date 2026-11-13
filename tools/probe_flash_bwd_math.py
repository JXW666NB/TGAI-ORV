# -*- coding: utf-8 -*-
"""flash 反向的 numpy 参考 + 与 ORV autograd 的梯度对拍（在 q/k/v 层面，隔离投影）。

反向公式（标准 flash，与 `kernel_backward.h` 同思路）：

```
P   = exp(S - LSE)                  # 归一化后就是 softmax 结果
dV += Pᵀ·dO
dP  = dO·Vᵀ
dS  = P ⊙ (dP − rowsum(dP ⊙ P))     # softmax 导数
dQ += dS·K · scale
dK += dSᵀ·Q · scale
```

这里用**整体物化**的方式算同一个梯度（只是不保存 P，按块重算），
目的是先确认公式与 ORV autograd 一致；块化只是把同样的式子按块跑。
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V  # noqa: E402


def softmax_masked(S, causal):
    n = S.shape[-1]
    if causal:
        mask = np.tril(np.ones((S.shape[-2], n), bool))
        S = np.where(mask, S, -np.inf)
    m = S.max(-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(S - m)
    e = np.where(np.isfinite(S), e, 0.0)
    return e / np.maximum(e.sum(-1, keepdims=True), 1e-20)


def flash_backward_numpy(q, k, v, dO, causal=True):
    """返回 (dO 对应的 dq, dk, dv)。q,k,v 形状 (B,H,S,D)。"""
    scale = 1.0 / np.sqrt(q.shape[-1])
    S = (q @ k.transpose(0, 1, 3, 2)) * scale
    P = softmax_masked(S, causal)
    dV = P.transpose(0, 1, 3, 2) @ dO
    dP = dO @ v.transpose(0, 1, 3, 2)
    dS = P * (dP - (dP * P).sum(-1, keepdims=True))
    dQ = (dS @ k) * scale
    dK = (dS.transpose(0, 1, 3, 2) @ q) * scale
    return dQ, dK, dV


def main():
    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    B, H, S, D = 1, 2, 12, 8
    q = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)
    k = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)
    v = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)
    dO = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)

    qt = T.tensor(q, requires_grad=True)
    kt = T.tensor(k, requires_grad=True)
    vt = T.tensor(v, requires_grad=True)

    # ORV 物化路径（causal=True）
    out = T.functional.scaled_dot_product_attention(qt, kt, vt, causal=True)
    (out * T.tensor(dO)).sum().backward()

    gq = np.asarray(qt.grad.to_numpy(), np.float32)
    gk = np.asarray(kt.grad.to_numpy(), np.float32)
    gv = np.asarray(vt.grad.to_numpy(), np.float32)

    dq, dk, dv = flash_backward_numpy(q, k, v, dO, causal=True)
    print("dq max_err=%.3e" % float(np.abs(gq - dq).max()))
    print("dk max_err=%.3e" % float(np.abs(gk - dk).max()))
    print("dv max_err=%.3e" % float(np.abs(gv - dv).max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
