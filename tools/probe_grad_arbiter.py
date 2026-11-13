# -*- coding: utf-8 -*-
"""用有限差分判定：我的 flash 反向与 ORV autograd 谁对。

小规模、无掩码：loss = sum(O * dO)，对 q 逐元素扰动求数值梯度。
若数值梯度站在**我的公式**一边，则 ORV 的 SDPA 反向有问题（重要发现）。
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")

import tgai_go.train as T  # noqa: E402

NEG = -1e9


def softmax_rows(S):
    m = S.max(-1, keepdims=True)
    e = np.exp(S - m)
    return e / e.sum(-1, keepdims=True)


def fwd(q, k, v):
    D = q.shape[-1]
    P = softmax_rows((q @ k.swapaxes(-1, -2)) * (1.0 / np.sqrt(D)))
    return P @ v


def my_dq(q, k, v, dO):
    D = q.shape[-1]
    scale = 1.0 / np.sqrt(D)
    P = softmax_rows((q @ k.swapaxes(-1, -2)) * scale)
    dP = dO @ v.swapaxes(-1, -2)
    dS = P * (dP - (dP * P).sum(-1, keepdims=True))
    return (dS @ k) * scale


def main():
    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    B, H, S, D = 1, 1, 4, 3
    q = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)
    k = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)
    v = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)
    dO = rng.normal(0, .3, (B, H, S, D)).astype(np.float32)

    # ① ORV autograd 的 dq
    qt = T.tensor(q, requires_grad=True)
    kt = T.tensor(k)
    vt = T.tensor(v)
    out = T.functional.scaled_dot_product_attention(qt, kt, vt, causal=False)
    (out * T.tensor(dO)).sum().backward()
    gq_orv = np.asarray(qt.grad.to_numpy(), np.float32)

    # ② 我的公式
    gq_mine = my_dq(q, k, v, dO)

    # ③ 有限差分（中心差分，eps=1e-3 在 fp32 上更稳）
    eps = 1e-3
    gq_fd = np.zeros_like(q)
    for i in range(S):
        for j in range(D):
            qp = q.copy(); qp[0, 0, i, j] += eps
            qm = q.copy(); qm[0, 0, i, j] -= eps
            lp = float((fwd(qp, k, v) * dO).sum())
            lm = float((fwd(qm, k, v) * dO).sum())
            gq_fd[0, 0, i, j] = (lp - lm) / (2 * eps)

    print("ORV vs 有限差分 : %.3e" % float(np.abs(gq_orv - gq_fd).max()))
    print("我的 vs 有限差分: %.3e" % float(np.abs(gq_mine - gq_fd).max()))
    print("我的 vs ORV     : %.3e" % float(np.abs(gq_mine - gq_orv).max()))
    print("样例 ORV ", gq_orv.ravel()[:4])
    print("样例 我的", gq_mine.ravel()[:4])
    print("样例 FD  ", gq_fd.ravel()[:4])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
