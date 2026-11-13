# -*- coding: utf-8 -*-
"""top_k=2 MoE 验收：与 numpy 参考对拍（前向、选中下标、dx 有限差分）。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_moe as MOE  # noqa: E402

N, D, E, DFF, K = 8, 16, 4, 32, 2


def situ_np(x, w1, w2, w3, b1=4.0, b2=25.0):
    g = x @ w1.T
    u = x @ w2.T
    gate = (b1 * np.tanh(g / b1)) * (1.0 / (1.0 + np.exp(-g)))
    up = b2 * np.tanh(u / b2)
    return (gate * up) @ w3.T


def ref_moe(hn, router, W1, W2, W3, bias=None, k=K):
    z = hn @ router.T
    z = z - z.max(-1, keepdims=True)
    s = np.exp(z)
    s = s / s.sum(-1, keepdims=True)
    sel = s if bias is None else s + bias.reshape(1, -1)
    order = np.argsort(-sel, axis=-1, kind="stable")
    eids = order[:, :k]
    out = np.zeros_like(hn)
    for slot in range(k):
        for e in range(W1.shape[0]):
            pos = np.flatnonzero(eids[:, slot] == e)
            if pos.size == 0:
                continue
            out[pos] += situ_np(hn[pos], W1[e], W2[e], W3[e]) * s[pos, e].reshape(-1, 1)
    return out, eids, s


def main():
    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    hn0 = rng.normal(0, .3, (N, D)).astype(np.float32)
    router = rng.normal(0, .3, (E, D)).astype(np.float32)
    W1 = rng.normal(0, .2, (E, DFF, D)).astype(np.float32)
    W2 = rng.normal(0, .2, (E, DFF, D)).astype(np.float32)
    W3 = rng.normal(0, .2, (E, D, DFF)).astype(np.float32)
    wt = rng.normal(0, .3, (N, D)).astype(np.float32)

    ref, ref_eids, ref_s = ref_moe(hn0, router, W1, W2, W3)
    loss_ref = float((ref * wt).sum())

    # ORV 侧：权重用列表（与训练脚本一致）
    router_t = T.tensor(router)
    Wl = [[T.tensor(W1[e]), T.tensor(W2[e]), T.tensor(W3[e])] for e in range(E)]
    W1l = [w[0] for w in Wl]; W2l = [w[1] for w in Wl]; W3l = [w[2] for w in Wl]
    wt_t = T.tensor(wt)

    def fwd(hn_np):
        hn_t = T.tensor(hn_np, requires_grad=True)
        eids, gates, s_all = MOE.route_topk(hn_t, router_t, K)
        out = MOE.moe_topk_orv(hn_t, eids, gates, W1l, W2l, W3l)
        return hn_t, eids, gates, s_all, out

    hn_t, eids, gates, s_all, out = fwd(hn0)
    got = np.asarray(out.to_numpy(), np.float32)
    print("选中下标与参考一致：", bool(np.array_equal(np.sort(eids, 1), np.sort(ref_eids, 1))))
    print("softmax 与参考最大差：%.2e" % np.abs(np.asarray(s_all.to_numpy()) - ref_s).max())
    print("前向最大差：%.2e" % np.abs(got - ref).max())

    # 反向：解析梯度 vs 有限差分
    loss = (out * wt_t).sum()
    loss.backward()
    dx = np.asarray(hn_t.grad.to_numpy(), np.float32)
    eps = 1e-3
    err = 0.0
    for (i, j) in ((0, 0), (1, 3), (5, 7)):
        hp = hn0.copy(); hp[i, j] += eps
        hm = hn0.copy(); hm[i, j] -= eps
        lp = float((ref_moe(hp, router, W1, W2, W3)[0] * wt).sum())
        lm = float((ref_moe(hm, router, W1, W2, W3)[0] * wt).sum())
        num = (lp - lm) / (2 * eps)
        err = max(err, abs(num - dx[i, j]))
        print("  d[%d,%d] 解析=%.6f 数值=%.6f" % (i, j, dx[i, j], num))
    print("dx 最大误差：%.2e（阈值 1e-3）" % err)
    return 0 if err < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
