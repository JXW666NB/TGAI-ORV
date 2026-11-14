# -*- coding: utf-8 -*-
"""验收融合 RoPE：开关 fused_rope 前后，注意力的输出/梯度必须一致。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V  # noqa: E402


def run(x0, W, ln, d, n_q, n_kv, d_k, S, cos, sin):
    x = T.tensor(x0.copy(), requires_grad=True)
    y = V.attention_orv(V.rmsnorm_orv(x, ln), W["wq"], W["wk"], W["wv"], W["wo"],
                        cos, sin, d_k, n_q, n_kv)
    o = np.asarray(y.to_numpy(), np.float32)
    y.sum().backward()
    return o, np.asarray(x.grad.to_numpy(), np.float32)


def main():
    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    d, n_q, n_kv, S = 512, 8, 2, 512
    d_k = d // n_q
    W = {k: T.tensor(rng.normal(0, .02, s).astype(np.float32), requires_grad=True)
         for k, s in (("wq", (n_q * d_k, d)), ("wk", (n_kv * d_k, d)),
                      ("wv", (n_kv * d_k, d)), ("wo", (d, n_q * d_k)))}
    ln = T.tensor(rng.normal(1, .01, d).astype(np.float32), requires_grad=True)
    cos, sin = V.rope_cos_sin(S, d_k, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    x0 = rng.normal(0, .3, (1, S, d)).astype(np.float32)

    T.set_policy_value("fused_rope", False)
    o1, dx1 = run(x0, W, ln, d, n_q, n_kv, d_k, S, cos, sin)
    T.set_policy_value("fused_rope", True)
    o2, dx2 = run(x0, W, ln, d, n_q, n_kv, d_k, S, cos, sin)

    print("组合 RoPE  ：输出均值 %.6f" % float(o1.mean()))
    print("融合 RoPE  ：输出均值 %.6f" % float(o2.mean()))
    print("输出最大差 %.2e" % float(np.abs(o1 - o2).max()))
    print("dx  最大差 %.2e" % float(np.abs(dx1 - dx2).max()))
    ok = float(np.abs(o1 - o2).max()) < 1e-5 and float(np.abs(dx1 - dx2).max()) < 1e-4
    print("结论：%s" % ("一致（融合 RoPE 可用）" if ok else "不一致，须排查"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
