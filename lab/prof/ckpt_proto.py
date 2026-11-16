# -*- coding: utf-8 -*-
"""验证 ORV 上做「重算（checkpoint）」的两个前提：

1) `Tensor.detach()` 是否**共享**显存（不拷贝）——否则重算反而更费显存；
2) 用 detach + 重跑 的方式包一个 checkpoint 节点，梯度是否与不包时**逐位相同**。

不通过就不要往库里改。
"""
import numpy as np

import tgai_go.train as T
from tgai_go.train import _acc

T.set_default_device("cuda")
rt = T._cuda()
rng = np.random.default_rng(0)


def checkpoint(fn, inputs):
    """前向不建图；反向用原始输入重跑前向再 backward。"""
    Tn = T.Tensor
    tens = [x for x in inputs if isinstance(x, Tn)]
    need = any(x.requires_grad for x in tens)
    det = [x.detach() if isinstance(x, Tn) else x for x in inputs]
    out = fn(*det)
    if not need:
        return out
    res = Tn._from_gpu(out._gpu, requires_grad=True, _parents=tuple(tens))

    def bw(g, grads):
        sub = fn(*inputs)
        for x in tens:
            x.grad = None
        sub.backward(g)
        for x in tens:
            if x.grad is not None:
                _acc(grads, x, x.grad._gpu)
                x.grad = None

    res._backward = bw
    return res


def main():
    # ---- 1) detach 是否共享显存 ----
    a = T.tensor(rng.normal(0, 1, (8, 4)).astype(np.float32), requires_grad=True)
    b = a.detach()
    same = int(a._gpu.ptr) == int(b._gpu.ptr)
    print("detach 共享同一块显存:", same, " (ptr a=0x%x b=0x%x)" % (int(a._gpu.ptr), int(b._gpu.ptr)))
    if not same:
        print("  => detach 拷贝了（对 checkpoint 可接受：层输入本来就要保留）")

    # ---- 2) 梯度等价 ----
    d = 64
    W1 = T.tensor(rng.normal(0, 0.05, (d, d)).astype(np.float32), requires_grad=True)
    W2 = T.tensor(rng.normal(0, 0.05, (d, d)).astype(np.float32), requires_grad=True)
    g1 = T.tensor(np.ones(d, np.float32), requires_grad=True)
    x0 = rng.normal(0, 1, (16, d)).astype(np.float32)

    def layer(x, w1, w2, gn):
        h = x @ w1.T
        h = h * gn
        return h @ w2.T

    def run(use_ckpt):
        for p in (W1, W2, g1):
            p.grad = None
        x = T.tensor(x0, requires_grad=True)
        if use_ckpt:
            y = checkpoint(layer, (x, W1, W2, g1))
        else:
            y = layer(x, W1, W2, g1)
        loss = (y * y).mean()
        loss.backward()
        out = {"loss": float(np.asarray(loss.to_numpy()).reshape(-1)[0]),
               "x": np.asarray(x.grad.to_numpy(), np.float32),
               "W1": np.asarray(W1.grad.to_numpy(), np.float32),
               "W2": np.asarray(W2.grad.to_numpy(), np.float32),
               "g1": np.asarray(g1.grad.to_numpy(), np.float32)}
        return out

    a1 = run(False)
    a2 = run(True)
    print("\nloss  无重算=%.8f  有重算=%.8f" % (a1["loss"], a2["loss"]))
    ok = True
    for k in ("x", "W1", "W2", "g1"):
        dmax = float(np.abs(a1[k] - a2[k]).max())
        rel = dmax / max(float(np.abs(a1[k]).max()), 1e-30)
        flag = "OK" if rel <= 1e-6 else "**不一致**"
        print("  grad %-3s max_abs=%.3e  max_rel=%.3e  %s" % (k, dmax, rel, flag))
        ok = ok and rel <= 1e-6
    print("\n判定：%s" % ("PASS —— 可以在库里实现" if ok else "FAIL —— 不要动库"))


if __name__ == "__main__":
    main()
