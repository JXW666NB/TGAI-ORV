# -*- coding: utf-8 -*-
"""对拍合并分派与逐槽分派：前向、梯度、耗时。"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_moe as MOE  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    N, D, E, DFF, K = 2048, 512, 8, 2048, 2
    rng = np.random.default_rng(0)
    x0 = rng.normal(0, .3, (N, D)).astype(np.float32)
    router = T.tensor(rng.normal(0, .3, (E, D)).astype(np.float32))
    W1 = [T.tensor(rng.normal(0, .05, (DFF, D)).astype(np.float32)) for _ in range(E)]
    W2 = [T.tensor(rng.normal(0, .05, (DFF, D)).astype(np.float32)) for _ in range(E)]
    W3 = [T.tensor(rng.normal(0, .05, (D, DFF)).astype(np.float32)) for _ in range(E)]

    res = {}
    for tag, merged in (("逐槽", False), ("合并", True)):
        rt.empty_cache()
        x = T.tensor(x0.copy(), requires_grad=True)
        eids, gates, _s = MOE.route_topk(x, router, K)
        t0 = time.time()
        y = MOE.moe_topk_orv(x, eids, gates, W1, W2, W3, merged=merged)
        y.to_numpy()
        rt.sync()
        fwd = time.time() - t0
        loss = (y * y).sum()
        loss.backward()
        rt.sync()
        tot = time.time() - t0
        res[tag] = (np.asarray(y.to_numpy(), np.float32),
                    np.asarray(x.grad.to_numpy(), np.float32), fwd, tot)
        del x, y, loss
        rt.empty_cache()

    y0, dx0, f0, t0_ = res["逐槽"]
    y1, dx1, f1, t1 = res["合并"]
    print("前向最大差 %.2e   dx 最大差 %.2e" % (float(np.abs(y0 - y1).max()),
                                              float(np.abs(dx0 - dx1).max())))
    print("前向耗时 逐槽 %.4fs → 合并 %.4fs" % (f0, f1))
    print("前+反向  逐槽 %.4fs → 合并 %.4fs（%.2fx）" % (t0_, t1, t0_ / max(t1, 1e-9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
