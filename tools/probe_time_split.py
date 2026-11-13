# -*- coding: utf-8 -*-
"""分解一层的时间：rmsnorm / attention / MoE / 残差（前向+反向，S=2048）。"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    N, d, dff, E = 2048, 512, 2048, 8
    rng = np.random.default_rng(0)
    W = lambda *s: T.tensor(rng.normal(0, .02, s).astype(np.float32), requires_grad=True)
    ln1, ln2 = W(d), W(d)
    wq, wk, wv, wo = W(d, d), W(128, d), W(128, d), W(d, d)
    router = W(E, d)
    W1, W2, W3 = [W(dff, d) for _ in range(E)], [W(dff, d) for _ in range(E)], [W(d, dff) for _ in range(E)]
    cos, sin = V.rope_cos_sin(N, 64, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)

    def bench(name, fn, reps=3):
        xx = T.tensor(rng.normal(0, .02, (1, N, d)).astype(np.float32), requires_grad=True)
        y = fn(xx); y.sum().backward()
        del xx, y
        rt.empty_cache()
        t0 = time.time()
        for _ in range(reps):
            x = T.tensor(rng.normal(0, .02, (1, N, d)).astype(np.float32), requires_grad=True)
            y = fn(x)
            y.sum().backward()
            del x, y
        dt = (time.time() - t0) / reps
        print("%-16s %7.3f s（前向+反向）" % (name, dt))

    bench("rmsnorm", lambda x: V.rmsnorm_orv(x, ln1))
    bench("attention", lambda x: V.attention_orv(V.rmsnorm_orv(x, ln1), wq, wk, wv, wo,
                                                 cos, sin, 64, 8, 2, flash=True))
    bench("MoE(含路由)", lambda x: MOE.moe_dispatch_orv(
        V.rmsnorm_orv(x, ln2).reshape(N, d), *MOE.route_top1(
            V.rmsnorm_orv(x, ln2).reshape(N, d), router)[:2], W1, W2, W3))
    bench("matmul 512x2048", lambda x: x.reshape(N, d) @ W1[0].transpose(0, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
