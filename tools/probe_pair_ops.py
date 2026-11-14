# -*- coding: utf-8 -*-
"""单块对（512×512，BH=8）逐算子计时：找出 11 ms 花在哪。"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402

NEG = -1e9


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    BH, R, C, D = 8, 512, 512, 64
    rng = np.random.default_rng(0)
    qb = T.tensor((rng.normal(0, .3, (BH, R, D))).astype(np.float32))
    kb = T.tensor((rng.normal(0, .3, (BH, C, D))).astype(np.float32))
    vb = T.tensor((rng.normal(0, .3, (BH, C, D))).astype(np.float32))
    db = T.tensor((rng.normal(0, .3, (BH, R, D))).astype(np.float32))
    lb = T.tensor(rng.normal(0, 1, (BH, R)).astype(np.float32))
    mask = T.tensor(np.where((np.arange(C)[None, :] - np.arange(R)[:, None]) <= 0,
                             0.0, NEG).astype(np.float32).reshape(1, R, C))
    acc = T.tensor(np.zeros((BH, R, D), np.float32))
    m = T.tensor(np.full((BH, R, 1), -1e9, np.float32))
    l = T.tensor(np.zeros((BH, R, 1), np.float32))

    def bench(name, fn, reps=20):
        fn(); rt.sync()
        t0 = time.time()
        for _ in range(reps):
            y = fn()
        rt.sync()
        dt = (time.time() - t0) / reps
        print("%-28s %8.3f ms" % (name, dt * 1e3))
        return y

    sb = bench("QK^T  (BH,R,C)", lambda: qb @ kb.transpose(1, 2))
    bench("+ 掩码（广播加）", lambda: sb + mask)
    bench("max(axis=-1)", lambda: sb.max(axis=-1))
    bench("exp（减行最大值）", lambda: (sb - lb.reshape(BH, R, 1)).exp())
    bench("行和 sum(-1)", lambda: sb.sum(axis=-1))
    bench("PV  (BH,R,C)@(BH,C,D)", lambda: sb @ vb)
    bench("acc*alpha + pv", lambda: acc * 0.5 + acc)
    bench("整链：QK+mask+softmax+PV", lambda: ((sb + mask).exp() @ vb))

    with T.no_grad():
        bench("同上(no_grad)", lambda: ((sb + mask).exp() @ vb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
