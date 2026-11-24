# -*- coding: utf-8 -*-
"""b1：rmsnorm 模块单独基准（前向、反向、以及"同一形状不同批量"的伸缩）。

判据：GPU 时间应随元素数线性增长；若小形状上时间几乎不变（常数），
说明主机入队成本占主导，而不是 GPU 算得慢。
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")
sys.path.insert(0, "D:/orv_pip_0962/bench")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V  # noqa: E402
from btimer import Timer, rnd  # noqa: E402


def main():
    T.set_default_device("cuda")
    tm = Timer()
    d = 512
    print("=== rmsnorm_orv ===")
    for rows in (128, 512, 2048, 8192):
        x = T.tensor(rnd(rows, d, seed=rows))
        w = T.tensor(np.ones(d, np.float32))
        flops = 4 * rows * d          # 约：平方和 + 归一化 + 乘权重
        bytes_moved = 3 * rows * d * 4
        tm.report("前向 rows=%-5d" % rows,
                  lambda: V.rmsnorm_orv(x, w), flops=flops,
                  bytes_moved=bytes_moved)

    print("=== rmsnorm 反向 ===")
    for rows in (128, 2048):
        x = T.tensor(rnd(rows, d, seed=rows), requires_grad=True)
        w = T.tensor(np.ones(d, np.float32), requires_grad=True)

        def fwd_bwd():
            y = V.rmsnorm_orv(x, w)
            (y * y).sum().backward()
        tm.report("前+反向 rows=%-5d" % rows, fwd_bwd, reps=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
