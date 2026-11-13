# -*- coding: utf-8 -*-
"""最小复现：no_grad 下反复做同样的小运算并丢弃结果，live 是否增长？

若增长 ⇒ 不是注意力的问题，是"临时张量在 no_grad 下也不被释放"的引擎问题。
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)
    a = T.tensor(rng.normal(size=(64, 64)).astype(np.float32))
    b = T.tensor(rng.normal(size=(64, 64)).astype(np.float32))
    rt.empty_cache()
    base = rt.pool_stats()["live_bytes"]
    print("基线 live=%.1f MiB" % (base / 2**20))

    with T.no_grad():
        for i in range(1, 1001):
            c = (a @ b) * 0.5 + a          # 3 个中间量，每个 16 KB
            y = c.exp()
            del c, y
            if i % 200 == 0:
                cur = rt.pool_stats()["live_bytes"]
                print("i=%-5d live=%.1f MiB（增量 %+.1f）"
                      % (i, cur / 2**20, (cur - base) / 2**20))
    final = rt.pool_stats()["live_bytes"]
    print("结束 live=%.1f MiB（增量 %+.1f）；若随迭代线性增长 ⇒ 临时张量未释放"
          % (final / 2**20, (final - base) / 2**20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
