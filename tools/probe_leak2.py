# -*- coding: utf-8 -*-
"""最小复现 v2：no_grad 下反复做与 flash 反向相同的**形状与算子组合**，live 是否增长。

形状全部取 (BH=8, 64, 64) 与 (8, 64, 64)，与 flash 反向里的分片一致。
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    BH, R, C, D = 8, 64, 64, 64
    rng = np.random.default_rng(0)
    mk = lambda *s: T.tensor(rng.normal(size=s).astype(np.float32) * 0.1)
    qb = mk(BH, R, D); kb = mk(BH, C, D); vb = mk(BH, C, D); db = mk(BH, R, D)
    lb = mk(BH, R)
    rt.empty_cache()
    base = rt.pool_stats()["live_bytes"]

    def pair():
        sb = (qb @ kb.transpose(1, 2))
        arr = np.zeros((1, R, C), np.float32)
        sb = sb + T.tensor(arr)
        p = (sb - lb.reshape(BH, R, 1)).exp()
        dp = db @ vb.transpose(1, 2)
        dsum = (dp * p).sum(axis=-1).reshape(BH, R, 1)
        ds = p * (dp - dsum)
        dq = (ds @ kb)
        dk = (ds.transpose(1, 2) @ qb)
        dv = (p.transpose(1, 2) @ db)
        return dq, dk, dv

    keep = []
    with T.no_grad():
        for i in range(1, 201):
            r = pair()
            keep.append(r[0])        # 只保留 dq，其余应被释放
            del r
            if i % 40 == 0:
                cur = rt.pool_stats()["live_bytes"]
                print("i=%-4d live=%.1f MiB（增量 %+.1f，保留 %d 个 dq）"
                      % (i, cur / 2**20, (cur - base) / 2**20, len(keep)))
    print("结束 live=%.1f MiB；若增量≈(#pair 中未保留的部分) ⇒ 有未释放的临时量"
          % (rt.pool_stats()["live_bytes"] / 2**20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
