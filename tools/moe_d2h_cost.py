# -*- coding: utf-8 -*-
"""量化 MoE 那条 D2H 的代价：`moe_compact` + 9 个 int32 回主机，每次多少钱？

不需要加载模型（用随机索引合成），几秒钟即可。
运行：python moe_d2h_cost.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")


def main():
    import torch
    import tgai_go.train as T

    T.set_default_device("cuda")
    rt = T._cuda()
    n, k, n_exp = 128, 2, 8
    rng = np.random.default_rng(0)
    # 每个 token 选 k 个不同专家
    idx = np.stack([rng.choice(n_exp, k, replace=False) for _ in range(n)]).astype(np.int32)
    idx_buf = rt.h2d(np.ascontiguousarray(idx), dtype=np.int32)

    def sync():
        torch.cuda.synchronize()

    def with_d2h():
        off_buf, tok_buf, slot_buf = rt.moe_compact(idx_buf, n, n_exp, k)
        return np.asarray(rt.d2h(off_buf), dtype=np.int32).reshape(-1)

    def without_d2h():
        return rt.moe_compact(idx_buf, n, n_exp, k)

    with_d2h(); sync()
    t0 = time.time()
    for _ in range(500):
        with_d2h()
    sync()
    t_with = (time.time() - t0) / 500

    without_d2h(); sync()
    t0 = time.time()
    for _ in range(500):
        without_d2h()
    sync()
    t_without = (time.time() - t0) / 500

    off = with_d2h()
    print("epoch 偏移量: %s" % off.tolist())
    print("moe_compact + d2h : %.3f ms/次" % (t_with * 1000))
    print("仅 moe_compact    : %.3f ms/次" % (t_without * 1000))
    print("=> D2H 同步自身约 %.3f ms/次；28 层/步 ≈ %.1f ms/步"
          % ((t_with - t_without) * 1000, (t_with - t_without) * 28 * 1000))


if __name__ == "__main__":
    main()
