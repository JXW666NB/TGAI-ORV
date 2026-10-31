# -*- coding: utf-8 -*-
"""数据并行的单 rank 执行器（给 tests/test_dist_cpu.py 用子进程调起）。

为什么用独立进程：DataParallel 走的是「共享内存 + 文件屏障」，
多线程共享同一进程的文件描述符/信号量语义与真实多进程不同，测出来的东西不算数。

用法（内部）：
    python tests/dp_worker.py <rank> <world> <job> <outdir>
"""
import json
import os
import sys


def main():
    rank = int(sys.argv[1])
    world = int(sys.argv[2])
    job = sys.argv[3]
    outdir = sys.argv[4]

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "python"))
    from tgai_go.train.dist import DataParallel

    class P:
        def __init__(self, arr):
            self.data = arr
            self.grad = None

    dp = DataParallel(world=world, rank=rank, job=job, timeout=60.0)
    # rank0 的初值非零，rank1 全零 —— broadcast 之后两边必须一样
    p = P(1.0 + 0.0 * __import__("numpy").arange(3, dtype="float32") if rank == 0
          else __import__("numpy").zeros(3, dtype="float32"))
    dp.broadcast([p])
    data_after = [float(x) for x in p.data]

    # 梯度：rank0 = [1,1,1]，rank1 = [3,3,3] ⇒ 平均 = [2,2,2]
    import numpy as np
    p.grad = np.full(3, 1.0 if rank == 0 else 3.0, dtype="float32")
    dp.all_reduce_grads([p])
    grad_after = [float(x) for x in (p.grad.data if hasattr(p.grad, "data") else p.grad)]

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "rank%d.json" % rank), "w", encoding="utf-8") as f:
        json.dump({"rank": rank, "data": data_after, "grad": grad_after}, f)


if __name__ == "__main__":
    main()
