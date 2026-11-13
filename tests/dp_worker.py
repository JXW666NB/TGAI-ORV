# -*- coding: utf-8 -*-
"""数据并行的单 rank 执行器（给 tests/test_dist_cpu.py 用子进程调起）。

为什么用独立进程：DataParallel 走的是「共享内存 + 文件屏障」，
多线程共享同一进程的文件描述符/信号量语义与真实多进程不同，测出来的东西不算数。

用法（内部）：
    python tests/dp_worker.py <rank> <world> <job> <outdir> [mode]

`mode`：
  * `basic`（默认）：广播 + 梯度平均
  * `missing`：**某个 rank 缺一层梯度**（MoE 专家没被路由到时就是这个情况）。
    旧实现在这里会**卡死**：各 rank 跳过的条目不同 ⇒ 字节总数不同 ⇒
    共享内存名字不同 ⇒ 一边创建、另一边永远等不到。修好后必须正常完成，
    且缺失方按 0 参与平均。
"""
import json
import os
import sys


def main():
    rank = int(sys.argv[1])
    world = int(sys.argv[2])
    job = sys.argv[3]
    outdir = sys.argv[4]
    mode = sys.argv[5] if len(sys.argv) > 5 else "basic"

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "python"))
    import numpy as np

    from tgai_go.train.dist import DataParallel

    class P:
        """CPU 形态的假参数（`.data` 是 ndarray，没有 `device` 属性 ⇒ 走 CPU 分支）。"""

        def __init__(self, arr):
            self.data = arr
            self.grad = None

        def numel(self):
            return int(self.data.size)

        @property
        def shape(self):
            return tuple(self.data.shape)

    dp = DataParallel(world=world, rank=rank, job=job, timeout=60.0)
    out = {"rank": rank, "mode": mode}

    if mode == "basic":
        # rank0 的初值非零，rank1 全零 —— broadcast 之后两边必须一样
        p = P(np.ones(3, dtype=np.float32) if rank == 0 else np.zeros(3, dtype=np.float32))
        dp.broadcast([p])
        out["data"] = [float(x) for x in p.data]
        # 梯度：rank0 = [1,1,1]，rank1 = [3,3,3] ⇒ 平均 = [2,2,2]
        p.grad = np.full(3, 1.0 if rank == 0 else 3.0, dtype=np.float32)
        dp.all_reduce_grads([p])
        g = p.grad.data if hasattr(p.grad, "data") else p.grad
        out["grad"] = [float(x) for x in g]

    elif mode == "missing":
        a = P(np.ones(3, dtype=np.float32) * (1.0 if rank == 0 else 2.0))
        b = P(np.ones(3, dtype=np.float32) * (5.0 if rank == 0 else 6.0))
        dp.broadcast([a, b])
        a.grad = np.full(3, 1.0 if rank == 0 else 3.0, dtype=np.float32)   # 两边都有
        # ★ rank1 故意不给 b 的梯度
        b.grad = np.full(3, 5.0, dtype=np.float32) if rank == 0 else None
        # ★ 让 rank0 晚一步进归约：把"快的一方先 close()"这个竞态**变成确定性复现**。
        #   旧实现里 close() 会删掉自己的 barrier 标记 ⇒ 慢的一方永远等不到，直接卡死。
        if rank == 0:
            import time
            time.sleep(0.5)
        dp.all_reduce_grads([a, b])
        ga = a.grad.data if hasattr(a.grad, "data") else a.grad
        gb = b.grad.data if hasattr(b.grad, "data") else b.grad
        out["a_data"] = [float(x) for x in a.data]
        out["b_data"] = [float(x) for x in b.data]
        out["a_grad"] = [float(x) for x in ga]
        out["b_grad"] = [float(x) for x in gb]
    else:
        raise SystemExit("未知 mode: %r" % mode)

    dp.close()
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "rank%d_%s.json" % (rank, mode)), "w",
              encoding="utf-8") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
