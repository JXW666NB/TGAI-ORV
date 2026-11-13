# -*- coding: utf-8 -*-
"""GPU 数据并行探针：**同一张卡上跑 2 个 rank**，验证广播/归约在 CUDA 路径上真的生效。

## 为什么要在真卡上跑

CPU 上用假参数测不出这两类问题（都实测过，2026-11-13）：

1. `broadcast()` 读的是 `p.data`，而 CUDA 张量的 `.data` 是 **None**，
   `np.ascontiguousarray(None)` 返回 `[nan]` 且**不报错** ⇒ 权重根本没广播；
2. 非 0 号 rank 的写回写进了**临时 numpy 数组**，与张量无关 ⇒ 同样静默无效。

这两个都只在"参数真的是 CUDA 张量"时暴露，所以必须有这一层。

## 判据（与单进程等效对拍）

两个 rank 各算一份梯度，平均之后必须等于"把两份样本合起来一个大 batch"的结果：

    参数 p：rank0 梯度 1，rank1 梯度 3  ⇒ 平均 2.0
    参数 q：rank0 梯度 5，rank1 **没有梯度**（模拟 MoE 没路由到）⇒ 平均 (5+0)/2 = 2.5

用法（内部）：
    python tools/dist_gpu_probe.py <rank> <world> <job> <outdir>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "python"))


def _host(g):
    """梯度可能是 ORV Tensor（CUDA/CPU）或 ndarray，统一取到主机端。"""
    if g is None:
        return []
    if hasattr(g, "to_numpy"):
        return np.asarray(g.to_numpy()).reshape(-1)
    return np.asarray(g).reshape(-1)


def main():
    rank = int(sys.argv[1])
    world = int(sys.argv[2])
    job = sys.argv[3]
    outdir = sys.argv[4]

    import tgai_go.train as T
    from tgai_go.train.dist import DataParallel

    T.set_default_device("cuda")

    # rank0 初值 1，rank1 初值 0 —— 广播后两边都必须是 1
    p = T.tensor(np.full(4, 1.0 if rank == 0 else 0.0, dtype=np.float32),
                 requires_grad=True)
    # q 两边初值相同，用来单独考察"缺梯度"的情形
    q = T.tensor(np.full(4, 7.0, dtype=np.float32), requires_grad=True)

    dp = DataParallel(world=world, rank=rank, job=job, timeout=120.0)
    dp.broadcast([p, q])

    p_after = [float(x) for x in p.to_numpy().reshape(-1)]
    q_after = [float(x) for x in q.to_numpy().reshape(-1)]

    # 本 rank 的"局部梯度"：p 是 1 或 3；q 只有 rank0 参与（rank1 不反传 q）
    c = 1.0 if rank == 0 else 3.0
    loss = (p * c).sum()
    if rank == 0:
        loss = loss + (q * 5.0).sum()
    loss.backward()
    # rank1 上 q.grad 应为 None（没参与前向）
    q_grad_local = None if rank == 1 else [float(x) for x in q.grad.to_numpy().reshape(-1)]

    dp.all_reduce_grads([p, q])

    out = {
        "rank": rank,
        "p_after": p_after,
        "q_after": q_after,
        "q_grad_local": q_grad_local,
        "p_grad": [float(x) for x in _host(p.grad)],
        "q_grad": [float(x) for x in _host(q.grad)],
        "p_device": str(getattr(p, "device", "?")),
        # ★ 缺梯度的一侧也必须拿到 **CUDA 张量**梯度（否则优化器会报
        #   "需要 CUDA Tensor 梯度" 或静默用错数据）
        "q_grad_is_tensor": hasattr(q.grad, "to_numpy"),
        "q_grad_device": str(getattr(q.grad, "device", "?")),
    }
    dp.close()
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "rank%d.json" % rank), "w", encoding="utf-8") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
