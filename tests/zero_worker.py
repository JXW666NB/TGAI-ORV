# -*- coding: utf-8 -*-
"""ZeRO 的 **2 进程** CPU 对拍 worker（由 ``tests/test_zero_transport.py`` 拉起）。

为什么要真起两个进程：``all_gather`` / ``reduce_scatter`` / 共享内存槽 / 文件屏障
这些东西**单进程测不出来** —— 名字碰撞、barrier 配对、分段覆盖只有在两个真实进程
里才会暴露。这里用真的 :class:`tgai_go.train.dist.DataParallel`（host 后端）+
假运行时（``zero_fake.FakeRT``，numpy 实现设备缓冲与内核语义），因此**不需要显卡**。

用法（两个进程必须用同一个 ``job``/``TGAI_DP_DIR``）::

    python tests/zero_worker.py <rank> <world> <job>

输出：最后一行 ``ZERO_WORKER_JSON {...}``。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, HERE)

from tgai_go.train import dist as D          # noqa: E402
from tgai_go.train import zero as Z         # noqa: E402
from zero_fake import FakeRT, NBuf, NParam, reference_ssq_total   # noqa: E402

#: 形状固定（两个 rank 必须完全一致；含一个「缺梯度」的参数用来验证补 0 路径）
SHAPES = [7, 5, 9, 4]
MISSING_ON_RANK0 = 2


def _base_values(shape_list):
    rng = np.random.default_rng(20261121)
    return [rng.normal(0.0, 1.0, (n,)).astype(np.float32) for n in shape_list]


def _make_grad(r, shape_list):
    """rank 自己的「本地梯度」：所有 rank 都有，除了 rank0 故意缺第 2 个。"""
    base = _base_values(shape_list)
    out = []
    for i, g in enumerate(base):
        if r == 0 and i == MISSING_ON_RANK0:
            out.append(None)
        else:
            out.append((np.float32(r + 1.0) * g).astype(np.float32))
    return out


def _expected_reduced(world, shape_list):
    """跨 rank 求平均（与 dist 的 host all_reduce 同序：按 rank 求和后乘 1/world）。"""
    grads = [_make_grad(r, shape_list) for r in range(world)]
    out = []
    for i in range(len(shape_list)):
        cols = []
        for r in range(world):
            g = grads[r][i]
            cols.append(np.zeros(shape_list[i], np.float32) if g is None else g)
        acc = cols[0].copy()
        for c in cols[1:]:
            acc = np.float32(acc + c)
        acc = np.float32(acc * np.float32(1.0 / world))
        out.append(acc)
    return out


def main(rank, world, job):
    shapes = list(SHAPES)
    rt = FakeRT()
    if os.environ.get("ZERO_TEST_NOREG"):
        # 模拟「cuMemHostRegister 失败/老运行时没有 host_register」的退化路径：
        # 传输必须退回 pinned + copyto，且结果逐位不变。
        rt.host_register = None
    rng = np.random.default_rng(99)
    base = _base_values(shapes)
    params = [NParam(base[i].copy()) for i in range(len(shapes))]
    for i, p in enumerate(params):
        g = _make_grad(rank, shapes)[i]
        if g is not None:
            p.set_grad(g)

    dp = D.DataParallel(world=world, rank=rank, job=job, timeout=30,
                        backend="host")
    zt = Z.ZeroTrainer(params, zero=2, world=world, rank=rank, rt=rt)
    zt.attach(dp)

    t_rs = zt.reduce_grads()
    exp = _expected_reduced(world, shapes)

    # ① 本 rank 拥有的梯度必须等于「跨 rank 平均」的对应切片（逐位）
    owned_ok = True
    owned_maxdiff = 0.0
    for i in range(zt.plan.i0, zt.plan.i1):
        got = np.asarray(params[i].grad._gpu.np, np.float32)
        want = exp[i]
        owned_ok = owned_ok and np.array_equal(got, want)
        if got.size:
            owned_maxdiff = max(owned_maxdiff,
                                float(np.max(np.abs(got - want))))

    # ② 非本 rank 的梯度必须**还留着**（gnorm 要按参考顺序枚举全部梯度）
    kept_unowned = all(params[i].grad is not None
                       for i in range(len(params))
                       if not (zt.plan.i0 <= i < zt.plan.i1))

    # ③ 分布式 gnorm 必须与「不分片参考路径」逐位相同
    gnorm, gscale = zt.grad_norm_scale(5.0, defer=True)
    ref_total = reference_ssq_total([NBuf(e) for e in exp])
    ref_norm = float(np.sqrt(max(ref_total, 0.0)))
    ref_scale = (float(5.0 / (ref_norm + 1e-6)) if ref_norm > 5.0 else 1.0)
    gnorm_exact = (gnorm == ref_norm) and (gscale == ref_scale)

    # ④ defer 路径算完范数后，非本 rank 的梯度必须被释放（显存收益兑现）
    freed_after_norm = all(params[i].grad is None
                           for i in range(len(params))
                           if not (zt.plan.i0 <= i < zt.plan.i1))

    # ⑤ all_gather：每个 rank 只更新自己的切片，收回后全卡逐位一致
    owners = {}
    for r, (a, b) in enumerate(Z.plan_ranges(shapes, world)):
        for i in range(a, b):
            owners[i] = r
    for i in range(len(params)):
        if owners[i] == rank:
            v = np.float32((rank + 1) * 100.0) + np.float32(i)
            params[i]._gpu.np[:] = v     # 假装优化器只更新了自己拥有的参数
    t_g = zt.after_step()
    gather_ok = True
    for i in range(len(params)):
        own = owners[i]
        want = np.full(shapes[i], np.float32((own + 1) * 100.0) + np.float32(i),
                       np.float32)
        got = np.asarray(params[i]._gpu.np, np.float32)
        gather_ok = gather_ok and np.array_equal(got, want)

    st = zt.stats.as_dict()
    out = {"rank": int(rank), "world": int(world),
           "n_params": len(params), "n_owned": len(zt.plan.owned),
           "owned_ok": bool(owned_ok), "owned_maxdiff": owned_maxdiff,
           "kept_unowned_before_norm": bool(kept_unowned),
           "gnorm": float(gnorm), "gscale": float(gscale),
           "ref_norm": float(ref_norm), "ref_scale": float(ref_scale),
           "gnorm_exact": bool(gnorm_exact),
           "freed_after_norm": bool(freed_after_norm),
           "gather_ok": bool(gather_ok),
           "rs_sec": t_rs, "gather_sec": t_g,
           "stats": st}
    zt.close()
    dp.close()
    print("ZERO_WORKER_JSON " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    rank = int(sys.argv[1])
    world = int(sys.argv[2])
    job = sys.argv[3]
    sys.exit(main(rank, world, job))
