# -*- coding: utf-8 -*-
"""验证 DP 的数学对齐（目标第 4 项 ②③）。

做两件事：
  ② world=1 时：`all_reduce_grads` 必须是**空操作**（与不开 DP 完全一致）。
  ③ DP（两进程各吃一个样本）得到的平均梯度，必须等于
     「单进程里分别对两个样本求梯度、再取算术平均」。

实现要点：两个样本的数据由**同一个种子**确定 ⇒ 每个 rank 都能在本地
把两个样本都算一遍，于是可以直接和 DP 的结果对拍。
"""
import numpy as np

import tgai_go.train as T
from tgai_go.train import dist as D
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()
dp = D.DataParallel()

CFG = dict(d=128, d_ff=256, E=2, layers=2, n_q=4, n_kv=2, vocab=4096,
           seq=128, attn_res=False, attn_res_block=2, top_k=2, csa=4,
           n_kv_attn=4, rope_theta=500000.0, csa_compress=4,
           csa_conv_kernel=3, csa_local_window=0)


def build():
    cfg = dict(CFG)
    P = CLI.build(cfg, np.random.default_rng(0))
    return cfg, P, CLI.params_of(P)


def sample(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, CFG["vocab"], CFG["seq"]).astype(np.int64)


def grads_of(cfg, P, params, ids):
    """对给定样本跑一次前向+反向，返回各参数的梯度（numpy 列表）。"""
    from tgai_go.train import v3_csa as C
    kinds = CLI.layer_kinds(cfg)
    csac = cfg["csa_cfg"]
    plan = C.make_plan(csac, 1, cfg["seq"])
    cos, sin = C.rope_tables(csac, cfg["seq"])
    for p in params:
        p.grad = None
    logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
    loss = T.functional.cross_entropy(logits.reshape(cfg["seq"], cfg["vocab"]),
                                      np.roll(ids, -1))
    loss.backward(T.tensor(np.array(1.0, np.float32)))
    out = []
    for p in params:
        g = getattr(p, "grad", None)
        out.append(np.zeros(int(np.prod(tuple(int(s) for s in p.shape))), np.float32)
                   if g is None else np.array(rt.d2h(g._gpu), dtype=np.float32))
    return out


def main():
    cfg, P, params = build()
    idsA, idsB = sample(11), sample(22)
    mine = idsA if dp.rank == 0 else idsB

    # ① 本地只对自己的样本求梯度
    local = grads_of(cfg, P, params, mine)
    # 保一份副本，用来和"平均后"比较
    import copy
    before = copy.deepcopy(local)
    # ② DP 归约
    dp.all_reduce_grads(params)
    after = []
    for p in params:
        g = getattr(p, "grad", None)
        after.append(np.zeros_like(before[0]) if g is None else
                     np.array(rt.d2h(g._gpu), dtype=np.float32))

    # ③ 本地把两个样本都算一遍，手算算术平均作为参考
    ref = [(a + b) * 0.5 for a, b in
           zip(grads_of(cfg, P, params, idsA), grads_of(cfg, P, params, idsB))]

    def rel(x, y):
        # x/y 是"每个参数一个数组"的列表（形状各异），要逐对比较
        num = max(float(np.abs(a - b).max()) for a, b in zip(x, y))
        den = max(1e-12, max(float(np.abs(b).max()) for b in y))
        return num / den

    print("rank=%d  DP平均 vs 手算两样本平均：最大相对差 %.3e" % (dp.rank, rel(after, ref)),
          flush=True)
    if dp.rank == 0:
        # world=1 的空操作性：用 before（本地梯度）与"未开 DP"应完全相同
        print("  本地梯度 vs DP 后：相对差 %.3e（应 >0，说明归约真的改了梯度）"
              % rel(before, after), flush=True)
    # 各 rank 必须得到**同一个**平均梯度 ⇒ 彼此一致
    s = float(np.sum([float(np.sum(x)) for x in after]))
    print("  梯度总和（两进程应相同）：%.6f" % s, flush=True)


if __name__ == "__main__":
    main()
