# -*- coding: utf-8 -*-
"""dh=64 flash 路径的参考输出抓取 / 对拍。

用途：把 `tgai_attn_nop.cu` 改成"DH 可配置"之前，先记下现在（dh=64）的
前向输出与 q/k/v 梯度；改完之后逐位比对，确保既有路径**一点没变**。

    python attn_ref.py --save      # 抓参考，写 attn_ref.npz
    python attn_ref.py --check     # 对拍，报告最大绝对差

判据：`--check` 报出的最大绝对差必须 **恰好 0.0**（同一条代码路径、同一张卡）。
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "attn_ref.npz")

#: 覆盖若干形状（BH / S / DH 固定 64），包含 S 不是 tile 整数倍的情形
CASES = [
    (2, 64, 64),
    (2, 128, 64),
    (4, 100, 64),     # S 非 2 的幂 / 非 tile 倍数，考边界块
    (1, 257, 64),     # 再一个奇数长度
]


def _build(bh, seq, dh, seed=0):
    import tgai_go.train as T
    rng = np.random.default_rng(seed)

    def dev(a):
        return T.Tensor._from_gpu(T._cuda().h2d(np.ascontiguousarray(a, np.float32)))

    q = dev(rng.normal(0, 0.5, (bh, seq, dh)))
    k = dev(rng.normal(0, 0.5, (bh, seq, dh)))
    v = dev(rng.normal(0, 0.5, (bh, seq, dh)))
    for t in (q, k, v):
        t.requires_grad = True
    return q, k, v


def run_all():
    import tgai_go.train as T
    from tgai_go.train import functional as Fn

    T.set_default_device("cuda")
    out = {}
    for bh, seq, dh in CASES:
        q, k, v = _build(bh, seq, dh)
        ok = T._sdpa_lowmem_ok(q, k, v, True, dh)
        assert ok, "这个形状没走 flash 路径（bh=%d seq=%d dh=%d）" % (bh, seq, dh)
        o = Fn.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=True)
        g = T.Tensor._from_gpu(
            T._cuda().h2d(np.ones((bh, seq, dh), np.float32)))
        o.backward(g)
        tag = "%d_%d_%d" % (bh, seq, dh)
        out[tag + "_o"] = np.asarray(o.to_numpy(), np.float32)
        for nm, t in (("q", q), ("k", k), ("v", v)):
            gg = t.grad
            arr = gg.to_numpy() if hasattr(gg, "to_numpy") else gg
            out[tag + "_d" + nm] = np.asarray(arr, np.float32)
    return out


def main():
    mode = "--check" if "--check" in sys.argv else "--save"
    got = run_all()
    if mode == "--save":
        np.savez(NPZ, **got)
        print("已抓参考：%d 个数组 -> %s" % (len(got), NPZ))
        return 0

    if not os.path.isfile(NPZ):
        print("没有参考文件，先跑 --save")
        return 1
    ref = np.load(NPZ)
    worst = 0.0
    worst_key = ""
    miss = []
    for key in ref.files:
        if key not in got:
            miss.append(key)
            continue
        d = float(np.max(np.abs(got[key] - ref[key])))
        if d > worst:
            worst, worst_key = d, key
        if not np.array_equal(got[key], ref[key]):
            print("  不一致: %-16s 最大绝对差 %.3e" % (key, d))
    print("数组数 %d｜最大绝对差 %.3e（%s）"
          % (len(ref.files), worst, worst_key or "-"))
    if miss:
        print("缺失键:", miss)
    ok = (worst == 0.0) and not miss
    print("结论:", "逐位一致 PASS" if ok else "**有差异 FAIL**")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
