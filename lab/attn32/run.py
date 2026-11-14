# -*- coding: utf-8 -*-
"""隔离实验 runner：跑一组注意力数值并存盘，或比较两组。

用法：
    python lab/attn32/run.py --out D:/_lab_base
    python lab/attn32/run.py --cu lab/attn32/kernel.cu --out D:/_lab_copy
    python lab/attn32/run.py --compare D:/_lab_base D:/_lab_copy

要点：`--cu` 必须在 **import tgai_go.train 之前**设置环境变量，
因为内核源码路径在首次编译时就确定了。
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")


def compute():
    import tgai_go.train as T
    from tgai_go.train import functional as Fn

    T.set_default_device("cuda")
    BH, S, DH = 6, 2000, 64          # 2000 不是 32 的整数倍，顺带覆盖尾部边界
    rng = np.random.default_rng(0)
    q = T.tensor(rng.normal(0, .3, (BH, S, DH)).astype(np.float32), requires_grad=True)
    k = T.tensor(rng.normal(0, .3, (BH, S, DH)).astype(np.float32), requires_grad=True)
    v = T.tensor(rng.normal(0, .3, (BH, S, DH)).astype(np.float32), requires_grad=True)
    w = T.tensor(rng.normal(0, .3, (BH, S, DH)).astype(np.float32))
    y = Fn.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=True)
    (y * w).sum().backward()
    T._cuda().sync()
    return {
        "y": np.asarray(y.to_numpy(), np.float32),
        "dq": np.asarray(q.grad.to_numpy(), np.float32),
        "dk": np.asarray(k.grad.to_numpy(), np.float32),
        "dv": np.asarray(v.grad.to_numpy(), np.float32),
    }


def do_run(out, cu):
    if cu:
        os.environ["TGAI_ATTN_NOP_CU"] = os.path.abspath(cu)
        print("内核覆盖: %s" % os.environ["TGAI_ATTN_NOP_CU"], flush=True)
    res = compute()
    for kk, arr in res.items():
        np.save("%s_%s.npy" % (out, kk), arr)
    print("已保存 %s_*.npy  y 和 %.6f  dq 和 %.6f"
          % (out, float(res["y"].sum()), float(res["dq"].sum())), flush=True)
    return 0


def do_compare(a, b):
    ok = True
    for tag in ("y", "dq", "dk", "dv"):
        va = np.load("%s_%s.npy" % (a, tag))
        vb = np.load("%s_%s.npy" % (b, tag))
        d = float(np.abs(va - vb).max())
        scale = float(np.abs(va).max())
        rel = d / scale if scale else 0.0
        good = rel < 1e-5
        print("%-3s 最大差 %.3e（量级 %.3e，相对 %.2e）%s"
              % (tag, d, scale, rel, "  OK" if good else "  **超差**"), flush=True)
        ok = ok and good
    print("\n结论: %s" % ("一致" if ok else "**不一致**"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cu", default=None, help="实验内核路径（不填则用生产内核）")
    ap.add_argument("--out", default=None, help="输出前缀")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        return do_compare(args.compare[0], args.compare[1])
    if not args.out:
        ap.error("要么给 --out，要么给 --compare A B")
    return do_run(args.out, args.cu)


if __name__ == "__main__":
    raise SystemExit(main())
