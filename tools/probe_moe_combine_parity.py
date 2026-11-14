# -*- coding: utf-8 -*-
"""MoE combine 改动的数值对拍：把前向输出与输入梯度存盘，供两版对比。

用法（在两份代码下各跑一次，文件名不同）：
    python tools/probe_moe_combine_parity.py <输出前缀>

之后用 --compare 比两个前缀：
    python tools/probe_moe_combine_parity.py --compare A B
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402


def build(n=256, d=128, d_ff=64, n_exp=8, top_k=3, seed=0):
    T.set_default_device("cuda")
    # 固定全局随机源：两版代码用同一个种子构造，权重逐位相同。
    # （不要去改 p.data / 猜内部字段 —— 不同版本的存储方式可能不同，那样反而引入差异）
    np.random.seed(seed)
    try:
        T.manual_seed(seed)
    except Exception:
        pass
    moe = T.nn.MoE(d, d_ff, n_exp, top_k=top_k, n_shared=1, shared_d_ff=96)
    rng = np.random.default_rng(seed + 999)
    x = rng.normal(0, 0.05, (n, d)).astype(np.float32)
    return moe, x


def run(prefix):
    moe, x_arr = build()
    xt = T.tensor(x_arr, requires_grad=True)
    y = moe(xt)
    loss = (y * y).sum()
    loss.backward()
    rt = T._cuda()
    rt.sync()
    yv = np.asarray(y.to_numpy(), np.float32)
    gv = np.asarray(xt.grad.to_numpy(), np.float32)
    np.save("%s_y.npy" % prefix, yv)
    np.save("%s_g.npy" % prefix, gv)
    print("已保存 %s_y.npy / %s_g.npy  y 和 %.6f  grad 和 %.6f"
          % (prefix, prefix, float(yv.sum()), float(gv.sum())), flush=True)
    return 0


def compare(a, b):
    ok = True
    for tag in ("y", "g"):
        va = np.load("%s_%s.npy" % (a, tag))
        vb = np.load("%s_%s.npy" % (b, tag))
        if va.shape != vb.shape:
            print("%s 形状不一致：%s vs %s" % (tag, va.shape, vb.shape))
            ok = False
            continue
        d = float(np.abs(va - vb).max())
        scale = float(np.abs(va).max())
        rel = d / scale if scale else 0.0
        print("%s 最大差 %.3e（量级 %.3e，相对 %.2e）%s"
              % (tag, d, scale, rel, "  OK" if rel < 1e-6 else "  **超差**"),
              flush=True)
        ok = ok and rel < 1e-6
    return 0 if ok else 1


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--compare":
        return compare(sys.argv[2], sys.argv[3])
    prefix = sys.argv[1] if len(sys.argv) > 1 else "cur"
    return run(prefix)


if __name__ == "__main__":
    raise SystemExit(main())
