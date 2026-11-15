# -*- coding: utf-8 -*-
"""量清 ORV 惰性转置（_lazy_T）的影响面：哪些算子读到的是未转置的原始缓冲。"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402

CASES = []


def check(name, got, want):
    got = np.asarray(got, np.float32)
    want = np.asarray(want, np.float32)
    ok = got.shape == want.shape and np.allclose(got, want, atol=1e-6)
    CASES.append((name, ok, got, want))
    print("%-22s %s" % (name, "OK" if ok else "**错**"))
    if not ok:
        print("   got :", got.tolist())
        print("   want:", want.tolist())


def main():
    T.set_default_device("cuda")
    a = np.arange(6, dtype=np.float32).reshape(3, 2)
    A = np.arange(6, dtype=np.float32).reshape(3, 2)
    x = T.from_numpy(a)
    xt = x.T
    print("x.T to_numpy:", xt.to_numpy().tolist(), "（期望", A.T.tolist(), "）")

    check("to_numpy", xt.to_numpy(), A.T)
    check("reshape(6)", xt.reshape(6).to_numpy(), A.T.reshape(6))
    check("contiguous->numpy", xt.contiguous().to_numpy(), A.T)
    check("add scalar", (xt + 1.0).to_numpy(), A.T + 1.0)
    check("mul 2", (xt * 2.0).to_numpy(), A.T * 2.0)
    check("neg", (-xt).to_numpy(), -A.T)
    check("sum all", np.array([xt.sum().to_numpy()]), np.array([A.T.sum()]))
    check("mean all", np.array([xt.mean().to_numpy()]), np.array([A.T.mean()]))
    check("sum axis=1", xt.sum(axis=1).to_numpy(), A.T.sum(axis=1))
    check("exp", xt.exp().to_numpy(), np.exp(A.T))
    check("T.T", xt.T.to_numpy(), A)
    # 与另一个张量相加
    y = T.from_numpy(np.ones((2, 3), np.float32))
    check("add tensor", (xt + y).to_numpy(), A.T + np.ones((2, 3), np.float32))
    # matmul（文档说会走 gemm_nt）
    y2 = T.from_numpy(np.ones((3, 4), np.float32))
    check("matmul", T.matmul(xt, y2).to_numpy(), A.T @ np.ones((3, 4), np.float32))
    # 反向
    xg = T.from_numpy(a)
    xg.requires_grad = True
    (xg.T * xg.T).sum().backward()
    check("grad of x.T*x.T", xg.grad.to_numpy(), 2 * A)

    bad = [c for c in CASES if not c[1]]
    print("\n=== 共 %d 项，错 %d 项 ===" % (len(CASES), len(bad)))
    for nm, _ok, _g, _w in bad:
        print("  错:", nm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
