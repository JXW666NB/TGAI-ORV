# -*- coding: utf-8 -*-
"""上一层隔离：一次 matmul 的 fwd+bwd，amp 开 vs 关，梯度与 fp64 参考比。

目的：入口层（gemm_f16_*）已证明正确（max_rel ~3e-4）。
若本层错，则问题在 `_sgemm` 的 fp16 分发 / 权重影子缓存；
若本层对，则问题在模型的调用方式（多层的影子复用等）。
"""
import numpy as np

import tgai_go.train as T

rt = T._cuda()
T.set_default_device("cuda")
rng = np.random.default_rng(0)

M, K, N = 512, 1024, 2816


def one(amp_on):
    if amp_on:
        T.amp_on(dtype="fp16", weight_cache=True)
    else:
        T.amp_off(dtype="fp16")
    x_n = rng.normal(0, 1.0, (M, K)).astype(np.float32)
    w_n = rng.normal(0, 0.02, (N, K)).astype(np.float32)
    x = T.tensor(x_n.copy(), requires_grad=True)
    w = T.tensor(w_n.copy(), requires_grad=True)
    y = x @ w.T
    y.sum().backward()
    dx = np.asarray(x.grad.to_numpy(), np.float32)
    dw = np.asarray(w.grad.to_numpy(), np.float32)
    # fp64 参考
    x64, w64 = x_n.astype(np.float64), w_n.astype(np.float64)
    y64 = x64 @ w64.T
    dx64 = np.ones_like(y64) @ w64          # dL/dx = ones @ w
    dw64 = np.ones_like(y64).T @ x64        # dL/dw = ones.T @ x
    out = {}
    for name, got, ref in (("dx", dx, dx64), ("dw", dw, dw64)):
        d = float(np.abs(got - ref).max())
        s = max(float(np.abs(ref).max()), 1e-30)
        out[name] = d / s
    T.amp_off(dtype="fp16")
    rt.f16_cache_clear()
    return out


def main():
    print("=== matmul fwd+bwd 梯度对拍（参考 = fp64）===")
    off = one(False)
    on = one(True)
    print("amp_off: dx_rel=%.3e  dw_rel=%.3e" % (off["dx"], off["dw"]))
    print("amp_on : dx_rel=%.3e  dw_rel=%.3e" % (on["dx"], on["dw"]))
    print()
    for k in ("dx", "dw"):
        r = on[k]
        verdict = "OK(fp16 量级)" if r < 2e-3 else "**BAD**"
        print("%s: amp_on/amp_off 误差 = %.3e  %s" % (k, r, verdict))


if __name__ == "__main__":
    main()
