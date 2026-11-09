# -*- coding: utf-8 -*-
"""LayerNorm 的 CUDA 路径实测（纯组合实现，依赖 mean/sqrt 的 CUDA 内核）。

运行（需独占 GPU）：python layernorm_gpu_check.py
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log


def rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def main():
    import torch
    import tgai_go.train as T

    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    D = 32
    x = rng.normal(0, 1, (4, D)).astype(np.float32)
    w = rng.normal(1, 0.1, (D,)).astype(np.float32)
    b = rng.normal(0, 0.1, (D,)).astype(np.float32)

    ln = T.nn.LayerNorm(D, eps=1e-5)
    np.copyto(np.asarray(ln.weight.to_numpy(), np.float32), w) if False else None
    # 参数在 CUDA 上：用 h2d 覆盖
    rt = T._cuda()
    ln.weight._gpu = rt.h2d(w)
    ln.bias._gpu = rt.h2d(b)

    xt = T.tensor(x.copy(), requires_grad=True)
    y = ln(xt)
    y.sum().backward()
    got = np.asarray(y.to_numpy(), np.float32)
    gx = np.asarray(xt.grad.to_numpy(), np.float32)

    ref = torch.nn.LayerNorm(D, eps=1e-5).cuda()
    with torch.no_grad():
        ref.weight.copy_(torch.from_numpy(w).cuda())
        ref.bias.copy_(torch.from_numpy(b).cuda())
    xr = torch.from_numpy(x.copy()).cuda().requires_grad_(True)
    yr = ref(xr)
    yr.sum().backward()

    r1 = rel(got, yr.detach().cpu().numpy())
    r2 = rel(gx, xr.grad.cpu().numpy())
    log("LayerNorm CUDA：前向 相对=%.2e  反向 相对=%.2e ⇒ %s"
        % (r1, r2, "PASS" if (r1 <= 1e-5 and r2 <= 1e-5) else "FAIL"))


if __name__ == "__main__":
    main()
