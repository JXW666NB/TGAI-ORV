# -*- coding: utf-8 -*-
"""GPU 上验证 0.9.16 沉到 C 端的算子：abs / relu / clamp / log（前向 + 反向 vs torch）。

运行（需独占 GPU）：python ewise_cuda_check.py
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
    return float(np.abs(a - b).max() / max(1e-9, float(np.abs(b).max())))


def main():
    import torch
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)
    xn = rng.normal(0, 1, (64, 32)).astype(np.float32)
    gw = rng.normal(0, 1, (64, 32)).astype(np.float32)
    ok = True

    cases = [
        ("abs", lambda t: t.abs(), lambda t: t.abs()),
        ("relu", lambda t: t.relu(), lambda t: torch.relu(t)),
        ("clamp(-0.5,0.7)", lambda t: t.clamp(-0.5, 0.7), lambda t: t.clamp(-0.5, 0.7)),
        ("log(abs+1)", lambda t: (t.abs() + 1.0).log(), lambda t: (t.abs() + 1.0).log()),
    ]
    for name, ours, theirs in cases:
        x = T.tensor(xn, requires_grad=True)
        y = ours(x)
        yn = np.asarray(y.to_numpy(), np.float32)
        (y * T.tensor(gw)).sum().backward()
        gx = np.asarray(x.grad.to_numpy(), np.float32)

        tx = torch.from_numpy(xn).requires_grad_(True)
        ty = theirs(tx)
        (ty * torch.from_numpy(gw)).sum().backward()
        r1 = rel(yn, ty.detach().numpy())
        r2 = rel(gx, tx.grad.numpy())
        log("%-16s fwd %.2e  bwd %.2e" % (name, r1, r2))
        ok = ok and r1 <= 1e-6 and r2 <= 1e-6

    T.set_default_device("cpu")
    log("总结论：%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
