# -*- coding: utf-8 -*-
"""GPU 上验证归约算子：max/min/var/std/norm 的 CUDA 与 CPU 一致，max 反向与 torch 一致。"""
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
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (3, 4, 5)).astype(np.float32)
    ok = True

    for name, kw in (("max", {"axis": 1}), ("min", {"axis": 2}), ("var", {"axis": 0}),
                     ("std", {"axis": 1}), ("norm", {"axis": 2})):
        T.set_default_device("cpu")
        ref = np.asarray(getattr(T.tensor(x), name)(**kw).to_numpy(), np.float32)
        T.set_default_device("cuda")
        got = np.asarray(getattr(T.tensor(x), name)(**kw).to_numpy(), np.float32)
        T.set_default_device("cpu")
        r = rel(got, ref)
        log("%-5s CUDA-vs-CPU %.2e" % (name, r))
        ok = ok and r <= 1e-5

    # max 反向（CUDA）vs torch
    T.set_default_device("cuda")
    xt = T.tensor(x, requires_grad=True)
    y = xt.max(axis=1)
    g = rng.normal(0, 1, tuple(np.asarray(y.to_numpy()).shape)).astype(np.float32)
    (y * T.tensor(g)).sum().backward()
    got = np.asarray(xt.grad.to_numpy(), np.float32)
    tr = torch.from_numpy(x).requires_grad_(True)
    out = tr.max(dim=1).values
    (out * torch.from_numpy(g)).sum().backward()
    r = rel(got, tr.grad.numpy())
    log("max 反向 CUDA-vs-torch %.2e" % r)
    ok = ok and r <= 1e-6
    T.set_default_device("cpu")
    log("总结论：%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
