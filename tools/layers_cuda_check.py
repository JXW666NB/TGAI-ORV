# -*- coding: utf-8 -*-
"""GPU 上验证补充层：池化与 BatchNorm 的 CUDA 与 CPU 一致，且与 torch 一致。

运行（需独占 GPU）：python layers_cuda_check.py
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
    from tgai_go.train.layers_extra import AvgPool2d, BatchNorm2d, Dropout, MaxPool2d

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (2, 3, 6, 6)).astype(np.float32)
    ok = True

    # 1) 池化：CUDA vs CPU vs torch
    for cls, tcls in ((MaxPool2d, torch.nn.MaxPool2d), (AvgPool2d, torch.nn.AvgPool2d)):
        m = cls(3, stride=2, padding=1)
        T.set_default_device("cuda")
        xc = T.tensor(x, requires_grad=True)
        yc = m(xc)
        T.set_default_device("cpu")
        ycpu = m(T.tensor(x))
        ref = tcls(3, stride=2, padding=1)(torch.from_numpy(x))
        r1 = rel(yc.to_numpy(), ycpu.to_numpy())
        r2 = rel(yc.to_numpy(), ref.detach().numpy())
        log("%s: CUDA-vs-CPU %.2e, CUDA-vs-torch %.2e" % (cls.__name__, r1, r2))
        ok = ok and max(r1, r2) <= 1e-6

        # CUDA 反向
        g = rng.normal(0, 1, tuple(ref.shape)).astype(np.float32)
        T.set_default_device("cuda")
        (yc * T.tensor(g)).sum().backward()
        tr = torch.from_numpy(x).requires_grad_(True)
        out = tcls(3, stride=2, padding=1)(tr)
        (out * torch.from_numpy(g)).sum().backward()
        r3 = rel(xc.grad.to_numpy(), tr.grad.numpy())
        log("  %s 反向: CUDA-vs-torch %.2e" % (cls.__name__, r3))
        ok = ok and r3 <= 1e-5

    # 2) BatchNorm2d：CUDA vs CPU（训练模式，含 running 统计）
    T.set_default_device("cpu")
    bn_cpu = BatchNorm2d(3); bn_cpu.train()
    y_cpu = bn_cpu(T.tensor(x))
    T.set_default_device("cuda")
    bn_gpu = BatchNorm2d(3); bn_gpu.train()
    y_gpu = bn_gpu(T.tensor(x))
    r = rel(y_gpu.to_numpy(), y_cpu.to_numpy())
    log("BatchNorm2d: CUDA-vs-CPU %.2e；running_mean 差 %.2e" %
        (r, rel(bn_gpu.running_mean, bn_cpu.running_mean)))
    ok = ok and r <= 1e-5

    # 3) Dropout CUDA：训练时均值≈1、eval 恒等
    T.set_default_device("cuda")
    d = Dropout(0.3); d.train()
    ov = d(T.tensor(np.ones((500, 500), np.float32))).to_numpy()
    d.eval()
    ev = d(T.tensor(np.ones((500, 500), np.float32))).to_numpy()
    log("Dropout CUDA: 训练均值 %.4f（≈1）零比例 %.3f（≈0.3）eval 恒等 %s"
        % (float(ov.mean()), float((ov == 0).mean()), bool(np.array_equal(ev, np.ones_like(ev)))))
    ok = ok and abs(float(ov.mean()) - 1.0) < 0.02 and bool(np.array_equal(ev, np.ones_like(ev)))

    T.set_default_device("cpu")
    log("总结论：%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
