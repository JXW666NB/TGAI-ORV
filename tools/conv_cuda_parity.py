# -*- coding: utf-8 -*-
"""GPU conv2d 与 torch 对照：前向 + dx/dw/db（判据 1e-5）。

运行（需独占 GPU）：python conv_cuda_parity.py
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

CFGS = [
    # (N,C,H,W, kh,kw, stride, padding, dilation)
    (2, 3, 5, 6, 3, 3, 1, 0, 1),
    (1, 2, 6, 6, 3, 3, 1, 1, 1),
    (2, 2, 7, 7, 3, 3, 2, 1, 1),
    (1, 3, 8, 5, 5, 3, 1, 2, 1),
    (1, 1, 9, 9, 3, 3, 1, 2, 2),
    (2, 4, 5, 5, 2, 2, 2, 1, 1),
]
COUT = 4


def rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(1e-9, float(np.abs(b).max())))


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.train.conv import conv2d

    T.set_default_device("cuda")
    rt = T._cuda()
    worst = 0.0
    for idx, (N, C, H, W, kh, kw, s, p, d) in enumerate(CFGS):
        rng = np.random.default_rng(idx)
        xn = rng.normal(0, 1, (N, C, H, W)).astype(np.float32)
        wn = rng.normal(0, 0.2, (COUT, C, kh, kw)).astype(np.float32)
        bn = rng.normal(0, 0.1, (COUT,)).astype(np.float32)

        x = T.tensor(xn, requires_grad=True)
        w = T.tensor(wn, requires_grad=True)
        b = T.tensor(bn, requires_grad=True)
        y = conv2d(x, w, b, stride=s, padding=p, dilation=d)
        yn = np.asarray(y.to_numpy(), np.float32)
        oh, ow = yn.shape[2], yn.shape[3]
        gw = rng.normal(0, 1, (N, COUT, oh, ow)).astype(np.float32)
        (y * T.tensor(gw)).sum().backward()
        rt.sync()
        dxn = np.asarray(x.grad.to_numpy(), np.float32)
        dwn = np.asarray(w.grad.to_numpy(), np.float32)
        dbn = np.asarray(b.grad.to_numpy(), np.float32)

        tx = torch.from_numpy(xn).requires_grad_(True)
        tw = torch.from_numpy(wn).requires_grad_(True)
        tb = torch.from_numpy(bn).requires_grad_(True)
        ty = torch.nn.functional.conv2d(tx, tw, tb, stride=s, padding=p, dilation=d)
        ty.backward(torch.from_numpy(gw))

        r = [rel(yn, ty.detach().numpy()), rel(dxn, tx.grad.numpy()),
             rel(dwn, tw.grad.numpy()), rel(dbn, tb.grad.numpy())]
        worst = max(worst, max(r))
        log("cfg%d N=%d C=%d %dx%d k=%dx%d s=%d p=%d d=%d | fwd %.2e dx %.2e dw %.2e db %.2e"
            % (idx, N, C, H, W, kh, kw, s, p, d, r[0], r[1], r[2], r[3]))
    log("最差 %.2e（判据 1e-5）⇒ %s" % (worst, "PASS" if worst <= 1e-5 else "FAIL"))
    return 0 if worst <= 1e-5 else 1


if __name__ == "__main__":
    sys.exit(main())
