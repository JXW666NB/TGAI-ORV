# -*- coding: utf-8 -*-
"""GPU 上验证组卷积（groups=2 / 4）：与 torch 对照前向 + dx/dw/db。

运行（需独占 GPU）：python conv_groups_cuda.py
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
    from tgai_go.train.conv import conv2d

    T.set_default_device("cuda")
    rt = T._cuda()
    ok = True
    for groups in (2, 4):
        for s, p, d in ((1, 1, 1), (2, 1, 1)):
            rng = np.random.default_rng(groups + s)
            N, C, H, W, kh, kw, cout = 2, 4, 7, 7, 3, 3, 4
            xn = rng.normal(0, 1, (N, C, H, W)).astype(np.float32)
            wn = rng.normal(0, 0.2, (cout, C // groups, kh, kw)).astype(np.float32)
            bn = rng.normal(0, 0.1, (cout,)).astype(np.float32)

            x = T.tensor(xn, requires_grad=True)
            w = T.tensor(wn, requires_grad=True)
            b = T.tensor(bn, requires_grad=True)
            y = conv2d(x, w, b, stride=s, padding=p, dilation=d, groups=groups)
            yn = np.asarray(y.to_numpy(), np.float32)
            gw = rng.normal(0, 1, yn.shape).astype(np.float32)
            (y * T.tensor(gw)).sum().backward()
            rt.sync()

            tx = torch.from_numpy(xn).requires_grad_(True)
            tw = torch.from_numpy(wn).requires_grad_(True)
            tb = torch.from_numpy(bn).requires_grad_(True)
            ty = torch.nn.functional.conv2d(tx, tw, tb, stride=s, padding=p, dilation=d,
                                            groups=groups)
            ty.backward(torch.from_numpy(gw))
            rs = [rel(yn, ty.detach().numpy()), rel(x.grad.to_numpy(), tx.grad.numpy()),
                  rel(w.grad.to_numpy(), tw.grad.numpy()), rel(b.grad.to_numpy(), tb.grad.numpy())]
            log("groups=%d s=%d p=%d d=%d | fwd %.2e dx %.2e dw %.2e db %.2e"
                % (groups, s, p, d, *rs))
            ok = ok and max(rs) <= 1e-5
    T.set_default_device("cpu")
    log("总结论：%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
