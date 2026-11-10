# -*- coding: utf-8 -*-
"""GPU 上额外契约：Conv1d 委托路径、groups>1 明确报错、Conv2d 模块类。

运行（需独占 GPU）：python conv_cuda_extra.py
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
    from tgai_go.train.conv import Conv1d, Conv2d, conv1d, conv2d

    T.set_default_device("cuda")
    rt = T._cuda()
    ok = True

    # 1) Conv1d：NCL，stride/padding/dilation 全给
    rng = np.random.default_rng(0)
    N, C, L, kw, cout = 2, 3, 11, 3, 4
    xn = rng.normal(0, 1, (N, C, L)).astype(np.float32)
    wn = rng.normal(0, 0.2, (cout, C, kw)).astype(np.float32)
    bn = rng.normal(0, 0.1, (cout,)).astype(np.float32)
    for s, p, d in ((1, 1, 1), (2, 1, 1), (1, 2, 2)):
        x = T.tensor(xn, requires_grad=True)
        w = T.tensor(wn, requires_grad=True)
        b = T.tensor(bn, requires_grad=True)
        y = conv1d(x, w, b, stride=s, padding=p, dilation=d)
        yn = np.asarray(y.to_numpy(), np.float32)
        gw = rng.normal(0, 1, yn.shape).astype(np.float32)
        (y * T.tensor(gw)).sum().backward()
        rt.sync()
        tx = torch.from_numpy(xn).requires_grad_(True)
        tw = torch.from_numpy(wn).requires_grad_(True)
        tb = torch.from_numpy(bn).requires_grad_(True)
        ty = torch.nn.functional.conv1d(tx, tw, tb, stride=s, padding=p, dilation=d)
        ty.backward(torch.from_numpy(gw))
        rs = [rel(yn, ty.detach().numpy()), rel(x.grad.to_numpy(), tx.grad.numpy()),
              rel(w.grad.to_numpy(), tw.grad.numpy()),
              rel(b.grad.to_numpy(), tb.grad.numpy())]
        log("conv1d s=%d p=%d d=%d | fwd %.2e dx %.2e dw %.2e db %.2e" % (s, p, d, *rs))
        ok = ok and max(rs) <= 1e-5

    # 2) groups>1 必须报错
    try:
        x = T.tensor(xn, requires_grad=False)
        w2 = T.tensor(rng.normal(0, 0.2, (4, 1, 1, kw)).astype(np.float32), requires_grad=False)
        conv2d(x.reshape((N, C, 1, L)), w2, None, stride=1, padding=0, dilation=1, groups=3)
        log("groups>1 未报错 ⇒ FAIL")
        ok = False
    except NotImplementedError as e:
        log("groups>1 正确报错：%s" % str(e)[:60])

    # 3) 模块类：Conv2d 前向 + 反向
    m = Conv2d(2, 3, 3, padding=1)
    m.to("cuda") if hasattr(m, "to") else None
    xm = T.tensor(rng.normal(0, 1, (1, 2, 5, 5)).astype(np.float32), requires_grad=True)
    ym = m(xm)
    ym.sum().backward()
    rt.sync()
    gm = np.asarray(m.weight.grad.to_numpy(), np.float32)
    log("Conv2d 模块类：输出 %s，权重梯度 max|.|=%.4g" % (tuple(ym.shape), float(np.abs(gm).max())))
    ok = ok and np.isfinite(gm).all() and float(np.abs(gm).max()) > 0

    log("总结论：%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
