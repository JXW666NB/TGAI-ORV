# -*- coding: utf-8 -*-
"""逐算子对拍 ORV 的 audio-ops 插件与 PyTorch（同一个进程里两边都有）。

ORV 是纯 Python + ctypes，所以能和真 torch 共存于一个解释器 —— 这才做得到直接对拍。
（注意：不要 activate `torch-compat`，那会覆盖 sys.modules['torch']。）
"""
import sys

import numpy as np

sys.path.insert(0, "D:/orv_src/python")          # ORV 源码
sys.path.insert(0, "D:/orv_src/examples/plugins")


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.train import plugin

    T.set_default_device("cuda")
    plugin.load("D:/orv_src/examples/plugins/audio_ops.py")
    plugin.activate("audio-ops")
    A = T.audio
    print("插件已激活；T.audio 有 %d 个算子" % len([k for k in dir(A) if not k.startswith("_")]))

    rng = np.random.default_rng(0)
    x = rng.uniform(-3.0, 3.0, (4, 1024)).astype(np.float32)
    x_pos = rng.uniform(0.05, 3.0, (4, 1024)).astype(np.float32)

    results = []

    def check(name, orv_fn, torch_fn, arr, atol=2e-6):
        o = orv_fn(T.tensor(arr)).to_numpy()
        t = torch_fn(torch.from_numpy(arr).cuda()).cpu().numpy()
        d = float(np.abs(o - t).max())
        results.append((name, d, d <= atol))
        print("  %-14s 最大差 %.3e  %s" % (name, d, "OK" if d <= atol else "**超差**"))

    print("逐元素（op_native 现场编译的内核）：")
    check("sin", A.sin, torch.sin, x)
    check("cos", A.cos, torch.cos, x)
    check("sigmoid", A.sigmoid, torch.sigmoid, x)
    check("softplus", A.softplus, torch.nn.functional.softplus, x)
    check("log1p", A.log1p, torch.log1p, x_pos)
    check("log10", A.log10, torch.log10, x_pos)
    check("leaky_relu", A.leaky_relu, lambda t: torch.nn.functional.leaky_relu(t, 0.01), x)

    print("组合实现（只用 ORV 已有算子）：")
    o = A.hann_window(1024).to_numpy()
    t = torch.hann_window(1024).cpu().numpy()
    d = float(np.abs(o - t).max())
    results.append(("hann_window", d, d <= 1e-6))
    print("  %-14s 最大差 %.3e  %s" % ("hann_window", d, "OK" if d <= 1e-6 else "**超差**"))

    a = rng.normal(0, 1, (8, 64)).astype(np.float32)
    b = rng.normal(0, 1, (8, 64)).astype(np.float32)
    ta, tb = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
    for name, ofn, tfn in (
        ("l1_loss", lambda p, q: A.l1_loss(T.tensor(p), T.tensor(q)),
         lambda p, q: torch.nn.functional.l1_loss(p, q)),
        ("mse_loss", lambda p, q: A.mse_loss(T.tensor(p), T.tensor(q)),
         lambda p, q: torch.nn.functional.mse_loss(p, q)),
    ):
        o = float(ofn(a, b).to_numpy())
        t = float(tfn(ta, tb).cpu().numpy())
        d = abs(o - t)
        results.append((name, d, d <= 1e-6))
        print("  %-14s 差 %.3e  %s" % (name, d, "OK" if d <= 1e-6 else "**超差**"))

    # pad
    p = rng.normal(0, 1, (2, 10)).astype(np.float32)
    o = A.pad_last(T.tensor(p), 2, 3).to_numpy()
    t = torch.nn.functional.pad(torch.from_numpy(p).cuda(), (2, 3)).cpu().numpy()
    d = float(np.abs(o - t).max())
    results.append(("pad_last", d, d <= 0 and np.array_equal(o, t)))
    print("  %-14s 逐位相等 %s" % ("pad_last", np.array_equal(o, t)))

    ok = sum(1 for r in results if r[2])
    print("\n合计 %d 项，通过 %d 项" % (len(results), ok))
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
