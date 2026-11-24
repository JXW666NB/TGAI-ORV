# -*- coding: utf-8 -*-
"""最小测速：多分辨率 STFT 损失，ORV(svs-ops 包) vs PyTorch，各测 3 轮。

刻意做到：① 每步都立刻打印（远端通道会掐长命令，增量输出能留下证据）；
          ② 只依赖 numpy + torch + ORV，不引别的。
"""
import sys
import time

print("[a] start", flush=True)
import numpy as np                                    # noqa: E402

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import torch                                          # noqa: E402
import tgai_go.train as T                             # noqa: E402

print("[b] imported torch %s" % torch.__version__, flush=True)

FFT = [64, 128, 256, 512, 1024, 2048]
HOP = [s // 4 for s in FFT]


def orv_loss(S, x, xh, use_fast=True):
    fn = S.stft_mag_fast if use_fast else S.stft_mag
    tot = None
    for f, h in zip(FFT, HOP):
        w = S.hann_window(f)
        a = fn(x, f, hop_length=h, window=w)
        b = fn(xh, f, hop_length=h, window=w)
        p = ((a - b) * (a - b)).mean()
        la = a.clamp(min=1e-4).log()
        lb = b.clamp(min=1e-4).log()
        p = p + ((la - lb) * (la - lb)).mean()
        tot = p if tot is None else tot + p
    return tot


def th_loss(x, xh):
    tot = 0.0
    for f, h in zip(FFT, HOP):
        w = torch.hann_window(f, device=x.device)
        a = torch.stft(x.squeeze(1), f, h, f, w, return_complex=True).abs()
        b = torch.stft(xh.squeeze(1), f, h, f, w, return_complex=True).abs()
        tot = tot + ((a - b) ** 2).mean()
        tot = tot + ((torch.clamp(a, min=1e-4).log() - torch.clamp(b, min=1e-4).log()) ** 2).mean()
    return tot


def main():
    T.plugin.install("examples/bundles/svs-ops", replace=True)
    S = T.svs
    print("[c] bundle installed, ops=%d" % len([n for n in dir(S) if not n.startswith("_")]), flush=True)
    T.set_default_device("cuda")
    rt = T._cuda()
    B, L = 2, 96000
    rng = np.random.default_rng(0)
    a = rng.normal(0, .3, (B, 1, L)).astype(np.float32)
    b = (a + rng.normal(0, .05, (B, 1, L))).astype(np.float32)

    xo = T.from_numpy(a); xo.requires_grad = True
    xh = T.from_numpy(b)
    lo = orv_loss(S, xo, xh); lo.backward(); rt.sync()
    t0 = time.time()
    for _ in range(3):
        lo = orv_loss(S, xo, xh); lo.backward()
    rt.sync()
    orv_ms = (time.time() - t0) / 3 * 1000
    print("[d] ORV  %.1f ms  loss=%.6f" % (orv_ms, float(lo.to_numpy())), flush=True)

    xt = torch.tensor(a, device="cuda", requires_grad=True)
    xht = torch.tensor(b, device="cuda")
    lt = th_loss(xt, xht); lt.backward(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(3):
        lt = th_loss(xt, xht); lt.backward()
    torch.cuda.synchronize()
    th_ms = (time.time() - t0) / 3 * 1000
    print("[e] PyTorch %.1f ms  loss=%.6f" % (th_ms, float(lt.detach())), flush=True)
    print("[f] 比值 ORV/PyTorch = %.2fx   loss 相对差 %.2e"
          % (orv_ms / th_ms, abs(float(lo.to_numpy()) - float(lt.detach())) / abs(float(lt.detach()))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
