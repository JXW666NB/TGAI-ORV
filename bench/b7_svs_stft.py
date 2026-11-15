# -*- coding: utf-8 -*-
"""多分辨率 STFT 损失：ORV（svs-ops 插件）vs PyTorch 的**同形状**速度对比。

形状取自 yuaz-sgr 的 AE 训练步：audio `[B,1,96000]`（4 s @24 kHz），
分辨率 `[64,128,256,512,1024,2048]`、hop = n_fft/4（与 losses/stft_loss.py 一致）。

跑法（同一进程里两个框架都在）：
    set PYTHONPATH=D:\\orv_pip_0962\\Lib\\site-packages
    C:\\Python311\\python.exe examples\\bench_svs_stft.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import torch  # noqa: E402

import tgai_go.train as T  # noqa: E402
from tgai_go.train import plugin  # noqa: E402

FFT_SIZES = [64, 128, 256, 512, 1024, 2048]
HOPS = [s // 4 for s in FFT_SIZES]


def orv_loss(S, x, xh):
    total = None
    for fft, hop in zip(FFT_SIZES, HOPS):
        win = S.hann_window(fft)
        fn = getattr(S, "stft_mag_fast", S.stft_mag)
        a = fn(x, fft, hop_length=hop, window=win)
        b = fn(xh, fft, hop_length=hop, window=win)
        sc = ((a - b) * (a - b)).mean()
        la = (a.clamp(min=1e-4)).log()
        lb = (b.clamp(min=1e-4)).log()
        lg = ((la - lb) * (la - lb)).mean()
        part = sc + lg
        total = part if total is None else total + part
    return total


def torch_loss(x, xh):
    total = 0.0
    for fft, hop in zip(FFT_SIZES, HOPS):
        win = torch.hann_window(fft, device=x.device)
        a = torch.stft(x.squeeze(1), fft, hop, fft, win, return_complex=True).abs()
        b = torch.stft(xh.squeeze(1), fft, hop, fft, win, return_complex=True).abs()
        total = total + ((a - b) ** 2).mean()
        la, lb = torch.clamp(a, min=1e-4).log(), torch.clamp(b, min=1e-4).log()
        total = total + ((la - lb) ** 2).mean()
    return total


def main():
    B, L = 2, 96000
    rng = np.random.default_rng(0)
    a = (rng.normal(0, 0.3, (B, 1, L))).astype(np.float32)
    b = (a + rng.normal(0, 0.05, (B, 1, L))).astype(np.float32)

    T.plugin.install("examples/bundles/svs-ops", replace=True)
    S = T.svs
    T.set_default_device("cuda")
    rt = T._cuda()

    print('[1/3] ORV 预热中…', flush=True)
    # ---- ORV ----
    xo = T.from_numpy(a); xo.requires_grad = True
    xho = T.from_numpy(b); xho.requires_grad = True
    for _ in range(2):
        lo = orv_loss(S, xo, xho); rt.sync()
    rt.sync()
    print('[2/3] ORV 计时中…', flush=True)
    t0 = time.time()
    N = 5
    for _ in range(N):
        lo = orv_loss(S, xo, xho)
        lo.backward()
    rt.sync()
    orv_ms = (time.time() - t0) / N * 1000.0
    orv_val = float(lo.to_numpy())

    print('[3/3] PyTorch 计时中…', flush=True)
    # ---- PyTorch ----
    xt = torch.tensor(a, device="cuda", requires_grad=True)
    xht = torch.tensor(b, device="cuda", requires_grad=True)
    for _ in range(2):
        lt = torch_loss(xt, xht); torch.cuda.synchronize()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        lt = torch_loss(xt, xht)
        lt.backward()
    torch.cuda.synchronize()
    th_ms = (time.time() - t0) / N * 1000.0
    th_val = float(lt.detach())

    print("多分辨率 STFT 损失（6 档，B=%d, L=%d，前向+反向）" % (B, L))
    print("  ORV     : %.1f ms   loss=%.6f" % (orv_ms, orv_val))
    print("  PyTorch : %.1f ms   loss=%.6f" % (th_ms, th_val))
    rel = abs(orv_val - th_val) / (abs(th_val) or 1.0)
    print("  速度比   : ORV / PyTorch = %.2f×（<1 表示 ORV 更快）" % (orv_ms / th_ms))
    print("  loss 相对差: %.2e %s" % (rel, "OK" if rel < 1e-5 else "**超差**"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
