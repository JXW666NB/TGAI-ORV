# -*- coding: utf-8 -*-
"""拆解组合式 STFT 的耗时：分帧卷积 vs DFT 基矩阵乘 vs 幅度谱。"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import plugin  # noqa: E402

FFT_SIZES = [64, 128, 256, 512, 1024, 2048]


def timeit(fn, n=3):
    fn()
    T._cuda().sync()
    t0 = time.time()
    for _ in range(n):
        fn()
    T._cuda().sync()
    return (time.time() - t0) / n * 1000.0


def main():
    plugin.load("examples/plugins/svs_audio_ops.py")
    plugin.activate("svs-ops")
    S = T.svs
    T.set_default_device("cuda")
    B, L = 2, 96000
    rng = np.random.default_rng(0)
    x = T.from_numpy(rng.normal(0, .3, (B, L)).astype(np.float32))

    print("%-10s %10s %10s %10s %10s" % ("n_fft", "帧数", "分帧ms", "基乘ms", "合计ms"))
    tot_f = tot_m = 0.0
    for fft in FFT_SIZES:
        hop = fft // 4
        pad = S.pad1d(x, fft // 2, fft // 2, mode="reflect")
        win = S.hann_window(fft)
        kernel = None
        # 分帧
        def frame_only():
            x2 = pad.reshape(B, 1, int(pad.shape[1]))
            return T.conv1d(x2, _ident(fft), stride=hop)
        f_ms = timeit(frame_only)

        frames = frame_only()
        nf = int(frames.shape[2])
        cos_b, sin_b = _basis(fft)

        def matmul_only():
            flat = frames.reshape(B, fft, nf)
            return [T.matmul(cos_b, flat[i]) for i in range(B)]
        m_ms = timeit(matmul_only)
        tot_f += f_ms
        tot_m += m_ms
        print("%-10d %10d %10.1f %10.1f %10.1f" % (fft, nf, f_ms, m_ms, f_ms + m_ms))
    print("-" * 56)
    print("%-10s %10s %10.1f %10.1f %10.1f" % ("合计", "", tot_f, tot_m, tot_f + tot_m))
    print("\n一档损失要算 x 与 x_hat 两份 ⇒ ×2 ≈ %.0f ms（还有反向）" % ((tot_f + tot_m) * 2))
    return 0


_IDENT = {}


def _ident(n):
    if n not in _IDENT:
        w = np.zeros((n, 1, n), np.float32)
        for i in range(n):
            w[i, 0, i] = 1.0
        _IDENT[n] = T.from_numpy(w)
    return _IDENT[n]


_BAS = {}


def _basis(n):
    if n not in _BAS:
        k = np.arange(n // 2 + 1, dtype=np.float64)[:, None]
        j = np.arange(n, dtype=np.float64)[None, :]
        ang = 2.0 * np.pi * k * j / n
        _BAS[n] = (T.from_numpy(np.cos(ang).astype(np.float32)),
                   T.from_numpy((-np.sin(ang)).astype(np.float32)))
    return _BAS[n]


if __name__ == "__main__":
    raise SystemExit(main())
