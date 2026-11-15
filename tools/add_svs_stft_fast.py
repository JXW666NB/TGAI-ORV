# -*- coding: utf-8 -*-
"""给 svs-ops 插件加 STFT 快路径：用「hop 块分解」替掉恒等核卷积分帧。

## 为什么

实测（`bench/b8_svs_stft_parts.py`，B=2、L=96000、6 档分辨率）：

| 环节 | 耗时 |
| --- | --- |
| 分帧（恒等核 `conv1d`） | **791.7 ms（99%）** |
| DFT 基矩阵乘 | 5.2 ms |
| 合计 | 796.9 ms |

结论：**不用 cuFFT**，只要把分帧干掉就能根治。

## 数学

hop = n_fft/4 时，一帧正好是 4 个连续 hop 块的拼接：

    X[k,i] = Σ_s  e^{-2πi·k·s·hop/n} · ( basis[:, s·hop:(s+1)·hop] @ seg_s[i] )

于是"取帧"变成 **4 次连续切片**（`blocks[:, s:s+nframes, :]`），
再对每段做一次矩阵乘并乘上相位因子。总乘加数与原来相同，但全走 matmul（5 ms 级），
不再是低效的卷积核。

纯组合实现 ⇒ 梯度由 autograd 自动给出，仍然不用写反向。

用法：python tools/add_svs_stft_fast.py
"""
import io

P = "examples/plugins/svs_audio_ops.py"

CODE = '''

def stft_mag_fast(x, n_fft, hop_length=None, win_length=None, window=None,
                  center=True, pad_mode="reflect"):
    """|STFT(x)| 的快路径（hop 块分解，无分帧卷积）。返回 [B, n_fft//2+1, n_frames]。"""
    import tgai_go.train as T

    n_fft = int(n_fft)
    hop = int(hop_length or n_fft // 4)
    win = int(win_length or n_fft)
    if win != n_fft:
        raise ValueError("stft_mag_fast 要求 win_length == n_fft。修法：把窗补零到 n_fft")
    if n_fft % hop != 0:
        return stft_mag(x, n_fft, hop, win, window, center, pad_mode)   # 退回通用路径
    nseg = n_fft // hop

    if x.ndim == 3:
        x = x[:, 0] if int(x.shape[1]) == 1 else x.reshape(int(x.shape[0]), -1)
    if window is None:
        window = hann_window(n_fft)
    if center:
        x = pad1d(x, win // 2, win // 2, mode=pad_mode)

    b, ln = int(x.shape[0]), int(x.shape[1])
    nblocks_needed = (ln - n_fft) // hop + 1 + nseg - 1
    total = nblocks_needed * hop
    if ln < total:
        x = pad1d(x, 0, total - ln, mode="constant")
        ln = total
    nframes = (ln - n_fft) // hop + 1
    nb = ln // hop

    blocks = x.reshape(b, nb, hop)                 # [B, nb, hop]
    cos_b, sin_b = _dft_basis(n_fft)               # [F, n]
    nfreq = int(cos_b.shape[0])

    re = im = None
    for s in range(nseg):
        wseg = window[s * hop:(s + 1) * hop].reshape(1, 1, hop)
        seg = blocks[:, s:s + nframes, :] * wseg   # [B, nframes, hop]
        cs = cos_b[:, s * hop:(s + 1) * hop]       # [F, hop]
        sn = sin_b[:, s * hop:(s + 1) * hop]
        As, Bs = [], []
        for i in range(b):
            As.append(_rowwise_mm(cs, seg[i]))     # [nframes, F]
            Bs.append(_rowwise_mm(sn, seg[i]))
        A = As[0] if b == 1 else T.cat([a.reshape(1, nframes, nfreq) for a in As], dim=0)
        Bq = Bs[0] if b == 1 else T.cat([v.reshape(1, nframes, nfreq) for v in Bs], dim=0)
        if b == 1:
            A = A.reshape(1, nframes, nfreq)
            Bq = Bq.reshape(1, nframes, nfreq)

        theta = 2.0 * np.pi * np.arange(nfreq, dtype=np.float64)[:, None] * (s * hop) / n_fft
        cph = _tv(np.cos(theta).astype(np.float32)).reshape(1, 1, nfreq)
        sph = _tv((-np.sin(theta)).astype(np.float32)).reshape(1, 1, nfreq)

        pre = cph * A - sph * Bq
        pim = cph * Bq + sph * A
        re = pre if re is None else re + pre
        im = pim if im is None else im + pim

    mag = (re * re + im * im + 1.0e-12).sqrt()     # [B, nframes, F]
    return mag.transpose(1, 2)                     # -> [B, F, nframes]

'''


def main():
    s = io.open(P, encoding="utf-8").read()
    if "def stft_mag_fast" in s:
        print("已经加过 stft_mag_fast")
        return 0
    anchor = "def cumsum(x, dim=-1):"
    assert anchor in s, "找不到 cumsum 锚点"
    s = s.replace(anchor, CODE.lstrip("\n") + anchor, 1)
    assert '"stft_mag": stft_mag,' in s
    s = s.replace('"stft_mag": stft_mag,',
                  '"stft_mag": stft_mag,\n    "stft_mag_fast": stft_mag_fast,', 1)
    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    import ast
    ast.parse(io.open(P, encoding="utf-8").read())
    print("已加 stft_mag_fast（hop 块分解），并注册进命名空间")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
