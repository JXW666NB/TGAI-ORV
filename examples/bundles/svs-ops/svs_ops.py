# -*- coding: utf-8 -*-
"""svs-ops —— 虚拟歌手（SVS / DDSP 歌声合成）训练所需的算子插件。

## 为什么需要它

ORV 是**通用训练引擎**，不是音频库。拿真实 SVS 项目 `yuaz-sgr`
（DDSP + 残差向量量化 + 潜空间流匹配 GAN）盘点，模型路径用到 60+ 个 torch 接口，
ORV 同名同位约一半。缺的那批不是 LLM 需要的，而是**音频/信号处理**专用：

| 类别 | 缺的接口 |
| --- | --- |
| 频域 | `torch.stft`、`torch.hann_window` |
| 形状 | `F.interpolate`、`F.pad`、`F.avg_pool1d` |
| 逐元素 | `sigmoid`、`softplus`、`log1p`、`log10`、`pow` |
| 扫描/索引 | `torch.cumsum`、`torch.gather` |
| 损失 | `F.l1_loss`、`F.mse_loss`、`F.binary_cross_entropy(_with_logits)` |

## 做法：**组合式实现，一个字都不写反向**

ORV 的 autograd 对组合出来的算子自动求导，所以只要每一步都能用已有算子表达，
梯度就免费且正确。三个关键技巧：

1. **STFT 不需要复数张量**：分帧写成 `conv1d`（恒等核 + `stride=hop`），
   再乘预先算好的余弦/正弦 DFT 基矩阵得到实部/虚部，`|X| = sqrt(re²+im²)`。
   全程 fp32，梯度自然流过。
2. **cumsum 写成三角矩阵乘法**：`cumsum(x) = L @ x`，L 是下三角全 1。
3. **pad / 翻转用置换矩阵乘**，不用负步长切片（ORV 没实现）。

## 诚实边界

| 未实现 | 原因 |
| --- | --- |
| `istft` | 训练只用 `stft` 算谱损失；推理需要时再补 |
| `linalg.pinv` / `unique` / `searchsorted` | 出现在预处理与推理脚本，不在训练热路径 |
| 真正 cuFFT 性能 | 基矩阵乘比 cuFFT 慢；换原生 FFT 内核是后续优化 |

`isfinite` 作为**掩码**在主机侧算（它本身不可导），结果与 torch 一致。

## 用法

```python
import tgai_go.train as T
T.plugin.load("examples/plugins/svs_audio_ops.py")
T.plugin.activate("svs-ops")

win = T.svs.hann_window(1024)
mag = T.svs.stft_mag(wav, 1024, hop_length=256, window=win)   # [B, F, T]
loss = T.svs.l1_loss(mag, mag_hat)
```
"""
import types

import numpy as np

LN10 = 2.302585092994046

_BASIS = {}
_CACHE = {}


def _tv(a):
    import tgai_go.train as T
    return T.from_numpy(np.ascontiguousarray(a, np.float32))


_TENSOR_CACHE = {}


def _cached_tensor(key, builder):
    """按 (key, 当前设备) 缓存**张量本体**（不是 numpy）。

    为什么必要：_tv(np_array) 每次调用都会重新做一次 H2D 上传。
    DFT 基矩阵在 2048 档是 8.4 MB × 2 个，一轮训练要传十几次 ⇒ 光传输就几十毫秒。
    实测：缓存张量后 ORV 的多分辨率 STFT 损失从 218.8 ms 降到（见 docs 记录）。
    """
    import tgai_go.train as T
    dev = str(T.get_device())
    k = (key, dev)
    t = _TENSOR_CACHE.get(k)
    if t is None:
        t = _tv(builder())
        _TENSOR_CACHE[k] = t
    return t


def _dft_basis(n_fft):
    key = ("dft", n_fft)
    if key not in _CACHE:
        k = np.arange(n_fft // 2 + 1, dtype=np.float64)[:, None]
        n = np.arange(n_fft, dtype=np.float64)[None, :]
        ang = 2.0 * np.pi * k * n / n_fft
        _CACHE[key] = (np.cos(ang).astype(np.float32), (-np.sin(ang)).astype(np.float32))
    c, s = _CACHE[key]
    return _cached_tensor(('cos', n_fft), lambda: c), _cached_tensor(('sin', n_fft), lambda: s)


def _ident_kernel(n_fft):
    key = ("ident", n_fft)
    if key not in _CACHE:
        w = np.zeros((n_fft, 1, n_fft), np.float32)
        for i in range(n_fft):
            w[i, 0, i] = 1.0
        _CACHE[key] = w
    return _cached_tensor(('ident', n_fft), lambda: _CACHE[key])


def _tri(n):
    key = ("tri", n)
    if key not in _CACHE:
        _CACHE[key] = np.tril(np.ones((n, n), np.float32))
    return _cached_tensor(('tri', n), lambda: _CACHE[key])


def _reverse_last(n):
    """返回反序置换矩阵 P（n×n），P @ x 沿最后一维翻转。"""
    key = ("rev", n)
    if key not in _CACHE:
        _CACHE[key] = np.eye(n, dtype=np.float32)[::-1].copy()
    return _cached_tensor(('rev', n), lambda: _CACHE[key])


# ── 逐元素（只用 abs/exp/log/sqrt/tanh/clamp，避免比较运算）────────────
def sigmoid(x):
    """1/(1+e^-x)，用 tanh 表达：0.5*(tanh(0.5x)+1) —— 数值稳定且梯度精确。"""
    return (x * 0.5).tanh() * 0.5 + 0.5


def softplus(x):
    """log(1+e^x) = max(x,0) + log(1+e^{-|x|})。"""
    ax = x.abs()
    return x.clamp(min=0.0) + (1.0 + (-ax).exp()).log()


def log1p(x):
    return (x + 1.0).log()


def log10(x):
    return x.log() * (1.0 / LN10)


def pow_(x, p):
    p = float(p)
    if p == 2.0:
        return x * x
    if p == 0.5:
        return x.sqrt()
    if p == 1.0:
        return x
    return (x.log() * p).exp()


def leaky_relu(x, slope=0.01):
    """max(x,0) + slope*min(x,0)，用 clamp 表达。"""
    return x.clamp(min=0.0) + x.clamp(max=0.0) * float(slope)


# ── 形状 ──────────────────────────────────────────────────────────────
def _pad_const(x, pl, pr):
    import tgai_go.train as T
    parts = []
    if pl > 0:
        parts.append(T.zeros(tuple(list(x.shape)[:-1] + [pl])))
    parts.append(x)
    if pr > 0:
        parts.append(T.zeros(tuple(list(x.shape)[:-1] + [pr])))
    return T.cat(parts, dim=-1) if len(parts) > 1 else x



def _rowwise_mm(mat, x):
    """mat:[m,n] 作用在 x 的**最后一维**（x 可多维）：返回 [..., m]。

    ORV 的 matmul 只吃 2D，所以把前置维展平、逐行按列向量乘、再按列拼回并转置。
    """
    import tgai_go.train as T
    n = int(mat.shape[1])
    m = int(mat.shape[0])
    lead = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
    flat = x.reshape(lead, n)
    # 注意：不能用 cat(...).T.reshape(...) —— ORV 的惰性转置(_lazy_T)不被 reshape 识别，
    # 会静默给出未转置的数据。改为逐行 reshape 成 [1, m] 再按 dim=0 拼。
    rows = [T.matmul(mat, flat[i].reshape(n, 1)).reshape(1, m) for i in range(lead)]
    y = T.cat(rows, dim=0) if lead > 1 else rows[0]
    return y.reshape(tuple(list(x.shape[:-1]) + [m]))


def pad1d(x, pad_left, pad_right, mode="constant"):
    """沿最后一维填充。mode: constant / replicate / reflect。"""
    import tgai_go.train as T
    pl, pr = int(pad_left), int(pad_right)
    if pl == 0 and pr == 0:
        return x
    if mode == "constant":
        return _pad_const(x, pl, pr)
    if mode == "replicate":
        parts = []
        if pl > 0:
            edge = x[:, 0:1] if x.ndim == 2 else x[..., 0:1]
            parts.append(T.cat([edge] * pl, dim=-1) if pl > 1 else edge)
        parts.append(x)
        if pr > 0:
            edge = x[:, -1:] if x.ndim == 2 else x[..., -1:]
            parts.append(T.cat([edge] * pr, dim=-1) if pr > 1 else edge)
        return T.cat(parts, dim=-1)
    if mode == "reflect":
        # abcd -> cb|abcd|cb（不含边界点）
        parts = []
        if pl > 0:
            seg = x[..., 1:pl + 1]
            parts.append(_rowwise_mm(_reverse_last(int(seg.shape[-1])), seg))
        parts.append(x)
        if pr > 0:
            n = int(x.shape[-1])
            seg = x[..., n - pr - 1:n - 1]
            parts.append(_rowwise_mm(_reverse_last(int(seg.shape[-1])), seg))
        return T.cat(parts, dim=-1)
    raise ValueError("pad1d 不支持的 mode=%r。修法：constant / replicate / reflect" % (mode,))


def hann_window(n, periodic=True):
    n = int(n)
    if periodic:
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)
    else:
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / max(n - 1, 1))
    return _tv(w)


# ── STFT 幅度谱 ───────────────────────────────────────────────────────
def stft_mag(x, n_fft, hop_length=None, win_length=None, window=None,
             center=True, pad_mode="reflect"):
    """|STFT(x)|。x: [B, T] 或 [B, 1, T] → [B, n_fft//2+1, n_frames]。"""
    import tgai_go.train as T
    n_fft = int(n_fft)
    hop = int(hop_length or n_fft // 4)
    win = int(win_length or n_fft)
    if win != n_fft:
        raise ValueError("stft_mag 要求 win_length == n_fft。修法：把窗补零到 n_fft")
    if x.ndim == 3:
        x = x[:, 0] if int(x.shape[1]) == 1 else x.reshape(int(x.shape[0]), -1)
    if center:
        x = pad1d(x, win // 2, win // 2, mode=pad_mode)

    b, t = int(x.shape[0]), int(x.shape[1])
    x2 = x.reshape(b, 1, t)
    frames = T.conv1d(x2, _ident_kernel(n_fft), stride=hop)      # [B, n_fft, frames]
    if window is not None:
        frames = frames * window.reshape(1, n_fft, 1)

    cos_b, sin_b = _dft_basis(n_fft)
    nf = int(frames.shape[2])
    flat = frames.reshape(b, n_fft, nf)
    # T.stack 在 CUDA 上未实现，改用 cat + reshape
    nfreq = int(cos_b.shape[0])
    re = T.cat([T.matmul(cos_b, flat[i]).reshape(1, nfreq, nf) for i in range(b)], dim=0)
    im = T.cat([T.matmul(sin_b, flat[i]).reshape(1, nfreq, nf) for i in range(b)], dim=0)
    return (re * re + im * im + 1.0e-12).sqrt()


# ── 扫描 / 索引 ───────────────────────────────────────────────────────
def stft_mag_fast(x, n_fft, hop_length=None, win_length=None, window=None,
                  center=True, pad_mode="reflect"):
    """|STFT(x)| 的快路径：hop 块分解，**不做分帧卷积**。

    一帧 = nseg 个连续 hop 块；块内用**局部坐标 u=0..hop-1** 的基矩阵，
    段间再乘相位 e^{-2πi·k·s·hop/n}。（2026-11-15 修：原先误用全局列索引，等于算了两遍相位。）
    """
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
    # ★ 帧数必须锚定在**原始信号长度**上（与 torch 的 center 语义一致）。
    #   后面为了凑成 hop 的整数倍会再补零，但那是给 reshape 用的，不能改变帧数。
    nframes = (ln - n_fft) // hop + 1
    nb = max(nframes + nseg - 1, (ln + hop - 1) // hop)
    total = nb * hop
    if ln < total:
        x = pad1d(x, 0, total - ln, mode="constant")
        ln = total

    blocks = x.reshape(b, nb, hop)                 # [B, nb, hop]
    cos_b, sin_b = _dft_basis(n_fft)               # [F, n]
    nfreq = int(cos_b.shape[0])

    re = im = None
    for s in range(nseg):
        wseg = window[s * hop:(s + 1) * hop].reshape(1, 1, hop)
        seg = blocks[:, s:s + nframes, :] * wseg   # [B, nframes, hop]
        # 段内是局部坐标：基矩阵永远取前 hop 列（不是 [s*hop:(s+1)*hop]）
        cs = cos_b[:, 0:hop]                       # [F, hop]
        sn = sin_b[:, 0:hop]
        # ★ 两个操作数都是 2D：一次 matmul 就够（[F,hop] @ [hop,nframes]）。
        #   早期版本用逐帧循环，光 Python 调度就要 3000 次矩阵乘，慢在这里。
        As, Bs = [], []
        for i in range(b):
            segT = seg[i].T                        # [hop, nframes]，走 gemm_nt 不拷贝
            As.append(T.matmul(cs, segT))           # [F, nframes]
            Bs.append(T.matmul(sn, segT))
        A = As[0].reshape(1, nfreq, nframes) if b == 1 else \
            T.cat([a.reshape(1, nfreq, nframes) for a in As], dim=0)
        Bq = Bs[0].reshape(1, nfreq, nframes) if b == 1 else \
            T.cat([v.reshape(1, nfreq, nframes) for v in Bs], dim=0)

        theta = 2.0 * np.pi * np.arange(nfreq, dtype=np.float64)[:, None] * (s * hop) / n_fft
        cph = _tv(np.cos(theta).astype(np.float32)).reshape(1, nfreq, 1)
        sph = _tv((-np.sin(theta)).astype(np.float32)).reshape(1, nfreq, 1)

        pre = cph * A - sph * Bq
        pim = cph * Bq + sph * A
        re = pre if re is None else re + pre
        im = pim if im is None else im + pim

    return (re * re + im * im + 1.0e-12).sqrt()    # [B, F, nframes]

def _interp_weights(n_in, n_out, mode):
    """返回 [n_out, n_in] 的插值权重矩阵（numpy，常量）。"""
    n_in, n_out = int(n_in), int(n_out)
    key = ("interp", n_in, n_out, mode)
    if key in _CACHE:
        return _CACHE[key]
    W = np.zeros((n_out, n_in), np.float32)
    # ★ 必须用 **float32** 算：torch 的 F.interpolate 就是 float32 取整，
    #   用 float64 会在边界上差 1 格（实测 96256->189 相对差 0.84）。
    scale32 = np.float32(n_in) / np.float32(n_out)
    if mode == "nearest":
        idx = np.floor(np.arange(n_out, dtype=np.float32) * scale32).astype(np.int64)
        idx = np.clip(idx, 0, n_in - 1)
        W[np.arange(n_out), idx] = 1.0
    elif mode == "linear":
        src = (np.arange(n_out, dtype=np.float32) + np.float32(0.5)) * scale32 - np.float32(0.5)
        src = np.clip(src, 0.0, n_in - 1)
        lo = np.floor(src).astype(np.int64)
        hi = np.minimum(lo + 1, n_in - 1)
        frac = (src - lo).astype(np.float32)
        W[np.arange(n_out), lo] += 1.0 - frac
        W[np.arange(n_out), hi] += frac
    else:
        raise ValueError("interpolate1d 不支持的 mode=%r。修法：用 'linear' 或 'nearest'" % (mode,))
    _CACHE[key] = W
    return W


def interpolate1d(x, size=None, scale_factor=None, mode="linear"):
    """沿**最后一维**插值到 `size`（或按 `scale_factor` 放大）。

    与 `torch.nn.functional.interpolate(..., size=n, mode=mode)` 等价（`align_corners=False`）。
    纯组合式：权重矩阵是常量，梯度由 autograd 给出。
    """
    if size is None:
        if scale_factor is None:
            raise ValueError("interpolate1d 需要 size 或 scale_factor。修法：传其中一个")
        size = int(round(int(x.shape[-1]) * float(scale_factor)))
    n_in, n_out = int(x.shape[-1]), int(size)
    if n_in == n_out:
        return x
    W = _cached_tensor(("interp", n_in, n_out, mode), lambda: _interp_weights(n_in, n_out, mode))
    # W:[out,in] 作用在最后一维；_rowwise_mm 会把前置维展平后逐行乘
    return _rowwise_mm(W, x)


def cumsum(x, dim=-1):
    """沿最后一维累加（三角矩阵乘）。"""
    import tgai_go.train as T
    if dim not in (-1, x.ndim - 1):
        raise ValueError("cumsum 只支持最后一维。修法：先 permute 把该维移到最后")
    n = int(x.shape[-1])
    return _rowwise_mm(_tri(n), x)


def gather_last(x, index_np):
    """沿最后一维按整数索引取值。index_np: numpy 数组，形状与 x 去掉最后一维后一致。"""
    import tgai_go.train as T
    n = int(x.shape[-1])
    idx = np.asarray(index_np, np.int64).reshape(-1)
    oh = np.zeros((idx.size, n), np.float32)
    oh[np.arange(idx.size), idx] = 1.0
    y = _rowwise_mm(_tv(oh), x)          # [..., m]
    return y.reshape(tuple(list(x.shape[:-1]) + [1]))


def isfinite_np(x):
    """不可导掩码：返回 float32 的 0/1（主机侧算，与 torch.isfinite 一致）。"""
    return _tv(np.isfinite(np.asarray(x, np.float32)).astype(np.float32))


# ── 损失 ──────────────────────────────────────────────────────────────
def l1_loss(a, b, reduction="mean"):
    d = (a - b).abs()
    return d.mean() if reduction == "mean" else d.sum()


def mse_loss(a, b, reduction="mean"):
    d = a - b
    v = d * d
    return v.mean() if reduction == "mean" else v.sum()


def softplus_loss_term(x):
    return softplus(x)


def bce_with_logits(logits, target):
    """max(x,0) - x·t + log(1+e^{-|x|})，逐元素后取均值。"""
    return (logits.clamp(min=0.0) - logits * target + softplus(-logits.abs())).mean()


def bce(prob, target):
    eps = 1.0e-7
    p = prob.clamp(eps, 1.0 - eps)
    return (0.0 - (target * p.log() + (1.0 - target) * (1.0 - p).log())).mean()


def avg_pool1d(x, kernel_size, stride=None, padding=0):
    """一维平均池化（x: [B, C, T] 或 [B, T]）。"""
    import tgai_go.train as T
    m = T.nn.AvgPool1d(kernel_size, stride if stride is not None else kernel_size, padding)
    return m(x)


# ── 安装 ──────────────────────────────────────────────────────────────
_OPS = {
    "sigmoid": sigmoid, "softplus": softplus, "log1p": log1p, "log10": log10,
    "pow": pow_, "leaky_relu": leaky_relu,
    "pad1d": pad1d, "hann_window": hann_window, "stft_mag": stft_mag,
    "cumsum": cumsum, "gather_last": gather_last, "isfinite_np": isfinite_np,
    "l1_loss": l1_loss, "mse_loss": mse_loss,
    "bce_with_logits": bce_with_logits, "bce": bce, "avg_pool1d": avg_pool1d,
}


def install():
    """装出 `T.svs` 命名空间并返回它。"""
    import tgai_go.train as T
    ns = types.SimpleNamespace(**_OPS)
    T.svs = ns
    return ns


# 作为插件包分发时不需要自注册：manifest.json 声明算子名，plugin_meta.install() 负责注册。

__all__ = sorted(_OPS)
