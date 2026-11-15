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


def _dft_basis(n_fft):
    key = ("dft", n_fft)
    if key not in _CACHE:
        k = np.arange(n_fft // 2 + 1, dtype=np.float64)[:, None]
        n = np.arange(n_fft, dtype=np.float64)[None, :]
        ang = 2.0 * np.pi * k * n / n_fft
        _CACHE[key] = (np.cos(ang).astype(np.float32), (-np.sin(ang)).astype(np.float32))
    c, s = _CACHE[key]
    return _tv(c), _tv(s)


def _ident_kernel(n_fft):
    key = ("ident", n_fft)
    if key not in _CACHE:
        w = np.zeros((n_fft, 1, n_fft), np.float32)
        for i in range(n_fft):
            w[i, 0, i] = 1.0
        _CACHE[key] = w
    return _tv(_CACHE[key])


def _tri(n):
    key = ("tri", n)
    if key not in _CACHE:
        _CACHE[key] = np.tril(np.ones((n, n), np.float32))
    return _tv(_CACHE[key])


def _reverse_last(n):
    """返回反序置换矩阵 P（n×n），P @ x 沿最后一维翻转。"""
    key = ("rev", n)
    if key not in _CACHE:
        _CACHE[key] = np.eye(n, dtype=np.float32)[::-1].copy()
    return _tv(_CACHE[key])


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
            parts.append(edge.repeat_interleave(pl) if pl > 1 else edge)
        parts.append(x)
        if pr > 0:
            edge = x[:, -1:] if x.ndim == 2 else x[..., -1:]
            parts.append(edge.repeat_interleave(pr) if pr > 1 else edge)
        return T.cat(parts, dim=-1)
    if mode == "reflect":
        # abcd -> cb|abcd|cb（不含边界点）
        parts = []
        if pl > 0:
            seg = x[..., 1:pl + 1]
            parts.append(T.matmul(_reverse_last(seg.shape[-1]), seg))
        parts.append(x)
        if pr > 0:
            n = int(x.shape[-1])
            seg = x[..., n - pr - 1:n - 1]
            parts.append(T.matmul(_reverse_last(seg.shape[-1]), seg))
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
    re = T.stack([T.matmul(cos_b, flat[i]) for i in range(b)], dim=0)
    im = T.stack([T.matmul(sin_b, flat[i]) for i in range(b)], dim=0)
    return (re * re + im * im + 1.0e-12).sqrt()


# ── 扫描 / 索引 ───────────────────────────────────────────────────────
def cumsum(x, dim=-1):
    """沿最后一维累加（三角矩阵乘）。"""
    import tgai_go.train as T
    if dim not in (-1, x.ndim - 1):
        raise ValueError("cumsum 只支持最后一维。修法：先 permute 把该维移到最后")
    n = int(x.shape[-1])
    L = _tri(n)
    lead = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
    flat = x.reshape(lead, n)
    y = T.stack([T.matmul(L, flat[i]) for i in range(lead)], dim=0)
    return y.reshape(tuple(x.shape))


def gather_last(x, index_np):
    """沿最后一维按整数索引取值。index_np: numpy 数组，形状与 x 去掉最后一维后一致。"""
    import tgai_go.train as T
    n = int(x.shape[-1])
    idx = np.asarray(index_np, np.int64).reshape(-1)
    lead = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
    oh = np.zeros((idx.size, n), np.float32)
    oh[np.arange(idx.size), idx] = 1.0
    flat = x.reshape(lead, n)
    y = T.stack([T.matmul(_tv(oh), flat[i]) for i in range(lead)], dim=0)
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


def _register():
    from tgai_go.train import plugin
    plugin.api("svs-ops", install, replace=True)


_register()

__all__ = sorted(_OPS)
