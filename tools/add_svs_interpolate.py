# -*- coding: utf-8 -*-
"""给 svs-ops 加 `interpolate1d`（linear / nearest），与 torch 对拍。

yuaz-sgr 全模型用了 28 处 `F.interpolate(..., size=…)`，全是 1 维（最后一维）：
* `mode="linear"`（23 处）：`align_corners` 全部未传 ⇒ 等价 `align_corners=False`
* `mode="nearest"`（6 处）：保持 0/1 门控、索引类信号

实现方式：把插值写成**常量权重矩阵乘**（W[out, in]），于是
* 梯度由 autograd 自动给出（组合式，不写反向）；
* 与 torch 的 `align_corners=False` 公式逐位一致：

    src = (i + 0.5) * in/out - 0.5,  clamp 到 [0, in-1],  取相邻两点线性组合

nearest 用 `floor(i * in/out)`（torch 的 nearest 语义）。
"""
import io

ADD = '''

def _interp_weights(n_in, n_out, mode):
    """返回 [n_out, n_in] 的插值权重矩阵（numpy，常量）。"""
    n_in, n_out = int(n_in), int(n_out)
    key = ("interp", n_in, n_out, mode)
    if key in _CACHE:
        return _CACHE[key]
    W = np.zeros((n_out, n_in), np.float32)
    if mode == "nearest":
        idx = np.floor(np.arange(n_out, dtype=np.float64) * (n_in / n_out)).astype(np.int64)
        idx = np.clip(idx, 0, n_in - 1)
        W[np.arange(n_out), idx] = 1.0
    elif mode == "linear":
        scale = n_in / n_out
        src = (np.arange(n_out, dtype=np.float64) + 0.5) * scale - 0.5
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

'''


def main():
    p = "examples/bundles/svs-ops/svs_ops.py"
    s = io.open(p, encoding="utf-8").read()
    if "def interpolate1d(" in s:
        print("已存在")
        return 0
    anchor = "def cumsum(x, dim=-1):"
    assert anchor in s
    s = s.replace(anchor, ADD.lstrip("\n") + "\n" + anchor, 1)
    s = s.replace('    "hann_window": hann_window,',
                  '    "hann_window": hann_window,\n    "interpolate1d": interpolate1d,')
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    import ast
    ast.parse(io.open(p, encoding="utf-8").read())
    print("已加 interpolate1d 并注册")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
