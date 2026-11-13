# -*- coding: utf-8 -*-
"""`scatter_add` 的 Tensor 层实现（MoE 分派缺的那一块）。

## 运行时的语义（已实测钉住，不是猜的）

`cuda.py::scatter_add_rows(g, idx, n_src_rows, row_size, n_rows)` 的实现是：

```
out = zeros(n_src_rows, row_size)
out[idx[i]] += g[i]      for i in range(n_rows)     # 内核里 atomicAdd
```

也就是说它是 **gather 的反向**。拿它做 `scatter_add`（`out[ids[i]] += values[i]`）时：

* `g` = `values`（源行），`n_rows` = `len(values)`
* `n_src_rows` = **目标表行数**（这里是 MoE 的位置数 N）
* 反向 = `rt.gather_rows(g_out, idx, n_table, d, n_values)` —— 就是取行

两个原语都在运行时里，这里只做 Tensor 包装 + 自定义 VJP
（写法参照 `train/__init__.py::_sdpa_lowmem` 的 `_from_gpu(..., _parents=...)` + `out._backward`）。

## 为什么需要它

实测：`x[ids]` 可用（含反向），但 `x[ids] = v` 不支持，逐元素 `==` 也不返回张量。
所以 MoE 的"每专家只算选中行、再放回原位"这一步，只能靠 scatter_add。
不用它就只能"每专家全算再筛"，算力 ×E —— 那测的就不是 MoE 了。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))


def scatter_add(values, ids, n_rows):
    """`out[ids[i]] += values[i]`，`out` 形状 `(n_rows, d)`；支持 autograd。

    `values`：ORV 张量 `(n_values, d)`；`ids`：长度 `n_values` 的整数序列。
    """
    from tgai_go.train import _acc, _cuda
    import tgai_go.train as T

    if values.device != "cuda":
        raise NotImplementedError(
            "scatter_add 目前只有 CUDA 实现：运行时内核 `scatter_add_rows` 仅 CUDA 可用。"
            "修法：在 GPU 上调用；CPU 对拍请用 `v3_ops.scatter_add_numpy`（同语义）")
    if len(tuple(values.shape)) != 2:
        raise ValueError(
            "scatter_add 的 values 必须是二维 (n_values, d)，收到 %s。"
            "修法：把多出来的维度先 reshape 成二维再调用" % (tuple(values.shape),))

    rt = _cuda()
    idx_np = np.ascontiguousarray(np.asarray(ids).reshape(-1), np.int32)
    n_vals, d = int(values.shape[0]), int(values.shape[1])
    n_rows = int(n_rows)
    if idx_np.size and (idx_np.min() < 0 or idx_np.max() >= n_rows):
        raise ValueError(
            "scatter_add 下标越界：范围 [%d, %d]，n_rows=%d。修法：检查 topk 的专家下标"
            "是否超出 n_experts，或把 n_rows 传对"
            % (int(idx_np.min()), int(idx_np.max()), n_rows))

    ib = rt.h2d(idx_np, dtype=np.int32)
    buf = rt.scatter_add_rows(values._gpu, ib, n_rows, d, n_vals)

    need = bool(getattr(values, "requires_grad", False))
    out = T.Tensor._from_gpu(buf, requires_grad=need, _parents=(values,) if need else ())
    if not need:
        return out

    def bw(g, grads):
        # 反向 = 按 ids 取行（gather_rows 语义：out[i] = src[idx[i]]）
        _acc(grads, values, rt.gather_rows(g, ib, n_rows, d, n_vals))

    out._backward = bw
    return out
