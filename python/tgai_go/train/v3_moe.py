# -*- coding: utf-8 -*-
"""真 MoE 分派：按专家分组 → 只算该专家的 token → 散回原位（K=1）。

## 为什么这样实现

实测约束：取行可用（含反向）、赋值索引不可用。于是"只算选中行"必须配一个 scatter 才能放回，
而 `v3_scatter.scatter_add` 已经把那个能力补上了（运行时 `scatter_add_rows`，V100 零误差验证过）。

流程（每个 token 只激活 1 个专家时）：

```
eids  = topi[:, 0]                      # 每 token 的专家号
order = argsort(eids)                    # 按专家分组（稳定排序）
xs    = x[order]                         # 取行（gather，可反传）
对每个专家 e：rows = xs[a:b] → y_e = situ_glu(rows, W1[e], W2[e], W3[e])
y_s   = concat([y_0, y_1, ...])          # 顺序与 xs 一致
gate_s= gate[order]                      # 门控跟着排序
y     = scatter_add(y_s * gate_s, order, N)   # 放回原位
```

这样**算力与稀疏性都正确**：每个专家只处理自己的 token，没有"全算再筛"的 ×E 放大。

## 与 PyTorch 参考实现的对应关系（读源码确认，`model_v3.py::MoEv3.forward` ④）

| 参考实现（PyTorch） | 这里（ORV） |
| --- | --- |
| `x_dup = xf.repeat_interleave(top_k, dim=0)` | 本文件先做 K=1 |
| `sorter = torch.argsort(ids_flat, stable=True)` | `np.argsort(eids, kind="stable")` |
| `x_sorted = x_dup.index_select(0, sorter)` | `xs = x[order]`（int 数组索引） |
| 逐专家 `experts[eid](x_sorted[start:end])` | 同（切片 + SiTU-GLU） |
| `index_copy(0, sorter, sorted_out)` 散回 | `scatter_add(y_sorted*g, order, N)` |
| `out = out_flat.view(N, top_k, D).sum(1)` | K>1 时按槽位累加 |

参考实现注释里特别提过：散回要传 **sorter** 本身（不是逆置换），传错时 19/20 行都错（max|Δ|=9.8）。
这里的 `scatter_add` 语义是 `out[order[i]] += v[i]`，与 `index_copy` 一致 —— 同一坑。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))


def _situ_np(x, w1, w2, w3, beta1=4.0, beta2=25.0):
    """SiTU-GLU 的 numpy 实现（**本模块自带**，不依赖别的函数）。

    与参考实现 `model_v3.py::SiTUGLUFFN.forward` 同式：
        gate = (β₁·tanh(g/β₁)) ⊙ sigmoid(g);  up = β₂·tanh(u/β₂);  y = w3(gate ⊙ up)
    """
    x = np.asarray(x, np.float32)
    g = x @ np.asarray(w1, np.float32).T
    u = x @ np.asarray(w2, np.float32).T
    gate = (beta1 * np.tanh(g / beta1)) * (0.5 * (1.0 + np.tanh(0.5 * g)))
    up = beta2 * np.tanh(u / beta2)
    return (gate * up) @ np.asarray(w3, np.float32).T


def route_top1(hn, router_w):
    """top_k=1 的路由：`eids`（主机下标，仅用于分派）+ `gate`（保留梯度）。

    ## 为什么不用 `T.topk`

    实测：`T.topk` 在 CUDA 上**未实现**（只有 CPU 路径）：
    `NotImplementedError: CUDA topk 尚未实现`。

    top_k=1 时有等价替代：
        s_all = softmax(router(x));  gate = max(s_all, dim=-1)   ← 梯度流到最大项
        eids  = argmax(s_all)                                     ← 离散下标，不需梯度
    与 `topk(...,1)` 数学一致，且不需要 topk。
    """
    import tgai_go.train as T
    z = hn @ router_w.transpose(0, 1)
    s_all = T.functional.softmax(z, axis=-1)
    gate = s_all.max(axis=-1)
    sv = s_all.to_numpy() if hasattr(s_all, "to_numpy") else s_all
    eids = np.argmax(np.asarray(sv), axis=-1).reshape(-1).astype(np.int64)
    return eids, gate, s_all


def moe_dispatch_numpy(x, eids, gate, W1, W2, W3, beta1=4.0, beta2=25.0):
    """判定基准：逐专家对选中行算 SiTU-GLU，再按 order 散回原位。"""
    x = np.asarray(x, np.float32)
    eids = np.asarray(eids).reshape(-1).astype(np.int64)
    gate = np.asarray(gate, np.float32).reshape(-1)
    out = np.zeros_like(x)
    for e in range(np.asarray(W1).shape[0]):
        pos = np.flatnonzero(eids == e)
        if pos.size == 0:
            continue
        y = _situ_np(x[pos], W1[e], W2[e], W3[e], beta1, beta2)
        out[pos] += y * gate[pos].reshape(-1, 1)
    return out


def moe_dispatch_orv(x, eids_np, gate, W1, W2, W3, beta1=4.0, beta2=25.0):
    """ORV 张量版：gather → 逐专家 → cat → scatter_add 回原位。

    `eids_np` 是 `topi[:, 0]` 的 numpy 视图（路由下标本身不参与梯度，这是标准做法）。
    """
    import tgai_go.train as T
    from tgai_go.train import v3_ops as V
    from tgai_go.train import v3_scatter as S

    n_rows = int(x.shape[0])
    eids = np.asarray(eids_np).reshape(-1).astype(np.int64)
    order = np.argsort(eids, kind="stable")           # 按专家分组
    xs = x[order]                                     # gather（可反传）
    g_sorted = gate[order]

    E = int(W1.shape[0])
    counts = np.bincount(eids, minlength=E)
    parts, bounds = [], np.cumsum(counts)
    start = 0
    for e in range(E):
        end = int(bounds[e])
        if end > start:
            rows = xs[start:end]                      # 只取该专家的 token
            # ★ `W1[e]` 本身就是张量（对三维张量取整数下标），不要再包 T.tensor ——
            #   包了会对张量调 np.ascontiguousarray，报 "setting an array element with a sequence"
            parts.append(V.situ_glu_orv(rows, W1[e], W2[e], W3[e], beta1, beta2))
        start = end
    if not parts:
        return x * 0.0
    y_sorted = parts[0] if len(parts) == 1 else T.cat(parts, dim=0) if hasattr(T, "cat") else None
    if y_sorted is None:                              # T.cat 是模块级函数
        from tgai_go.train import cat as _cat
        y_sorted = _cat(parts, dim=0)
    # ★ ORV 的切片不支持 `None`（newaxis），必须用 reshape 造列向量（实测报错：切片暂不支持 None）
    return S.scatter_add(y_sorted * g_sorted.reshape(-1, 1), order, n_rows)
