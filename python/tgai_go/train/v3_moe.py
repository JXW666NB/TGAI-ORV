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


def route_topk(hn, router_w, k=2, bias=None):
    """top_k=k 的路由，语义严格对齐参考实现 `model_v3.py:1039-1057`：

        s_all = softmax(router(x), dim=-1)        # 对**全体**专家做 softmax
        sel   = s_all + bias                       # 偏置**只影响选择**，不进门控
        topi  = top-k(sel)                         # 每 token 取前 k
        gate  = s_all[topi]                        # 门控用**原始亲和度**，不重新归一化

    实现照 PyTorch 的 C 端思路（`aten/src/ATen/native/cuda/TensorTopK.cu`）：
    小切片不物化排序、不用 mask，**一个内核里每行一个线程完成选择**，
    反向等价于 `gather` 的反向（把门控梯度散射回原位，其余为 0）。

    返回 `(eids, gate2, s_all)`：
      * `eids`  (N, k) int64 主机下标 —— 只用于分派，不参与求导
      * `gate2` (N, k) 张量 —— 选中的原始亲和度，**带梯度**
      * `s_all` (N, E) 张量 —— 供调用方复用（z-loss 等）
    """
    import tgai_go.train as T
    rt = T._cuda()
    z = hn @ router_w.transpose(0, 1)
    s_all = T.functional.softmax(z, axis=-1)
    n, E = int(s_all.shape[0]), int(s_all.shape[1])
    b = (np.zeros(E, np.float32) if bias is None
         else np.asarray(bias, np.float32).reshape(E))
    bias_t = T.tensor(b)
    # 选择用 (s + bias)，门控取原始 s —— 两个现成内核：
    #   `moe_topk`       每个 token 的 top-k（内核内完成，不物化排序）
    #   `moe_gather_pairs` 按 idx 取 gate；反向用 `moe_scatter_pairs` 散射回原位
    sel = s_all + bias_t
    idx_buf, _val = rt.moe_topk(sel._gpu, int(k))
    gate_buf = rt.moe_gather_pairs(s_all._gpu, idx_buf, n, E, int(k))
    eids = np.asarray(idx_buf.to_numpy() if hasattr(idx_buf, "to_numpy") else idx_buf)
    eids = eids.astype(np.int64).reshape(n, -1)

    need = bool(s_all.requires_grad)
    gates = T.Tensor._from_gpu(gate_buf, requires_grad=need,
                               _parents=(s_all,) if need else ())
    if need:
        def bw(g, grads):
            ds = rt.moe_scatter_pairs(g, idx_buf, n, E, int(k))
            T._acc(grads, s_all, ds)

        gates._backward = bw
    return eids, gates, s_all


def moe_topk_orv(x, eids, gates, W1, W2, W3, beta1=4.0, beta2=25.0,
                 merged=True):
    """top_k>1 的分派。

    **合并路径（默认）**：把 (token, slot) 展平成 N*k 行，**一次**分派覆盖全部专家，
    再对 slot 维求和。相对逐槽分派的收益：
      · 专家权重只读一遍（原来是 k 遍）
      · 每个专家的 GEMM 行数从 N/E 变成 k·N/E，小 GEMM 的利用率显著改善
      · GEMM 发射次数从 k·E·3 降到 E·3
    数学上等于 `Σ_k gate_k · FFN_{i_k}(x)`，与参考实现一致；只有求和顺序变了，
    实测与逐槽路径的差异在 1e-6 量级（见 tools/probe_moe_merged.py）。

    `merged=False` 保留旧的逐槽路径，用于对拍与回退。
    """
    N, k = int(eids.shape[0]), int(eids.shape[1])
    import os as _os
    if _os.environ.get("TG_MOE_MERGED", "").strip().lower() in ("0", "false", "off"):
        merged = False
    if k == 1 or not merged:
        out = None
        for slot in range(k):
            part = moe_dispatch_orv(x, eids[:, slot], gates[:, slot], W1, W2, W3,
                                    beta1, beta2)
            out = part if out is None else out + part
        return out
    d = int(x.shape[1])
    tok = np.repeat(np.arange(N), k)                 # 每行的源 token
    e_flat = np.asarray(eids).reshape(N * k)
    g_flat = gates.reshape(N * k)                    # 每行的门控值（设备张量）
    # ★ 2026-11-15：把"先按 token 展开再按专家排序"两次 gather 合成**一次**。
    #   `x[tok]` 后再 `[order]` 等于 `x[tok[order]]`；预合成置换后只搬一趟，
    #   并省掉一份 (N·k, d) 的中间张量（实测前向因此从 0.125 s 降到更低）。
    from tgai_go.train import indexing as _idx
    perm = tok[np.argsort(e_flat, kind="stable")]
    x_perm = _idx.index_select(x, 0, perm)
    y = moe_dispatch_orv(x_perm, e_flat, g_flat, W1, W2, W3, beta1, beta2,
                         _presorted=perm)
    return y.reshape(N, k, d).sum(axis=1)


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


def moe_dispatch_orv(x, eids_np, gate, W1, W2, W3, beta1=4.0, beta2=25.0,
                     _presorted=None):
    """ORV 张量版：gather → 逐专家 → cat → scatter_add 回原位。

    `eids_np` 是 `topi[:, 0]` 的 numpy 视图（路由下标本身不参与梯度，这是标准做法）。
    `_presorted`：调用方**已经**把 x 按专家顺序 gather 好了时，传入那次 gather 用的
    置换 `perm`；此时内部跳过 gather，只保留"散回原位"那一步（用 perm 作索引）。
    """
    import tgai_go.train as T
    from tgai_go.train import v3_ops as V
    from tgai_go.train import v3_scatter as S

    n_rows = int(x.shape[0])
    eids = np.asarray(eids_np).reshape(-1).astype(np.int64)
    if _presorted is not None:
        order = np.asarray(_presorted, np.int64)      # 只用于散回原位
        xs = x
        g_sorted = gate
    else:
        order = np.argsort(eids, kind="stable")       # 按专家分组
        # ★ 2026-11-15：改用 `indexing.index_select`（设备侧 gather，反向是
        #   `rt.scatter_add_rows`）。原来的 `x[order]` 反向走 **D2H + np.add.at + H2D**，
        #   实测这一段 fwd+bwd 要 **25 ms**（整个分派 62 ms 里最大的一块）。
        from tgai_go.train import indexing as _idx
        xs = _idx.index_select(x, 0, order)
        # gate 是 1-D 且实测只值 0 ms，仍用原来的索引（设备 gather 对 1-D 的反向形状会出问题）
        g_sorted = gate[order]

    # 专家权重可以是三维张量（W1[e] 会拷贝）或**列表**（不拷贝，推荐）。
    E = len(W1) if isinstance(W1, (list, tuple)) else int(W1.shape[0])
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
