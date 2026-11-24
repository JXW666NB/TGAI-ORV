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


# ═════════════════════════════════════════════════════════════════════════
# 设备端分派的开关与"设备下标句柄"（TG_MOE_DEV，默认关）
# ═════════════════════════════════════════════════════════════════════════

def moe_dev_enabled():
    """`TG_MOE_DEV` 是否打开设备端分派（默认关）。

    只认 `1/true/on/yes`（大小写不敏感），其余一律当**关**：
    环境变量拼错时宁可退回已验证的旧路径，也不让一次训练启动失败。
    回滚方式：删掉该环境变量（或置 0）即可，不涉及代码回退。
    """
    v = os.environ.get("TG_MOE_DEV", "")
    return str(v).strip().lower() in ("1", "true", "on", "yes")


class DeviceRouteIdx:
    """`route_topk(..., return_idx_buf=True)` 的返回值：**设备端** top-k 下标句柄。

    为什么需要它：分派只需要下标去做分组，不需要它们出现在主机上；旧路径每层
    为此做一次 `int32[N,k]` 的阻塞 D2H（`idx_buf.to_numpy()`）。把 `GpuBuf` 装进
    这个小对象后，`moe_topk_orv` 可以直接消费它（走 `moe_compact` + 设备端分组），
    而任何仍然需要主机下标的旧代码（例如辅助损失里的
    `np.bincount(np.asarray(eids))`）会在 `to_numpy()`/`np.asarray()` 时**按需**
    做那一次 D2H —— 语义与旧路径完全一致，只是没有白做。

    `buf` 是 `rt.moe_topk` 产出的 `int32[N,k]` 设备缓冲（不在这里复制）。
    """

    __slots__ = ("buf", "n", "k")

    def __init__(self, buf, n, k):
        self.buf = buf
        self.n = int(n)
        self.k = int(k)

    @property
    def shape(self):
        return (self.n, self.k)

    @property
    def device(self):
        return "cuda"

    def to_numpy(self, dtype=np.int64):
        """把下标取回主机（`int64[N,k]`，与旧路径 `eids` 的 dtype/shape 对齐）。

        只在"确实需要主机下标"的地方调用；设备端分派路径不会调它。
        """
        import tgai_go.train as T
        rt = T._cuda()
        arr = np.asarray(rt.d2h(self.buf), dtype=dtype)
        return arr.reshape(self.n, self.k)

    def __array__(self, dtype=None, copy=None):     # numpy 2.x 会带 copy=
        arr = self.to_numpy()
        return arr.astype(dtype) if dtype is not None else arr

    def __repr__(self):
        return "DeviceRouteIdx(n=%d, k=%d)" % (self.n, self.k)


def route_topk(hn, router_w, k=2, bias=None, return_idx_buf=None):
    """top_k=k 的路由，语义严格对齐参考实现 `model_v3.py:1039-1057`：

        s_all = softmax(router(x), dim=-1)        # 对**全体**专家做 softmax
        sel   = s_all + bias                       # 偏置**只影响选择**，不进门控
        topi  = top-k(sel)                         # 每 token 取前 k
        gate  = s_all[topi]                        # 门控用**原始亲和度**，不重新归一化

    实现照 PyTorch 的 C 端思路（`aten/src/ATen/native/cuda/TensorTopK.cu`）：
    小切片不物化排序、不用 mask，**一个内核里每行一个线程完成选择**，
    反向等价于 `gather` 的反向（把门控梯度散射回原位，其余为 0）。

    返回 `(eids, gate2, s_all)`：
      * `eids`  (N, k) int64 **主机**下标 —— 只用于分派，不参与求导；
        `return_idx_buf=True` 时改为返回 `DeviceRouteIdx`（下标留在设备上）
      * `gate2` (N, k) 张量 —— 选中的原始亲和度，**带梯度**
      * `s_all` (N, E) 张量 —— 供调用方复用（z-loss 等）

    ## `return_idx_buf`（`TG_MOE_DEV` 的门）

    `True` ⇒ 不做 `idx_buf.to_numpy()`（省掉每层一次阻塞 D2H），返回
    `DeviceRouteIdx`；`moe_topk_orv` 认出它就走设备端分派（`moe_topk_dev`）。
    默认 `None` ⇒ 读环境变量 `TG_MOE_DEV`（默认关，行为与以前逐位一致）。
    注意：`eids` 从 `int64 (N,k)` 变成对象后，**仍**支持 `np.asarray(eids)` /
    `eids.to_numpy()`（两者都会按需做那一次 D2H），旧代码无需改。
    """
    import tgai_go.train as T
    rt = T._cuda()
    if return_idx_buf is None:
        return_idx_buf = moe_dev_enabled()
    if return_idx_buf and getattr(hn, "device", "cpu") != "cuda":
        raise NotImplementedError(
            "设备端路由句柄只有 CUDA 路径有（route_topk 本身也只支持 CUDA）。"
            "修法：把 hn 放到 GPU，或设 TG_MOE_DEV=0 走主机下标路径")
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
    if return_idx_buf:
        eids = DeviceRouteIdx(idx_buf, n, int(k))
    else:
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
    # ★ 设备端分派（TG_MOE_DEV）：`route_topk` 传下来的是 `DeviceRouteIdx`
    #   而不是主机 numpy 下标 ⇒ 走 `moe_topk_dev`（每层只读回 E+1 个偏移）。
    #   k==1 与非 merged 回退需要主机下标（逐槽循环），就地物化一次再走旧路径。
    import os as _os
    if _os.environ.get("TG_MOE_MERGED", "").strip().lower() in ("0", "false", "off"):
        merged = False
    if isinstance(eids, DeviceRouteIdx):
        if eids.n != int(x.shape[0]):
            raise ValueError(
                "路由下标句柄的 token 数 %d 与分派输入行数 %d 不一致。"
                "修法：确认 route_topk 与 moe_topk_orv 吃的是同一个 hn"
                % (eids.n, int(x.shape[0])))
        if eids.k > 1 and merged:
            return moe_topk_dev(x, eids, gates, W1, W2, W3, beta1, beta2)
        eids = eids.to_numpy()                       # 回退旧路径，需主机下标
    N, k = int(eids.shape[0]), int(eids.shape[1])
    if k == 1 or not merged:
        out = None
        for slot in range(k):
            part = moe_dispatch_orv(x, eids[:, slot], gates[:, slot], W1, W2, W3,
                                    beta1, beta2)
            out = part if out is None else out + part
        return out
    d = int(x.shape[1])
    tok = np.repeat(np.arange(N), k)                 # 每行的源 token
    from tgai_go.train import indexing as _idx
    # ★ 2026-11-15：**外层 token 展开也必须走设备 gather**。
    #   原来这里写 `x[tok]`，前向只要 0.07 ms，但它的反向是
    #   `D2H + np.add.at + H2D`（主机往返）—— 实测整条分派前向 3.26 ms、
    #   前+反向 26.3 ms，也就是反向 23 ms 里绝大部分是这一行。
    #   换成 index_select 后反向是 `rt.scatter_add_rows`（设备原子加）。
    x_rows = _idx.index_select(x, 0, tok)
    e_flat = np.asarray(eids).reshape(N * k)
    g_flat = gates.reshape(N * k)                    # 每行的门控值（设备张量）
    y = moe_dispatch_orv(x_rows, e_flat, g_flat, W1, W2, W3, beta1, beta2)
    from tgai_go.train import v3_scatter as S
    # ★ 2026-11-15：`moe_dispatch_orv` 返回的是**按输入行序**（即 (token,slot) 序）的结果，
    #   所以再按 `tok`（行 → token 号）散进 N 行，原子加天然完成 k 路求和，
    #   省掉原来的 reshape(N,k,d) + sum(axis=1)（多一趟 8 MB 读写与一份中间张量）。
    return S.scatter_add(y, tok, N)


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


# ═══════════════════════════════════════════════════════════════════════════
# 设备端分派（`TG_MOE_DEV=1`）：分组/计数留在设备上，主机每层只读回 E+1 个偏移
# ═══════════════════════════════════════════════════════════════════════════
#
# 旧路径（`moe_topk_orv` merged + `moe_dispatch_orv`）每层有**两处**主机往返：
#   ① `route_topk` 里 `idx_buf.to_numpy()` —— 把 top-k 下标读回主机做 argsort；
#   ② 反向 `gate[order]` 的 1-D 数组索引兜底 —— 走 `D2H + np.add.at + H2D`。
# 这两处都是为了"主机侧分组"，不是数学需要。本节的路径用运行时现成的
# `moe_compact` 在**设备上**完成"计数 + 稳定分组"，主机只读回 `int32[E+1]` 偏移
# （E=4 时 20 字节），然后按专家循环调设备原语，全程不再碰下标。
#
# 数值为什么能逐位一致（确定性模式下）：
#   * `moe_compact` 的分组顺序与旧路径 `np.argsort(eids, kind="stable")` **完全一致**
#     （专家升序；专家内按扁平号 t*k+s 升序）⇒ 每个专家的行集合与行序都不变；
#   * GEMM / SiTU-GLU / 门控乘 / 梯度都直接复用旧路径同一批 Tensor 算子，
#     连反向 kernel 与累加顺序都不变（只有"累加器"这一层是手写的）；
#   * `k=2` 时同一 token 的两条贡献来自两个不同专家，跨专家按升序做两次
#     `atomicAdd`，而 IEEE 浮点加法满足 a+b == b+a ⇒ 与旧路径的"槽位升序"累加
#     逐位相同（k>2 的三项以上求和顺序不保证，见 README_MOE_DEV.md 的风险段）。

def _dev_gather_rows(x, tok_buf, begin, m, d):
    """`rt.moe_gather_rows` 的张量包装（可反传：反向 = 把梯度散回源行）。

    语义：`out[i] = x[tok_buf[begin+i]]`，`i in [0, m)`。
    `tok_buf` 是 `rt.moe_compact` 给出的 `int32[N*k]` token 号设备缓冲。
    """
    import tgai_go.train as T
    rt = T._cuda()
    buf = rt.moe_gather_rows(x._gpu, tok_buf, int(begin), int(m), int(d))
    need = bool(x.requires_grad)
    out = T.Tensor._from_gpu(buf, requires_grad=need, _parents=(x,) if need else ())
    if not need:
        return out

    def bw(g, grads):
        if x.requires_grad:
            T._acc(grads, x, rt.moe_scatter_rows(
                g, tok_buf, int(begin), int(m), int(x.shape[0]), int(d)))

    out._backward = bw
    return out


def _dev_gather_slots(w, tok_buf, slot_buf, begin, m, k):
    """按 (token, slot) 从门控表 `w[N,k]` 里取出行门控（反向 = 设备端散射回原位）。

    与旧路径 `gate[order]` 取到的是**同一批值**；区别是反向不再走
    `__init__.py::_array_index_cuda` 的 1-D 主机兜底（那正是每层第二次 D2H）。
    """
    import tgai_go.train as T
    rt = T._cuda()
    n = int(w.shape[0])
    wbuf = w._gpu
    if tuple(int(s) for s in wbuf.shape) != (n, int(w.shape[1])):
        raise ValueError(
            "设备端门控取值要求 w 的设备缓冲是连续二维 [N,k]（收到 %s）。"
            "修法：让 w 是连续张量（`T.ascontiguous`/不产生非连续视图），"
            "或设 TG_MOE_DEV=0 回退旧的逐专家分派路径"
            % (tuple(int(s) for s in wbuf.shape),))
    buf = rt.moe_gather_slots(wbuf, tok_buf, slot_buf,
                              int(begin), int(m), int(k))
    need = bool(w.requires_grad)
    out = T.Tensor._from_gpu(buf, requires_grad=need, _parents=(w,) if need else ())
    if not need:
        return out

    def bw(g, grads):
        if w.requires_grad:
            T._acc(grads, w, rt.moe_scatter_slots(
                g, tok_buf, slot_buf, int(begin), int(m), n, int(k)))

    out._backward = bw
    return out


def moe_dispatch_dev(x, idx_dev, gates, W1, W2, W3, beta1=4.0, beta2=25.0):
    """设备端驱动的一次分派：`out[t] = Σ_slot gate[t,slot] · FFN_{e(t,slot)}(x[t])`。

    与 `moe_dispatch_orv`（主机 `np.argsort` + `np.bincount` + 逐专家切片）数学等价，
    差别只在**谁做分组**：

    ```
    off, tok, slot = rt.moe_compact(idx_buf, N, E, k)    # 设备端计数 + 稳定分组
    off_host = d2h(off)                                   # ★ 唯一一次 D2H：E+1 个 int32
    for e in 0..E-1:                                      # 空专家直接跳过
        rows   = moe_gather_rows(x, tok, off[e], cnt_e, d)
        gate_e = moe_gather_slots(gates, tok, slot, off[e], cnt_e, k)
        y_e    = situ_glu(rows) ⊙ gate_e                  # 与旧路径同一批设备算子
        moe_scatter_rows_add(y_e, tok, off[e], cnt_e, acc) # 直接累加进 [N,d]
    ```

    反向：
      * GEMM / SiTU-GLU / 门控乘全部是**旧路径同一批 Tensor 算子**的自动微分
        ⇒ dW1/dW2/dW3、dx、dgate 的 kernel 与累加顺序不变；
      * 只有"累加器"这一层是手写的：`d(y_e) = gather_rows(g, tok, off[e], cnt_e, d)`。
      * 路由下标不参与求导（与旧路径一致）。

    `idx_dev` 是 `DeviceRouteIdx`（`int32[N,k]` 设备缓冲），`gates` 是 `[N,k]` 设备张量。
    返回 `[N,d]` 张量；token 数必须等于 `x.shape[0]`。
    """
    import tgai_go.train as T
    from tgai_go.train import v3_ops as V

    rt = T._cuda()
    n_tok, k = int(x.shape[0]), int(idx_dev.k)
    d = int(x.shape[1])
    if idx_dev.n != n_tok:
        raise ValueError(
            "设备端分派的 token 数不一致：idx=%d, x=%d。修法：两者必须来自同一个 hn"
            % (idx_dev.n, n_tok))
    E = len(W1) if isinstance(W1, (list, tuple)) else int(W1.shape[0])
    if int(gates.shape[0]) != n_tok or int(gates.shape[1]) != k:
        raise ValueError(
            "门控形状 %s 与 (token=%d, top_k=%d) 不一致。修法：gates 必须是 route_topk 的返回"
            % (tuple(int(s) for s in gates.shape), n_tok, k))
    # `moe_gather_rows` 直接按 (ptr, 行号, row_stride=d) 取行，要求 x 的设备缓冲就是
    # 连续 [N,d]。惰性转置（`.T`）之类会在这里被拦下，而不是静默读错内存。
    xbuf = x._gpu
    if tuple(int(s) for s in xbuf.shape) != (n_tok, d):
        raise ValueError(
            "设备端分派要求 x 的设备缓冲是连续二维 [N,d]（收到 %s，逻辑 shape=%s）。"
            "修法：先 `x.reshape(%d, %d)` 或把惰性转置物化后再调用"
            % (tuple(int(s) for s in xbuf.shape), (n_tok, d), n_tok, d))

    # ① 设备端计数 + 稳定分组（顺序与 np.argsort(eids, kind="stable") 一致）
    off_buf, tok_buf, slot_buf = rt.moe_compact(idx_dev.buf, n_tok, E, k)
    # ② 唯一一次主机回读：E+1 个 int32。空专家跳过、逐专家 GEMM 的行数都靠它。
    off = np.asarray(rt.d2h(off_buf), dtype=np.int32).reshape(-1)

    # ③ 逐专家：gather → SiTU-GLU → 门控 → 直接累加进 [N,d]
    acc_buf = rt.zeros((n_tok, d), dtype=np.float32)
    parts = []
    for e in range(E):
        begin = int(off[e])
        count = int(off[e + 1]) - begin
        if count <= 0:
            continue
        rows = _dev_gather_rows(x, tok_buf, begin, count, d)
        y = V.situ_glu_orv(rows, W1[e], W2[e], W3[e], beta1, beta2)
        ww = _dev_gather_slots(gates, tok_buf, slot_buf, begin, count, k)
        y = y * ww.reshape(-1, 1)
        rt.moe_scatter_rows_add(y._gpu, tok_buf, begin, count, acc_buf)
        parts.append((y, begin, count))

    if not parts:                                    # 只可能是 N==0（正常配置到不了）
        return x * 0.0
    need = any(p[0].requires_grad for p in parts)
    out = T.Tensor._from_gpu(acc_buf, requires_grad=need,
                             _parents=tuple(p[0] for p in parts) if need else ())
    if need:
        def bw(g, grads, _parts=tuple(parts), _tok=tok_buf, _d=d):
            # 每个专家的梯度 = 从整体梯度里把它的那几行取回来（与旧路径等价）
            for y_e, b0, cnt in _parts:
                if y_e.requires_grad:
                    T._acc(grads, y_e,
                           rt.moe_gather_rows(g, _tok, b0, cnt, _d))

        out._backward = bw
    # 前向用过的 (tok, slot, off) 与逐专家输出必须活到反传（没有反传时也要活到
    # 散射 kernel 真正跑完 —— 池是 CPU 侧记账，提前还池会撞上还没执行的 kernel）。
    out._gpu.keepalive = (parts, tok_buf, slot_buf, off_buf)
    return out


def moe_topk_dev(x, idx_dev, gates, W1, W2, W3, beta1=4.0, beta2=25.0):
    """`moe_topk_orv(merged=True)` 的设备端版本（top_k>1，`TG_MOE_DEV=1`）。

    旧 merged 路径是：把 (t,s) 展平成 N*k 行 → `index_select(x, tok)` →
    逐专家算 → 再按 `scatter_add(y, tok, N)` 把 k 路加起来。
    设备端版本省掉两层主机索引：直接按 `moe_compact` 的 (tok,slot) 从 **x 本身**
    gather，逐专家算完直接 `moe_scatter_rows_add` 累加进 `[N,d]`
    —— 同一 token 的 k 条贡献由原子加天然求和，不需要中间的 N*k 行缓冲。

    D2H 次数：每层 **1 次**（E+1 个 int32），旧路径是 2 次
    （route_topk 的 idx + 反向 `gate[order]` 的梯度兜底）。
    """
    return moe_dispatch_dev(x, idx_dev, gates, W1, W2, W3, beta1, beta2)

