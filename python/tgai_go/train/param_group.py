# -*- coding: utf-8 -*-
"""共享存储参数 —— 让多个参数共用同一块底层缓冲。

## 为什么需要

有些结构天然需要"一组参数落在同一块连续内存里"：

* **MoE 的专家权重**：`[G, N, K]` 连续存放，一次 `SgemmStridedBatched` 就能算完全部专家；
  若 G 个专家各是独立参数，每次都要先把权重复制成连续缓冲，实测打包开销比省下的还多
  （见 docs/MOE_MODERN_ARCH.md 3.1）。
* 打包后的权重、分片后的 KV cache 等。

## 契约（重要，不遵守会静默出错）

1. **只有声明 `works_inplace=True` 的优化器能用于本组。**
   出式更新（`p.data = p.data - ...`）会把 `.data` 换成新数组、**断开视图**，
   于是"权重更新了、视图还是旧值"。这类 bug 极难查，所以在**构造时**就拒绝。
2. **视图不能单独交给优化器。** 优化器只该拿到 `group.parameters()`（即底层那一块）。
3. **视图是只读语义上的参数**：用来读/用于前向；写只能通过底层 `packed`（或原地写视图）。

## 平台状态（如实）

| 平台 | 状态 |
| --- | --- |
| CPU（numpy） | ✅ **可用**：视图是 numpy 切片，天然共享内存 |
| CUDA | ❌ **暂不支持**：`GpuBuf` 支持共享 `ptr` 的视图，但**没有 offset 字段**，做偏移视图需要新建 `ptr+offset` 的缓冲，会触碰按 ptr 索引的缓存/提交逻辑。**本版明确报错，不静默返回错误结果。** |

CUDA 支持是"专家权重连续存放 + 零打包 → 分组 GEMM 收益"的前置条件，属后续工作。
"""

from __future__ import annotations

import numpy as np

from . import plugin as _plugin

__all__ = ["ParameterGroup", "SharedStorageError"]


class SharedStorageError(Exception):
    """共享存储的契约被违反。"""


def _prod(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


class ParameterGroup:
    """一组共享同一底层缓冲的参数。

    ```python
    g = T.ParameterGroup([(4096, 1024)] * 8)      # 8 个专家权重
    y = x @ g.views[3].T                          # 视图可直接用于前向
    opt = T.optim.AdamW(g.parameters(), lr=1e-3)  # 优化器只拿底层那一块
    ```

    参数
    ----
    shapes : list[tuple[int, ...]]
        每个视图的形状；按顺序紧密排布在同一块缓冲里。
    dtype : numpy dtype
        目前只支持 float32（与 ORV 的数值路径一致）。
    """

    def __init__(self, shapes, dtype=np.float32):
        from . import Parameter

        shapes = [tuple(int(s) for s in sh) for sh in shapes]
        if not shapes:
            raise SharedStorageError(
                "ParameterGroup 至少需要一个形状" +
                "。修法：传 [(4096,1024)] * 8 这样的列表")
        if np.dtype(dtype) != np.float32:
            raise SharedStorageError(
                "ParameterGroup 目前只支持 float32，收到 %s" % np.dtype(dtype) +
                "。修法：用默认 dtype；fp16 视图需要后端支持偏移缓冲，属后续工作")

        self.shapes = shapes
        self.dtype = np.dtype(dtype)
        total = sum(_prod(s) for s in shapes)
        #: 底层那一块——**唯一**该交给优化器的对象
        self._arr = np.zeros(total, dtype=self.dtype)
        self.packed = Parameter(self._arr)

        # 建立视图：numpy 切片天然共享内存
        off = 0
        views = []
        for s in shapes:
            n = _prod(s)
            views.append(Parameter(self._arr[off:off + n].reshape(s)))
            off += n
        #: 各视图（用于前向读取）
        self.views = views

    # ---------- 契约检查 ----------
    def parameters(self):
        """交给优化器的参数列表 —— **只有底层一块**，不含视图。"""
        return [self.packed]

    def optimizer_class(self, name, **kwargs):
        """按名字取优化器并**校验它支持原地更新**；不满足当场报错。

        ```python
        opt = g.optimizer_class("AdamW")(g.parameters(), lr=1e-3)
        ```
        """
        cls = _plugin.require_inplace(name)     # 能力不足会在这里抛
        return cls

    def assert_optimizer_ok(self, opt):
        """检查一个**已实例化**的优化器是否安全用于本组。

        判据：它拿到的参数里不能含视图（视图被单独优化 = 权重与视图会不同步）。
        """
        params = getattr(opt, "params", None) or getattr(opt, "param_groups", None)
        if params is None:
            return True
        try:
            got = list(params)
        except TypeError:
            return True
        view_ids = {id(v) for v in self.views}
        bad = [p for p in got if id(p) in view_ids]
        if bad:
            raise SharedStorageError(
                "优化器拿到了 %d 个**视图**参数，视图不该被单独优化" % len(bad) +
                "。修法：用 group.parameters()（只含底层那一块），"
                "或改用 works_inplace=True 的优化器并只喂 packed")
        return True

    def assert_shared(self):
        """自检：视图确实与底层共享内存（写底层，视图应立刻可见）。"""
        if not self.views:
            return True
        v = self.views[0]
        base = np.asarray(self.packed.data)
        old = float(np.asarray(v.data).ravel()[0])
        base.ravel()[0] = old + 1.0
        try:
            now = float(np.asarray(v.data).ravel()[0])
        finally:
            base.ravel()[0] = old
        if now != old + 1.0:
            raise SharedStorageError(
                "视图与底层不共享内存（写底层后视图没变）" +
                "。修法：确认没有对 .data 做过出式赋值")
        return True

    def __repr__(self):
        return "ParameterGroup(%d 个视图, 共 %d 元素, dtype=%s)" % (
            len(self.views), self._arr.size, self.dtype)
