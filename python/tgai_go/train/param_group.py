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
        #: 权威主机副本就是 `_arr`（不是 `packed.data`）。
        #: 构造期间**强制 CPU**：若进程默认设备已是 cuda，`Parameter(_arr)` 会立刻
        #: 在设备上再建一份（实测过：那份是全零，之后改主机数据不会同步），
        #: 既白占显存，又会让 to_cuda 读到陈旧内容。
        import sys as _sys
        _pk = _sys.modules.get("tgai_go.train")
        _prev = getattr(_pk, "_DEVICE", "cpu")
        try:
            if _pk is not None:
                _pk._DEVICE = "cpu"
            self.packed = Parameter(self._arr)
            off = 0
            views = []
            for s in shapes:
                n = _prod(s)
                views.append(Parameter(self._arr[off:off + n].reshape(s)))
                off += n
        finally:
            if _pk is not None:
                _pk._DEVICE = _prev
        #: 各视图（用于前向读取）
        self.views = views
        #: 当前设备（"cpu" / "cuda"），由 to_cuda() 更新
        self._device = "cpu"

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

    def load(self, array):
        """把数据**原地**写进底层缓冲（保持视图有效），并同步到 GPU（若已上设备）。

        为什么必须用这个方法而不是 `group.packed.data = ...`：
        `.data` 赋值是**出式**的，会把底层换成新数组、断开与视图的共享
        （这正是 §契约 第 1 条要防的事）。`load` 用切片赋值，原地写，共享不断。

        参数
        ----
        array : 可广播到 `(total,)` 的数组
        """
        src = np.ascontiguousarray(np.asarray(array), dtype=np.float32).reshape(-1)
        if src.size != self._arr.size:
            raise SharedStorageError(
                "load 的元素数不对：给 %d，整组共 %d" % (src.size, self._arr.size) +
                "。修法：传入与 shapes 总元素数一致的数组（或用 load_shaped 按视图分别给）")
        self._arr[:] = src
        if getattr(self, "_device", "cpu") == "cuda" and getattr(self.packed, "_gpu", None) is not None:
            from . import _cuda

            rt = _cuda()
            base = rt.h2d(self._arr)
            self.packed._gpu = base
            off = 0
            for v in self.views:
                n = _prod(v.shape)
                v._gpu = rt.sub_view(base, v.shape, off)
                off += n
        return self

    def to_cuda(self):
        """把整组搬到 GPU：视图在 GPU 上也是**共享同一段显存的偏移视图**。

        实现靠 `CudaRuntime.sub_view`（在底层缓冲内做偏移视图）。这样
        「专家权重连续存放 + 一次 strided GEMM」才可能成立 —— 不需要每次重排权重。

        注意契约不变：**优化器只能拿 `packed`，不能拿视图**；
        只有 `works_inplace=True` 的优化器允许用于本组（视图靠共享内存同步，
        出式更新会断开）。
        """
        from . import _cuda

        rt = _cuda()
        # **权威副本是 `self._arr`**，每次都从它上传。
        # 不要"复用已有的 `packed._gpu`"：主机副本可能已被改动（测试里就是），
        # 复用陈旧设备缓冲会读到旧值 —— 实测踩过，偏差 1.3e+02。
        host = np.ascontiguousarray(self._arr, dtype=np.float32).reshape(-1)
        base = rt.h2d(host)
        self.packed._gpu = base
        off = 0
        for v in self.views:
            n = _prod(v.shape)
            v._gpu = rt.sub_view(base, v.shape, off)
            off += n
        self._device = "cuda"
        return self

    def assert_shared_gpu(self):
        """GPU 侧自检：每个视图的设备指针必须**正好是底层指针 + 元素偏移**。

        这是结构性检查（不猜任何运行时 API），能直接证明"视图确实落在同一块显存里、
        且位置正确"。数值层面的共享由 `tools/param_group_gpu_check.py` 用真实算子验证。
        """
        if getattr(self, "_device", "cpu") != "cuda":
            return True
        base = self.packed._gpu
        if base is None:
            return True
        esz = int(base.dtype.itemsize)
        off = 0
        for v in self.views:
            n = _prod(v.shape)
            vb = v._gpu
            if vb is None:
                raise SharedStorageError(
                    "视图 %r 没有设备缓冲" % (v.shape,) +
                    "。修法：先调 group.to_cuda() 再访问 GPU 侧视图")
            want = int(base.ptr) + off * esz
            if int(vb.ptr) != want:
                raise SharedStorageError(
                    "视图 %r 的设备指针 %d != 期望 %d（底层 %d + 偏移 %d 字节）"
                    % (v.shape, int(vb.ptr), want, int(base.ptr), off * esz) +
                    "。修法：检查切分顺序是否与构造时一致")
            off += n
        return True

    def __repr__(self):
        return "ParameterGroup(%d 个视图, 共 %d 元素, dtype=%s)" % (
            len(self.views), self._arr.size, self.dtype)
