# -*- coding: utf-8 -*-
"""CUDA 上的数组索引 `x[idx]`（idx 为整型数组 / 列表 / Tensor）。

## 背景

CPU 路径本来就用 `np.add.at` 支持了数组索引（见 `Tensor.__getitem__` 的 `_has_array_index` 分支），
但 CUDA 路径直接抛 `NotImplementedError: 切片暂不支持索引类型 ndarray`。
用户第一次写 embedding 查表 / gather-CE 就会撞上。

## 本次实现的范围（刻意收窄）

* **一维整型索引**，作用在**首轴**上（`x[idx]` / `x[idx, :]`）。
  这是 embedding 与 gather 的绝大多数用法。
* 前向：1 维源用 `gather_flat`，多维源用 `gather_rows`（都是已有融合核）。
* 反向：**主机中转**做 `np.add.at`（正确处理重复下标）。**它比融合核慢**，
  所以文档里明确写「embedding 请用 nn.Embedding」。这是有意的取舍：
  先保证正确、可用，再谈快。

不在本次范围：多维花式索引、布尔掩码（两者本来也不支持）。
"""
import ast
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "python", "tgai_go", "train", "__init__.py")
s = io.open(P, encoding="utf-8").read()

if "_array_index_cuda" in s:
    print("已打过补丁，跳过")
    sys.exit(0)

OLD = '''    def _getitem_cuda(self, idx):
        rt = _cuda()
        out_shape, base, divs, strides = _slice_gather_plan(self.shape, idx)'''

NEW = '''    def _getitem_cuda(self, idx):
        rt = _cuda()
        arr = _int_index_array(idx)
        if arr is not None:
            return self._array_index_cuda(arr)
        out_shape, base, divs, strides = _slice_gather_plan(self.shape, idx)'''

if OLD not in s:
    print("找不到 _getitem_cuda 锚点")
    sys.exit(1)
s = s.replace(OLD, NEW, 1)

# 在 _getitem_cuda 之后插入 _array_index_cuda
ANCHOR = "    def repeat_interleave(self, repeats, dim=None):"
IMPL = '''    def _array_index_cuda(self, arr):
        """首轴整型数组索引的设备实现。`arr` 是**已归一化**（无负数、已查越界）的 int64 数组。

        前向走融合核；反向走主机遍历（`np.add.at` 才能正确处理重复下标）。
        反向比融合核慢，但 correctness 优先；embedding 场景请用 `nn.Embedding`。
        """
        rt = _cuda()
        shape = tuple(int(x) for x in self.shape)
        if not shape:
            raise NotImplementedError(
                "0 维张量不支持数组索引。修法：先 reshape 成至少一维")

        n_src = shape[0]
        tail = shape[1:]
        row_size = 1
        for d in tail:
            row_size *= int(d)
        n_rows = int(arr.size)

        idx_buf = rt.h2d(arr.astype("int32"), dtype=__import__("numpy").int32)
        src = self._cuda_logical_storage()
        if len(tail) == 0:
            buf = rt.gather_flat(src, idx_buf, n_src)
            out_shape = (n_rows,)
        else:
            buf = rt.gather_rows(src, idx_buf, n_src, row_size, n_rows)
            out_shape = (n_rows,) + tail
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad,
                               _parents=(self,))
        if tuple(int(x) for x in out.shape) != out_shape:
            out = out.reshape(out_shape)

        def bw(g, grads):
            if not self.requires_grad:
                return
            # 主机中转：g 可能是多维，先摊平成 (n_rows, row_size)
            g_np = np.asarray(g.to_numpy(), np.float32)
            full = np.zeros(shape, dtype=np.float32)
            if row_size == 1 and len(tail) == 0:
                np.add.at(full, arr, g_np.reshape(-1))
            else:
                np.add.at(full, arr, g_np.reshape((n_rows,) + tail))
            _acc(grads, self, rt.h2d(full))
        out._backward = bw
        return out

    def repeat_interleave(self, repeats, dim=None):'''

if ANCHOR not in s:
    print("找不到 repeat_interleave 锚点")
    sys.exit(1)
s = s.replace(ANCHOR, IMPL, 1)

# 模块级辅助：把 idx 归一化成 int64 ndarray（或返回 None 表示不是数组索引）
HELPER_ANCHOR = "def _has_array_index(idx):"
HELPER = '''def _int_index_array(idx):
    """若不是「单个整型数组充当首轴索引」就返回 None。

    只认：`x[arr]` 或 `x[arr, :]`（arr 是一维整型）。其余形态交给原有切片计划。
    负数按 numpy 语义折回，越界在这里就报错 —— 别让它跑到设备上算成错误位置。
    """
    import numpy as _np

    if isinstance(idx, tuple):
        if len(idx) != 2 or not isinstance(idx[1], slice):
            return None
        if not (idx[1].start is None and idx[1].stop is None
                and idx[1].step is None):
            return None
        idx = idx[0]
    if isinstance(idx, slice) or idx is Ellipsis:
        return None
    if isinstance(idx, list):
        if not idx or not all(isinstance(v, (int, _np.integer)) for v in idx):
            return None
        idx = _np.asarray(idx, dtype=_np.int64)
    if isinstance(idx, Tensor):
        if getattr(idx, "device", "cpu") == "cuda":
            idx = _np.asarray(idx.to_numpy())
        else:
            idx = _np.asarray(idx.data)
    if not isinstance(idx, _np.ndarray):
        return None
    if idx.dtype.kind not in "iu":
        return None
    if idx.ndim != 1:
        return None
    return idx.astype(_np.int64, copy=False)


def _has_array_index(idx):'''

if HELPER_ANCHOR not in s:
    print("找不到 _has_array_index 锚点")
    sys.exit(1)
s = s.replace(HELPER_ANCHOR, HELPER, 1)

# 负数折回与越界检查放在 Tensor 层（需要 shape，helper 里拿不到）
CHECK_ANCHOR = '''        arr = _int_index_array(idx)
        if arr is not None:
            return self._array_index_cuda(arr)'''
CHECK_NEW = '''        arr = _int_index_array(idx)
        if arr is not None:
            n = int(self.shape[0]) if len(self.shape) else 0
            # 负数按 numpy 语义折回；越界立刻报错，不要下发到设备
            arr = np.where(arr < 0, arr + n, arr)
            if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= n):
                raise IndexError(
                    "数组索引越界：取值范围 [0, %d)，收到 [%d, %d]"
                    % (n, int(arr.min()), int(arr.max())) +
                    "。修法：检查索引数组，或确认首轴长度是 %d" % n)
            return self._array_index_cuda(arr)'''
s = s.replace(CHECK_ANCHOR, CHECK_NEW, 1)

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
try:
    ast.parse(s)
    print("补丁已应用，语法 OK")
except SyntaxError as ex:
    print("语法错误:", ex)
    sys.exit(1)
