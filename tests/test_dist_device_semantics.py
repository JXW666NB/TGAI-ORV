# -*- coding: utf-8 -*-
"""`dist` 的设备语义：**CUDA 张量的 `.data` 是 None** 这件事的回归测试。

## 背景（2026-11-13 发现的静默 bug）

`broadcast()` 旧实现里写的是 `np.ascontiguousarray(v.data, dtype=np.float32)`。
CUDA 张量的 `.data` 是 `None`（真实数据在 `._gpu`），而

    np.ascontiguousarray(None, dtype=np.float32)  ->  array([nan])

**不抛异常**。于是广播在 GPU 上会把每个参数当成"1 个元素的 nan"：
rank0 往共享内存写 nan，其余 rank 往一个与张量无关的临时数组里写 —— 权重根本没广播，
而且不报错。多卡会在"各卡模型不同"的状态下继续训练，梯度平均出来的是垃圾。

这些用例不碰 CUDA：用假的设备对象模拟 `.data is None` 的形态，断言取值/写回走对了路径。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import dist as D  # noqa: E402


class _FakeBuf(object):
    def __init__(self, arr):
        self.shape = tuple(arr.shape)
        self.dtype = np.dtype(np.float32)
        self.nbytes = int(arr.nbytes)
        self.ptr = 0xC0FFEE


class _FakeCudaParam(object):
    """模拟 CUDA 张量的形态：`.device == "cuda"`、`.data is None`、数据在 `._gpu`。"""

    def __init__(self, arr):
        self.device = "cuda"
        self.data = None                      # ← 这是关键
        self._arr = np.ascontiguousarray(arr, dtype=np.float32)
        self._gpu = _FakeBuf(self._arr)
        self.grad = None

    def to_numpy(self):
        return self._arr.copy()

    def numel(self):
        return int(self._arr.size)

    @property
    def shape(self):
        return tuple(self._arr.shape)


class _FakeRt(object):
    """只实现 `copy_h2d_into`，用来观测写回是否真的落到设备缓冲。"""

    def __init__(self):
        self.calls = []

    def copy_h2d_into(self, buf, val):
        self.calls.append((tuple(buf.shape), np.array(val, copy=True)))
        return buf


def test_naive_numpy_on_none_is_silent_nan():
    """把这个坑本身钉住：这就是为什么需要 `_get_data`。"""
    a = np.ascontiguousarray(None, dtype=np.float32)
    assert a.shape == (1,), "numpy 对 None 会给出 1 元素数组"
    assert np.isnan(a).all(), "值是 nan —— 且不会抛异常，所以失败是静默的"


def test_get_data_reads_cuda_through_to_numpy():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    got = D._get_data(_FakeCudaParam(arr))
    assert got.shape == (2, 2)
    assert np.array_equal(got, arr), "必须拿到真实数值，而不是 [nan]"


def test_set_data_writes_into_device_buffer(monkeypatch):
    rt = _FakeRt()
    import tgai_go.train as T
    monkeypatch.setattr(T, "_cuda", lambda: rt)

    p = _FakeCudaParam(np.zeros((2, 3), dtype=np.float32))
    want = np.arange(6, dtype=np.float32).reshape(2, 3)
    D._set_data(p, want)

    assert rt.calls, "CUDA 路径必须调用 copy_h2d_into 原地写入（h2d 会新分配并漏掉旧缓冲）"
    shape, val = rt.calls[0]
    assert shape == (2, 3)
    assert np.array_equal(val, want)


def test_set_data_rejects_shape_mismatch_with_fix_hint(monkeypatch):
    rt = _FakeRt()
    import tgai_go.train as T
    monkeypatch.setattr(T, "_cuda", lambda: rt)
    p = _FakeCudaParam(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(RuntimeError) as ei:
        D._set_data(p, np.zeros((3, 2), dtype=np.float32))
    msg = str(ei.value)
    assert "形状不符" in msg and "修法" in msg
    assert not rt.calls, "形状不符时不能写设备缓冲"


def test_shard_indices_contiguous_covers_all_without_overlap():
    seen = []
    for rank in range(3):
        dp = D.DataParallel(world=3, rank=rank, job="unit_shard")
        idx = dp.shard_indices(10, contiguous=True)
        assert idx == sorted(idx) and len(idx) == len(set(idx))
        seen.extend(idx)
    assert sorted(seen) == list(range(10)), "三个 rank 必须恰好覆盖 0..9 且不重叠"


def test_shard_indices_strided_covers_all_without_overlap():
    seen = []
    for rank in range(3):
        dp = D.DataParallel(world=3, rank=rank, job="unit_shard")
        idx = dp.shard_indices(10, contiguous=False)
        assert len(idx) == len(set(idx))
        seen.extend(idx)
    assert sorted(seen) == list(range(10))


def test_shard_indices_world_one_is_identity():
    dp = D.DataParallel(world=1, rank=0, job="unit_shard")
    assert dp.shard_indices(4) == [0, 1, 2, 3]
