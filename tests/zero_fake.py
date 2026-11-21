# -*- coding: utf-8 -*-
"""ZeRO 的**无卡**测试替身：一个 numpy 版运行时（`rt`）。

为什么需要它：ZeRO 的代码路径（打包/写回/共享内存集合通信/分布式范数）全都在
「设备缓冲 + 运行时原语」上。只要把运行时换成 numpy 实现，
**真实的 `dist.DataParallel` 与真实的 `zero.*` 就能在 CPU 上跑两个进程**，
从而在没有 GPU 的机器上验掉最容易错的部分（偏移数学、barrier 配对、槽位互不重叠）。

★ 这里刻意**照抄内核语义**（`sumsq_multi_shard` / `sumsq_multi_final` 的
  线程循环 + 树归约），因为 ZeRO-2 的 gnorm 逐位一致正是靠这套结合顺序；
  用 numpy 的 `sum()` 代替会把这个测试变成"验了个假东西"。
"""
import ctypes

import numpy as np


class NBuf(object):
    """numpy 载体 + 运行时缓冲需要的字段（`.n` / `.ptr` / `.shape` / `.dtype`）。

    `.ptr` 给一个假的非零地址（`_zero` 路径不会用到它；host 后端也不碰设备指针）。
    """

    __slots__ = ("np", "n", "ptr", "dtype", "keepalive", "known_scalar")

    def __init__(self, arr):
        self.np = np.ascontiguousarray(arr, dtype=np.float32)
        self.n = int(self.np.size)
        self.ptr = int(self.np.ctypes.data)
        self.dtype = np.dtype(np.float32)
        self.keepalive = None
        self.known_scalar = None

    @property
    def shape(self):
        return tuple(self.np.shape)

    @property
    def nbytes(self):
        return int(self.np.nbytes)


class NParam(object):
    """假参数：够 `zero` / `dist` 用（`_gpu` / `grad` / `numel()` / `shape` / `device`）。"""

    def __init__(self, arr, requires_grad=True):
        self._gpu = NBuf(arr)
        self.grad = None
        self.requires_grad = bool(requires_grad)
        self.device = "cuda"          # 让 dist/_set_grad 走 CUDA 分支（这里不触发）
        self._shape = tuple(np.shape(arr))

    def numel(self):
        return int(self._gpu.n)

    @property
    def shape(self):
        return self._shape

    def numpy(self):
        return self._gpu.np

    def set_grad(self, arr):
        self.grad = NParam(arr)
        return self.grad


class _Pinned(object):
    __slots__ = ("ptr", "nbytes", "_arr")

    def __init__(self, nbytes):
        self._arr = np.zeros(max(1, int(nbytes) // 4), dtype=np.float32)
        self.ptr = ctypes.c_void_p(self._arr.ctypes.data)
        self.nbytes = int(self._arr.nbytes)

    def array(self, dtype=np.float32, n=None):
        n = int(n) if n is not None else self.nbytes // np.dtype(dtype).itemsize
        return self._arr[:n]


class _Reg(object):
    """``host_register`` 的返回值：只是一块**已存在**主机内存的地址（不拥有它）。"""

    __slots__ = ("ptr", "nbytes")


class FakeRT(object):
    """numpy 运行时。只实现 ZeRO/DP 用到的那批原语。"""

    def __init__(self):
        self.cu = None
        self._stream = ctypes.c_void_p(0)
        self._pinned = {}

    # ---------- 分配 ----------
    def empty(self, shape, dtype=np.float32):
        return NBuf(np.zeros(tuple(shape) if isinstance(shape, (tuple, list)) else (shape,),
                             dtype=np.float32))

    def zeros(self, shape, dtype=np.float32):
        return self.empty(shape, dtype)

    def host_pinned(self, nbytes, slot=0):
        ent = self._pinned.get(slot)
        if ent is None or ent.nbytes < int(nbytes):
            ent = _Pinned(int(nbytes))
            self._pinned[slot] = ent
        return ent

    def host_register(self, obj, nbytes=None):
        """把已有主机内存（共享内存）注册成「页锁定」：返回带 ``.ptr`` 的托管对象。

        真实现走 ``cuMemHostRegister``（不复制、只是让驱动认这块地址）；这里返回同一
        块内存的地址即可 —— 这样 CPU 测试也能覆盖 **d2h 直接写进共享内存** 这条主路径
        （不覆盖的话，只有上卡才能发现偏移/对齐错误）。
        """
        arr = np.frombuffer(obj, dtype=np.uint8)
        if nbytes is not None and int(nbytes) > arr.nbytes:
            raise AssertionError("host_register 请求 %d 字节 > 缓冲 %d"
                                 % (int(nbytes), arr.nbytes))
        ph = _Reg()
        ph.ptr = ctypes.c_void_p(int(arr.ctypes.data))
        ph.nbytes = int(nbytes if nbytes is not None else arr.nbytes)
        return ph

    def sub_view(self, buf, shape, elem_offset=0):
        """**真视图**（与真运行时一致）：改 view 就是改原缓冲。"""
        n = 1
        for s in (shape if isinstance(shape, (tuple, list)) else (shape,)):
            n *= int(s)
        off = int(elem_offset)
        if off + n > int(buf.n):
            raise AssertionError("sub_view 越界：off=%d n=%d buf.n=%d" % (off, n, buf.n))
        v = NBuf.__new__(NBuf)
        v.np = buf.np[off:off + n]
        v.n = n
        v.ptr = int(buf.np.ctypes.data) + off * 4
        v.dtype = np.dtype(np.float32)
        v.keepalive = buf
        v.known_scalar = None
        return v

    # ---------- 搬运 ----------
    def copy_d2d(self, dst, src, nbytes=None, dst_off=0, src_off=0):
        n = int(nbytes if nbytes is not None else src.n * 4) // 4
        do = int(dst_off) // 4
        so = int(src_off) // 4
        dst.np[do:do + n] = src.np[so:so + n]
        return n

    def d2h_into(self, buf, dst, nbytes=None, dst_off=0, src_off=0):
        nb = int(nbytes if nbytes is not None else buf.n * 4)
        ctypes.memmove(int(dst.value) + int(dst_off),
                       int(buf.np.ctypes.data) + int(src_off), nb)
        return nb

    def h2d_into(self, buf, src, nbytes=None, dst_off=0, src_off=0):
        nb = int(nbytes if nbytes is not None else buf.n * 4)
        ctypes.memmove(int(buf.np.ctypes.data) + int(dst_off),
                       int(src.value) + int(src_off), nb)
        return nb

    def copy_h2d_into(self, buf, arr):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32)).reshape(-1)
        assert a.size == int(buf.n), "copy_h2d_into 长度不符：%d vs %d" % (a.size, buf.n)
        buf.np[:] = a
        return int(a.size)

    def d2h(self, buf):
        return buf.np.copy()

    def read_scalar(self, buf):
        return float(np.asarray(buf.np).reshape(-1)[0])

    def _memset32(self, buf, word):
        if int(word) != 0:
            raise AssertionError("FakeRT._memset32 只支持清零（word=0）")
        buf.np[:] = 0.0

    def scale_inplace(self, buf, scalar):
        buf.np *= np.float32(float(scalar))

    def sync(self):
        return None

    # ---------- 与内核逐位一致的两个平方和入口 ----------
    def _shard_offsets(self, bufs):
        nb = np.empty(len(bufs), dtype=np.int64)
        for i, b in enumerate(bufs):
            m = int(b.n)
            nb[i] = max(1, min(1024, (m + 255) // 256))
        off = np.zeros(len(bufs) + 1, dtype=np.int32)
        off[1:] = np.cumsum(nb)
        return nb, off

    def sumsq_multi_shard_into(self, bufs, part, elem_offset=0):
        bufs = [b for b in bufs if b is not None and int(b.n) > 0]
        if not bufs:
            return 0
        nb, off = self._shard_offsets(bufs)
        for t, b in enumerate(bufs):
            m = int(b.n)
            nt = int(nb[t])
            x = b.np.reshape(-1)
            for blk in range(nt):
                sm = np.zeros(256, np.float32)
                for tid in range(256):
                    acc = np.float32(0.0)
                    i = blk * 256 + tid
                    while i < m:
                        v = np.float32(x[i])
                        acc = np.float32(acc + np.float32(v * v))
                        i += 256 * nt
                    sm[tid] = acc
                o = 128
                while o > 0:
                    sm[:o] = np.float32(sm[:o] + sm[o:2 * o])
                    o >>= 1
                part.np[int(elem_offset) + int(off[t]) + blk] = sm[0]
        return int(off[-1])

    def sumsq_parallel(self, buf):
        """`sumsq_partial` + `sumsq_final` 的等价物（单张量两段归约）。

        真实现里这就是"grid=nb 写 part、grid=1 做树归约"，与
        `sumsq_multi_shard` + `sumsq_multi_final` 对单张量完全同序，所以直接用
        同一套复刻跑一遍即可。
        """
        _nb, off = self._shard_offsets([buf])
        part = NBuf(np.zeros(int(off[-1]), np.float32))
        self.sumsq_multi_shard_into([buf], part, 0)
        val = self.sumsq_blocks_total(part, off, 1)
        return NBuf(np.array([val], np.float32))

    def sumsq_blocks_total(self, part, off, n_tensors):
        total = np.float32(0.0)
        for t in range(int(n_tensors)):
            i0, i1 = int(off[t]), int(off[t + 1])
            sm = np.zeros(256, np.float32)
            for tid in range(256):
                acc = np.float32(0.0)
                i = i0 + tid
                while i < i1:
                    acc = np.float32(acc + np.float32(part.np[i]))
                    i += 256
                sm[tid] = acc
            o = 128
            while o > 0:
                sm[:o] = np.float32(sm[:o] + sm[o:2 * o])
                o >>= 1
            total = np.float32(total + sm[0])
        return float(total)

    def _zero_keep_alive(self, objs):
        return None


def reference_ssq_total(bufs):
    """**不分片**参考路径：一次把全部张量写进 part，再跑 final。逐位基准。"""
    rt = FakeRT()
    bufs = [b for b in bufs if b is not None and int(b.n) > 0]
    if not bufs:
        return 0.0
    _nb, off = rt._shard_offsets(bufs)
    part = NBuf(np.zeros(int(off[-1]), np.float32))
    rt.sumsq_multi_shard_into(bufs, part, 0)
    return rt.sumsq_blocks_total(part, off, len(bufs))
