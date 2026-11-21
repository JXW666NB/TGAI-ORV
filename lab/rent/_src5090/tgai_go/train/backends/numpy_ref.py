# -*- coding: utf-8 -*-
"""NumPy 参考后端 —— 后端接口的「活文档」，也是 Vulkan / ROCm 的接入模板。

## 它有什么用

1. **把接口契约钉死**：这里每一行都是「新后端必须做到什么」的可执行说明。
2. **无 GPU 也能测**：CI 上跑不了 CUDA，但能跑这个 → 后端抽象层出问题会被 CI 抓住。
3. **移植模板**：Vulkan/ROCm 照着这个结构填，把 numpy 调用换成 vkCmdDispatch / HIP kernel。

## 它不是什么

不是给用户用的推理/训练后端：**所有算子都是 O(n) 的 numpy 调用，没有 kernel 复用**，
只用于验证语义与接口。真实训练请用 `TG_BACKEND=cuda`。

## 已实现（24 / 79）

内存/拷贝 7、形状/布局 5、归约 3、GEMM 4、逐元素 6、softmax 1 ……
其余（融合核、rmsnorm、注意力、MoE、优化器）见 `backend.missing()` 的报告。

语义契约与 `tgai_go.train.backend` 文档一致：
* 缓冲区是连续行优先 float32（除非显式 dtype）；
* `sync()` 之前所有操作异步（本实现是同步的，但保持接口形状）；
* `gemm_nt(A,B)` = `A[M,K] @ B[N,K]^T`；`gemm_tn(A,B)` = `A[K,M]^T @ B[K,N]`。
"""
import numpy as np


class NpBuf:
    """与 GpuBuf 接口对齐的最小球（shape / n / dtype / nbytes）。"""

    __slots__ = ("arr", "shape", "n", "dtype", "nbytes")

    def __init__(self, arr, dtype=np.float32):
        self.arr = np.ascontiguousarray(arr, dtype=dtype)
        self.shape = tuple(self.arr.shape)
        self.n = int(self.arr.size)
        self.dtype = dtype
        self.nbytes = int(self.arr.nbytes)

    def __repr__(self):
        return "NpBuf(shape=%s, dtype=%s)" % (self.shape, np.dtype(self.dtype).name)


class NumpyBackend:
    """最小可用后端骨架。名字与 `tgai_go.train.backend.GROUPS` 一一对应。"""

    name = "numpy_ref"

    def __init__(self):
        self._scalar = {}

    # ---------------- 内存 / 拷贝 ----------------
    def empty(self, shape, dtype=np.float32):
        return NpBuf(np.empty(shape, dtype=dtype), dtype)

    def zeros(self, shape, dtype=np.float32):
        return NpBuf(np.zeros(shape, dtype=dtype), dtype)

    def full(self, shape, value, dtype=np.float32):
        return NpBuf(np.full(shape, value, dtype=dtype), dtype)

    def clone(self, buf):
        return NpBuf(buf.arr.copy(), buf.dtype)

    def h2d(self, arr, dtype=np.float32, cache_slot=None):
        return NpBuf(np.asarray(arr), dtype)

    def d2h(self, buf):
        return np.array(buf.arr, copy=True)

    def sync(self):
        return None

    # ---------------- 形状 / 布局 ----------------
    def reshape(self, buf, shape):
        return NpBuf(buf.arr.reshape(shape), buf.dtype)

    def permute(self, buf, axes):
        return NpBuf(np.transpose(buf.arr, axes), buf.dtype)

    def swap_last2(self, buf):
        return NpBuf(np.swapaxes(buf.arr, -1, -2), buf.dtype)

    def transpose2d(self, buf):
        return NpBuf(buf.arr.T, buf.dtype)

    def broadcast_to(self, buf, shape):
        return NpBuf(np.broadcast_to(buf.arr, shape), buf.dtype)

    # ---------------- 归约 ----------------
    def sum_all(self, buf):
        return NpBuf(np.array(buf.arr.sum(dtype=np.float64), dtype=np.float32))

    def sum_axis(self, buf, axis, keepdims=False):
        return NpBuf(np.sum(buf.arr, axis=axis, keepdims=keepdims, dtype=np.float64)
                     .astype(np.float32), buf.dtype)

    def read_scalar(self, buf):
        a = buf.arr
        if a.size != 1:
            raise ValueError(
                "read_scalar: 期望标量（size=1），实际 shape=%s。"
                "修法：先 sum_all 归约成标量再读" % (a.shape,))
        return float(a.reshape(()))

    # ---------------- GEMM ----------------
    def sgemm(self, A, B):
        return self._mm(A.arr, B.arr)

    def gemm_nt(self, A, B):
        return self._mm(A.arr, B.arr.T)

    def gemm_tn(self, A, B):
        return self._mm(A.arr.T, B.arr)

    def bmm(self, A, B):
        return self._mm(A.arr, B.arr)

    def sgemm_mode(self, A, B, mode):
        m = str(mode).upper()
        if m in ("NN", "NT", "TN", "TT"):
            a = A.arr.T if m[0] == "T" else A.arr
            b = B.arr.T if m[1] == "T" else B.arr
            return self._mm(a, b)
        raise ValueError("sgemm_mode: 不支持 mode=%r。修法：用 'NN'/'NT'/'TN'/'TT'" % (mode,))

    @staticmethod
    def _mm(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if a.shape[-1] != b.shape[-2]:
            raise ValueError(
                "matmul: 形状不匹配 —— A%s @ B%s（A 最后一维 %d ≠ B 倒数第二维 %d）。"
                "修法：检查 GEMM 的转置标志，或先 reshape/transpose"
                % (a.shape, b.shape, a.shape[-1], b.shape[-2]))
        return NpBuf(np.matmul(a, b).astype(np.float32))

    # ---------------- 逐元素 ----------------
    @staticmethod
    def _same(a, b, op, tag):
        if a.shape != b.shape:
            raise ValueError(
                "%s: 形状不一致 %s vs %s。修法：先 broadcast_to 把形状对齐"
                % (tag, a.shape, b.shape))
        return NpBuf(op(a.arr, b.arr).astype(np.float32))

    def add(self, a, b):
        return self._same(a, b, np.add, "add")

    def add_(self, x, y):
        if x.shape != y.shape:
            raise ValueError("add_: 形状不一致 %s vs %s。修法：先 broadcast_to" % (x.shape, y.shape))
        x.arr += y.arr
        return x

    def mul(self, a, b):
        return self._same(a, b, np.multiply, "mul")

    def scale(self, a, scalar):
        return NpBuf(a.arr * np.float32(scalar))

    def scale_inplace(self, a, scalar):
        a.arr *= np.float32(scalar)
        return a

    def div_scalar(self, a, scalar):
        if scalar == 0:
            raise ZeroDivisionError("div_scalar: 除数为 0。修法：检查分母（例如 RMSNorm 的 eps）")
        return NpBuf(a.arr / np.float32(scalar))

    def pow_scalar(self, a, p):
        return NpBuf(np.power(a.arr, np.float32(p)).astype(np.float32))

    def exp(self, a):
        return NpBuf(np.exp(a.arr).astype(np.float32))

    # ---------------- 注意力（最小） ----------------
    def softmax_lastdim(self, x):
        a = x.arr.astype(np.float64)
        a = a - a.max(axis=-1, keepdims=True)
        e = np.exp(a)
        return NpBuf((e / e.sum(axis=-1, keepdims=True)).astype(np.float32))


    # ---------------- 归约（补充） ----------------
    def sumsq_sequential(self, buf):
        """sum(x^2)，按 torch 的 float32 顺序累加语义（此处用 float64 更稳）。"""
        a = buf.arr.astype(np.float64)
        return NpBuf(np.array((a * a).sum(), dtype=np.float32))

    def unbroadcast(self, buf, shape):
        """把广播过的梯度按目标形状求和回去（反向必需）。"""
        a = buf.arr
        target = tuple(shape)
        while a.ndim > len(target):
            a = a.sum(axis=0)
        for i, sz in enumerate(target):
            if sz == 1 and a.shape[i] != 1:
                a = a.sum(axis=i, keepdims=True)
        return NpBuf(np.ascontiguousarray(a.astype(np.float32)))

    def sub_scale_(self, x, y, scalar):
        """x -= y * scalar（原地）。"""
        if x.shape != y.shape:
            raise ValueError("sub_scale_: 形状不一致 %s vs %s。修法：先 broadcast_to 对齐"
                             % (x.shape, y.shape))
        x.arr -= y.arr * np.float32(scalar)
        return x

    # ---------------- 通用矩阵乘 ----------------
    def matmul_general(self, A, B):
        """支持批量/广播的矩阵乘（与 torch.matmul 语义一致）。"""
        return self._mm(A.arr, B.arr)

    def sgemm_strided_batched(self, A, B, batch):
        return self._mm(A.arr, B.arr)

    def sgemm_nt_strided_batched(self, A, B, batch):
        return self._mm(A.arr, np.swapaxes(B.arr, -1, -2))

    def sgemm_tn_strided_batched(self, A, B, batch):
        return self._mm(np.swapaxes(A.arr, -1, -2), B.arr)


def make():
    """供 `backend.register("numpy_ref", tgai_go.train.backends.numpy_ref.make)` 使用。"""
    return NumpyBackend()
