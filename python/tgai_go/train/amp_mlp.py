# -*- coding: utf-8 -*-
"""真 AMP（激活存 fp16）：MLP 块内部全程 fp16 存储，外面仍是 fp32。

## 为什么只改 MLP

一个 24 层、D=896、FF=4864 的模型，**每层 MLP 有 3 个 [B*S, FF] 的大张量**
（`g` = gate 输出、`u` = up 输出、`act` = silu(g)*u）。B*S=2048 时每个 40 MB（fp32），
一层 120 MB，24 层 2.9 GB，**只算前向**。反向还有 `d_act`/`dg`/`du` 同量级。
把这些换成 fp16 存储 ⇒ 带宽直接减半，这正是 torch AMP 比我们快 1.77× 的主要原因。

## 数值策略（标准 AMP，不是"全 fp16"）

* **主权重 fp32**（不降精度），每步只是 cast 出一份 fp16 影子（缓存，优化器更新后失效）
* **GEMM 用 fp32 累加**（`CUBLAS_COMPUTE_32F`），只有输入输出是 fp16
* **残差相加仍在 fp32**（`out = fp32_gemm_result`，由调用方加到 fp32 残差上）
* silu 在 fp32 里算（fp16 存储、fp32 计算），与 torch autocast 语义一致

## 用法

```python
from tgai_go.train import amp_mlp
amp_mlp.enable()                      # 或 cfg["MLP_FP16"]=True 由模型自己开
y = amp_mlp.mlp(h, W1, W2, W3)        # h: [N, D] fp32 Tensor，返回 [N, D] fp32 Tensor
```

优化器 step 后必须让权重影子失效 —— 已挂在 `AdamW.step` 的失效钩子里，
外部只需要保证 `T.fp16_shadows_invalidate()` 会被调用（现有机制）。
"""
import ctypes
import importlib

import numpy as np

_ENABLED = [False]
_MOD = [None]
_CACHE = {}          # id(weight) -> (GpuBuf fp16, 权重 buffer id)


def enable(on=True):
    _ENABLED[0] = bool(on)


def enabled():
    return _ENABLED[0]


def invalidate():
    """权重更新后调用：丢掉 fp16 影子。"""
    _CACHE.clear()


# ---------------------------------------------------------------- CUDA 源码
_SRC = r"""
/* fp16 位模式与 fp32 互转（与 cuda.py 里的实现同源，这里自包含） */
__device__ __forceinline__ unsigned int amp_f32_to_f16_bits(float f) {
    unsigned int x = __float_as_uint(f);
    unsigned int s = x & 0x80000000u;
    unsigned int mag = x & 0x7fffffffu;
    if (mag >= 0x7f800000u) {                    /* inf / nan */
        unsigned int payload = (mag > 0x7f800000u) ? 0x200u : 0u;
        return (s >> 16) | 0x7c00u | payload;
    }
    if (mag >= 0x477ff000u) {                    /* >= 65520 一律进位成 inf */
        return (s >> 16) | 0x7c00u;
    }
    int e = (int)(mag >> 23);                    /* 偏置 127 的指数 */
    unsigned int m = mag & 0x7fffffu;
    int ne = e - 127 + 15;                       /* 换成偏置 15 */
    if (ne <= 0) {                               /* half 的次正规 / 0（< 2^-14） */
        if (ne < -10) return s >> 16;            /* 太小，舍到 0 */
        m |= 0x800000u;
        int sh = 14 - ne;                        /* 14..24，留 10 位 */
        unsigned int h = m >> sh;
        unsigned int rem = m & ((1u << sh) - 1u);
        unsigned int mid = 1u << (sh - 1);
        if (rem > mid || (rem == mid && (h & 1u))) h++;
        return (s >> 16) | h;
    }
    unsigned int h = ((unsigned int)ne << 10) | (m >> 13);
    unsigned int rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return (s >> 16) | h;
}

__device__ float tg_f16_bits_to_f32(unsigned int h) {
    unsigned int s = (h & 0x8000u) << 16;
    unsigned int e = (h >> 10) & 0x1fu;
    unsigned int m = h & 0x3ffu;
    if (e == 0u) {
        if (m == 0u) return __uint_as_float(s);
        int sh = 0;
        while (!(m & 0x400u)) { m <<= 1; sh++; }
        m &= 0x3ffu;
        return __uint_as_float(s | ((unsigned int)(113 - sh) << 23) | (m << 13));
    }
    if (e == 31u) return __uint_as_float(s | 0x7f800000u | (m << 13));
    return __uint_as_float(s | ((e + 112u) << 23) | (m << 13));
}

extern "C" __global__ void amp_cast_f32_to_f16(const float* __restrict__ a,
                                               unsigned short* __restrict__ o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = amp_f32_to_f16_bits(a[i]);
}

/* act = silu(g) * u，全部 fp16 存储、fp32 计算 */
extern "C" __global__ void amp_silu_mul_f16(const unsigned short* __restrict__ g,
                                            const unsigned short* __restrict__ u,
                                            unsigned short* __restrict__ o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gv = tg_f16_bits_to_f32((unsigned int)g[i]);
        float uv = tg_f16_bits_to_f32((unsigned int)u[i]);
        float s = 1.0f / (1.0f + __expf(-gv));
        o[i] = amp_f32_to_f16_bits(gv * s * uv);
    }
}

extern "C" __global__ void amp_silu_mul_f16_bwd(const unsigned short* __restrict__ dy,
                                                const unsigned short* __restrict__ g,
                                                const unsigned short* __restrict__ u,
                                                unsigned short* __restrict__ dg,
                                                unsigned short* __restrict__ du, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float d = tg_f16_bits_to_f32((unsigned int)dy[i]);
        float gv = tg_f16_bits_to_f32((unsigned int)g[i]);
        float uv = tg_f16_bits_to_f32((unsigned int)u[i]);
        float s = 1.0f / (1.0f + __expf(-gv));
        /* dsilu/dg = s * (1 + g*(1-s)) */
        float ds = s * (1.0f + gv * (1.0f - s));
        dg[i] = amp_f32_to_f16_bits(d * uv * ds);
        du[i] = amp_f32_to_f16_bits(d * gv * s);
    }
}
"""


def _cuda_mod():
    """取真正的 cuda 模块（tgai_go.train.cuda 这个属性指向运行时实例，不是模块）。"""
    return importlib.import_module("tgai_go.train.cuda")


def _mod():
    m = _MOD[0]
    if m is None:
        from tgai_go.train import ops_cuda
        m = ops_cuda.load_source(_SRC, cu_name="tgai_amp.cu")
        _MOD[0] = m
    return m


def _rt():
    import tgai_go.train as T
    return T._cuda()


def _launch(name, bufs, n):
    rt = _rt()
    grid, block = rt._ewise_grid(int(n))
    rt.launch(name, grid, block,
              [(int(b.ptr), ctypes.c_uint64) for b in bufs] + [(int(n), ctypes.c_int)],
              module=_mod())


def _f16_buf(shape):
    rt = _rt()
    n = int(np.prod(shape)) if shape else 1
    return rt.empty((n,), dtype=np.uint16)


def to_f16(x):
    """fp32 GpuBuf → fp16(位模式) GpuBuf。

    用 cuda.py 里**已经验证过**的 cast_f32_to_f16，不要自己手写位变换
    （我手写过一版，h 的往返误差 12.7%，正确应在 5e-4 量级）。
    """
    rt = _rt()
    out = _f16_buf((int(x.n),))
    if x.n:
        grid, block = rt._ewise_grid(int(x.n))
        rt.launch("cast_f32_to_f16", grid, block, [
            rt._ptr(x), rt._ptr(out), (int(x.n), ctypes.c_int),
        ])
    return out


def _w16(w):
    """权重 fp16 影子（按 buffer 身份缓存；优化器 step 后由 invalidate() 清掉）。"""
    key = id(w)
    ent = _CACHE.get(key)
    if ent is not None and ent[1] == id(w._gpu):
        return ent[0]
    buf = to_f16(w._gpu)
    _CACHE[key] = (buf, id(w._gpu))
    return buf


def _gemm16(opA, opB, m, n, k, A16, lda, B16, ldb, out_shape, ldc, out16=True):
    """cublasGemmEx：fp16 输入、fp32 累加、输出可选 fp16。"""
    rt = _rt()
    rt._bind_gemm_ex()
    C = _cuda_mod()
    ctype = C.CUDA_R_16F if out16 else C.CUDA_R_32F
    dt = np.uint16 if out16 else np.float32
    Cbuf = rt.empty(out_shape, dtype=dt)
    if not (m and n and k):
        return Cbuf
    rt._activate()
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)
    rt._chk_blas(rt.cublas.cublasGemmEx(
        rt.handle, opA, opB, int(m), int(n), int(k),
        ctypes.byref(alpha),
        ctypes.c_uint64(int(A16.ptr)), C.CUDA_R_16F, int(lda),
        ctypes.c_uint64(int(B16.ptr)), C.CUDA_R_16F, int(ldb),
        ctypes.byref(beta),
        ctypes.c_uint64(int(Cbuf.ptr)), ctype, int(ldc),
        C.CUBLAS_COMPUTE_32F,
        C.CUBLAS_GEMM_DEFAULT,
    ), "cublasGemmEx(amp)")
    return Cbuf


def _gemm_nt_out16(A16, B16, M, N, K):
    """C[M,N] = A16[M,K] @ B16[N,K]^T → fp16（行主序 ⟺ 列主序转置传参）。"""
    C = _cuda_mod()
    return _gemm16(C.CUDA_OP_T, C.CUDA_OP_N, N, M, K, B16, K, A16, K, (M, N), N, True)


def _gemm_nn_out16(A16, B16, M, N, K):
    """C[M,K] = A16[M,N] @ B16[N,K] → fp16。"""
    C = _cuda_mod()
    return _gemm16(C.CUDA_OP_N, C.CUDA_OP_N, K, M, N, B16, K, A16, N, (M, K), K, True)


def _gemm_nn_raw(A16, B16, M, N, K):
    C = _cuda_mod()
    return _gemm16(C.CUDA_OP_N, C.CUDA_OP_N, K, M, N, B16, K, A16, N, (M, K), K, False)


def _gemm_tn_raw(A16, B16, M, N, K):
    """C[N,K] = A16[M,N]^T @ B16[M,K] → fp32（权重梯度用）。"""
    C = _cuda_mod()
    return _gemm16(C.CUDA_OP_N, C.CUDA_OP_T, K, N, M, B16, K, A16, N, (N, K), K, False)


def silu_mul_f16(g16, u16, n):
    o = _f16_buf((int(n),))
    _launch("amp_silu_mul_f16", [g16, u16, o], int(n))
    return o


def silu_mul_bwd_f16(dy16, g16, u16, n):
    dg = _f16_buf((int(n),))
    du = _f16_buf((int(n),))
    _launch("amp_silu_mul_f16_bwd", [dy16, g16, u16, dg, du], int(n))
    return dg, du


def mlp(h, W1, W2, W3):
    """`y = (silu(h @ W1.T) * (h @ W2.T)) @ W3.T`，内部激活存 fp16。

    h/W1/W2/W3 都是 fp32 Tensor；返回 fp32 Tensor（残差由调用方在 fp32 里加）。
    形状：h [N, D]，W1/W2 [FF, D]，W3 [D, FF]。
    """
    import tgai_go.train as T
    t = T._as_tensor(h)
    if not enabled():
        act = T.silu_mul(t @ W1.T, t @ W2.T)
        return act @ W3.T
    N, D = int(t.shape[0]), int(t.shape[1])
    FF = int(W1.shape[0])
    h16 = to_f16(t._gpu)
    W116, W216, W316 = _w16(W1), _w16(W2), _w16(W3)
    g16 = _gemm_nt_out16(h16, W116, N, FF, D)
    u16 = _gemm_nt_out16(h16, W216, N, FF, D)
    act16 = silu_mul_f16(g16, u16, N * FF)
    y = _gemm_nt_out16(act16, W316, N, D, FF)   # [N,D] fp16
    # 回到 fp32：残差相加要 fp32
    y32 = _to_f32(y, (N, D))
    out = T.Tensor._from_gpu(y32, requires_grad=(t.requires_grad or W1.requires_grad),
                             _parents=(t, W1, W2, W3))

    def bw(g, grads):
        dy = g
        dy16 = to_f16(dy._gpu if hasattr(dy, "_gpu") else dy)
        d_act16 = _gemm_nn_out16(dy16, W316, N, D, FF)
        dg16, du16 = silu_mul_bwd_f16(d_act16, g16, u16, N * FF)
        if t.requires_grad:
            dh = _gemm_nn_raw(dg16, W116, N, FF, D)
            dh2 = _gemm_nn_raw(du16, W216, N, FF, D)
            T._acc(grads, t, _add_bufs(dh, dh2, (N, D)))
        if W1.requires_grad:
            T._acc(grads, W1, _gemm_tn_raw(dg16, h16, N, FF, D))
        if W2.requires_grad:
            T._acc(grads, W2, _gemm_tn_raw(du16, h16, N, FF, D))
        if W3.requires_grad:
            T._acc(grads, W3, _gemm_tn_raw(dy16, act16, N, D, FF))

    out._backward = bw
    return out


def _to_f32(a16, shape):
    """fp16(位模式) → fp32（带形状，供 Tensor._from_gpu 用）。"""
    rt = _rt()
    out = rt.empty(tuple(shape), dtype=np.float32)
    n = int(np.prod(shape)) if shape else 1
    if n:
        grid, block = rt._ewise_grid(n)
        rt.launch("cast_f16_to_f32", grid, block, [
            rt._ptr(a16), rt._ptr(out), (n, ctypes.c_int),
        ])
    return out


def _add_bufs(a, b, shape):
    rt = _rt()
    out = rt.empty(shape, dtype=np.float32)
    n = int(np.prod(shape))
    if n:
        grid, block = rt._ewise_grid(n)
        rt.launch("ewise_add", grid, block, [
            rt._ptr(a), rt._ptr(b), rt._ptr(out), (n, ctypes.c_int),
        ])
    return out
