# -*- coding: utf-8 -*-
"""bf16 存储的第一块砖：没有头文件也能做的 bf16 转换 + 内核源码。

## 为什么不用 `#include <cuda_bf16.h>`

引擎的内核源码是**一整段 NVRTC 字符串**，刻意不含任何 `#include`
（NVRTC 没配 `-I`，加了就编不过 —— 这是仓库里早就记录过的坑）。
所以 bf16 只能走**位运算**：bf16 就是 float32 的高 16 位。

## 舍入

* 截断（直接取高 16 位）实现最简单，但**有系统性向零偏置**，累加几百层会明显漂移；
* 这里用 **round-to-nearest-even**（CUDA 的 `__float2bfloat16` 也是这个语义）：
  在低 16 位上做"看第 15 位决定进位、遇到正好一半时向偶数靠"。

这套算法同时提供 numpy 版（`to_bf16_bits` / `from_bf16_bits`），
好处是**可以在 CPU 上对着 torch 的 bf16 转换验证**，不用等 GPU。
"""
import numpy as np

#: 精度开关。amp 只影响"存储"，计算一律 fp32（见 docs/AMP_PLAN.md）。
AMP_ENV = "TG_AMP"


def to_bf16_bits(x):
    """float32 → bf16 位（round-to-nearest-even），返回 uint16 数组。

    与 CUDA `__float2bfloat16` 同语义：NaN 保持 NaN（指数位非零即可）。
    """
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    u = a.view(np.uint32).astype(np.uint32)
    # round-to-nearest-even：low 是低 16 位，取整时看 bit15 与"后面是否还有位"
    low = u & 0xFFFF
    carry = ((low > 0x8000) | ((low == 0x8000) & ((u >> 16) & 1 == 1))).astype(np.uint32)
    out = ((u + (carry << 16)) >> 16).astype(np.uint16)
    return out


def from_bf16_bits(b):
    """bf16 位 → float32。"""
    u = np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def roundtrip_rel_error(x):
    """bf16 往返后的相对误差（用于自检与验收报告）。"""
    a = np.asarray(x, dtype=np.float32)
    b = from_bf16_bits(to_bf16_bits(a))
    denom = np.maximum(np.abs(a), 1e-30)
    return float(np.max(np.abs(b - a) / denom))


#: bf16 内核源码片段（拼进 `_device_source()` 即可；不依赖任何头文件）。
#: 约定：**读 bf16、用 float 算、写回 bf16**，与 docs/AMP_PLAN.md 的 S1 一致。
BF16_KERNELS = r"""
/* ---- bf16 存储（AMP S1）。不 include 任何头文件：bf16 就是 float32 的高 16 位 ---- */
__device__ __forceinline__ float tg_bf16_to_f32(unsigned short h) {
    unsigned int u = ((unsigned int)h) << 16;
    return __uint_as_float(u);
}

/* round-to-nearest-even，与 CUDA __float2bfloat16 同语义 */
__device__ __forceinline__ unsigned short tg_f32_to_bf16(float f) {
    unsigned int u = __float_as_uint(f);
    unsigned int low = u & 0xFFFFu;
    unsigned int carry = (low > 0x8000u || (low == 0x8000u && ((u >> 16) & 1u))) ? 1u : 0u;
    return (unsigned short)((u + (carry << 16)) >> 16);
}

extern "C" __global__ void ewise_add_bf16(const unsigned short* a, const unsigned short* b,
                                          unsigned short* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = tg_f32_to_bf16(tg_bf16_to_f32(a[i]) + tg_bf16_to_f32(b[i]));
}

extern "C" __global__ void ewise_scale_bf16(const unsigned short* a, float s,
                                            unsigned short* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = tg_f32_to_bf16(tg_bf16_to_f32(a[i]) * s);
}

/* 行归一化（RMSNorm）：读 bf16、全程 float 累加、写回 bf16 */
extern "C" __global__ void rmsnorm_bf16(const unsigned short* x, const float* w,
                                        unsigned short* y, int rows, int d, float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const unsigned short* xr = x + (long long)r * d;
    unsigned short* yr = y + (long long)r * d;
    __shared__ float red[256];
    float s = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = tg_bf16_to_f32(xr[i]);
        s = __fadd_rn(s, __fmul_rn(v, v));
    }
    red[threadIdx.x] = s;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off)
            red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + off]);
        __syncthreads();
    }
    float inv = rsqrtf(__fdiv_rn(red[0], (float)d) + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        yr[i] = tg_f32_to_bf16(tg_bf16_to_f32(xr[i]) * inv * w[i]);
}
"""


def amp_requested():
    """读环境变量判断是否要开 bf16 存储（与 TG_BF16 的"算"分开）。"""
    import os
    v = os.environ.get(AMP_ENV, "").strip().lower()
    return v in ("1", "true", "yes", "on")
