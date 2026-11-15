# -*- coding: utf-8 -*-
"""给 SiLU 的三个标量内核加 float4 向量化版本，并在主机侧派发。

依据（实测）：SiLU [8192,2560] 前向+反向 ORV 10.32 ms vs torch 0.89 ms（11.6 倍）。
根因：`silu_mul_fwd` / `silu_fwd_inplace` / `silu_bwd_da` 都是 `a[i]` 逐元素标量访存，
而引擎里其它逐元素内核（ewise_add_v4 / ewise_mul_v4 / ewise_exp_v4 ...）都已向量化。
cuda.py:523 的注释写着「照 PyTorch 的做法（ActivationSiluKernel.cu）：整个逐元素链一个内核」
—— 抄了「单内核」，漏了 torch 同时用的 float4。

原则：**逐位一致**。每个分量仍走 `silu32_pair` + `__fmul_rn`，与标量版相同；
只有访存宽度从 4 B 变 16 B。n%4!=0 或未对齐时自动退回标量内核（原路径不变）。
"""
import io
import re

P = "python/tgai_go/train/cuda.py"

# ---- 内核：插在标量版之后 ----
K_AFTER = """extern \"C\" __global__ void silu_bwd_da(const float* g, const float* a, float* da, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float y, dsilu;
        silu32_pair(a[i], &y, &dsilu);
        (void)y;
        da[i] = __fmul_rn(g[i], dsilu);
    }
}
"""

K_V4 = """

/* ★ 2026-11-15：float4 版本。逐分量仍走 silu32_pair + __fmul_rn，结果与标量版逐位相同，
   只是访存宽度变成 16 B。真机实测标量版比 torch 慢 11.6 倍，这里是主因。 */
extern \"C\" __global__ void silu_mul_fwd_v4(float4* a, const float4* b, int n4) {
    const int stride = (int)gridDim.x * (int)blockDim.x;
    for (int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x; i < n4; i += stride) {
        float4 x = a[i];
        float4 y = b[i];
        float4 r;
        float s, d;
        silu32_pair(x.x, &s, &d); r.x = __fmul_rn(s, y.x);
        silu32_pair(x.y, &s, &d); r.y = __fmul_rn(s, y.y);
        silu32_pair(x.z, &s, &d); r.z = __fmul_rn(s, y.z);
        silu32_pair(x.w, &s, &d); r.w = __fmul_rn(s, y.w);
        a[i] = r;
    }
}

extern \"C\" __global__ void silu_fwd_inplace_v4(float4* a, int n4) {
    const int stride = (int)gridDim.x * (int)blockDim.x;
    for (int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x; i < n4; i += stride) {
        float4 x = a[i];
        float4 r;
        float s, d;
        silu32_pair(x.x, &s, &d); r.x = s;
        silu32_pair(x.y, &s, &d); r.y = s;
        silu32_pair(x.z, &s, &d); r.z = s;
        silu32_pair(x.w, &s, &d); r.w = s;
        a[i] = r;
    }
}

extern \"C\" __global__ void silu_bwd_da_v4(const float4* g, const float4* a, float4* da, int n4) {
    const int stride = (int)gridDim.x * (int)blockDim.x;
    for (int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x; i < n4; i += stride) {
        float4 x = a[i];
        float4 gg = g[i];
        float4 r;
        float s, d;
        silu32_pair(x.x, &s, &d); r.x = __fmul_rn(gg.x, d);
        silu32_pair(x.y, &s, &d); r.y = __fmul_rn(gg.y, d);
        silu32_pair(x.z, &s, &d); r.z = __fmul_rn(gg.z, d);
        silu32_pair(x.w, &s, &d); r.w = __fmul_rn(gg.w, d);
        da[i] = r;
    }
}
"""

# ---- 主机侧派发 ----
D_MUL = """        if a.n:
            self._split_forget_ptr(a)
            grid, block = self._ewise_grid(a.n)
            self.launch("silu_mul_fwd", grid, block, [
                self._ptr(a), self._ptr(b), (int(a.n), ctypes.c_int),
            ])"""
D_MUL_V4 = """        if a.n:
            self._split_forget_ptr(a)
            if self._vec4_ok(a.n, a, b):
                n4 = int(a.n) >> 2
                grid, block = self._ewise_grid(n4)
                self.launch("silu_mul_fwd_v4", grid, block, [
                    self._ptr(a), self._ptr(b), (n4, ctypes.c_int),
                ])
                return a
            grid, block = self._ewise_grid(a.n)
            self.launch("silu_mul_fwd", grid, block, [
                self._ptr(a), self._ptr(b), (int(a.n), ctypes.c_int),
            ])"""

D_INP = """        if a.n:
            self._split_forget_ptr(a)
            grid, block = self._ewise_grid(a.n)
            self.launch("silu_fwd_inplace", grid, block, [
                self._ptr(a), (int(a.n), ctypes.c_int),
            ])"""
D_INP_V4 = """        if a.n:
            self._split_forget_ptr(a)
            if self._vec4_ok(a.n, a):
                n4 = int(a.n) >> 2
                grid, block = self._ewise_grid(n4)
                self.launch("silu_fwd_inplace_v4", grid, block, [
                    self._ptr(a), (n4, ctypes.c_int),
                ])
                return a
            grid, block = self._ewise_grid(a.n)
            self.launch("silu_fwd_inplace", grid, block, [
                self._ptr(a), (int(a.n), ctypes.c_int),
            ])"""

D_BWD = """        if n:
            grid, block = self._ewise_grid(n)
            self.launch("silu_bwd_da", grid, block, [
                self._ptr(g), self._ptr(a), self._ptr(da), (n, ctypes.c_int),
            ])"""
D_BWD_V4 = """        if n:
            if self._vec4_ok(n, g, a, da):
                n4 = n >> 2
                grid, block = self._ewise_grid(n4)
                self.launch("silu_bwd_da_v4", grid, block, [
                    self._ptr(g), self._ptr(a), self._ptr(da), (n4, ctypes.c_int),
                ])
                return da
            grid, block = self._ewise_grid(n)
            self.launch("silu_bwd_da", grid, block, [
                self._ptr(g), self._ptr(a), self._ptr(da), (n, ctypes.c_int),
            ])"""


def main():
    s = io.open(P, encoding="utf-8").read()
    if "silu_bwd_da_v4" in s:
        print("已经改过")
        return 0
    for old, new, tag in ((K_AFTER, K_AFTER + K_V4, "kernel"),
                          (D_MUL, D_MUL_V4, "silu_mul_fwd"),
                          (D_INP, D_INP_V4, "silu_fwd_inplace"),
                          (D_BWD, D_BWD_V4, "silu_bwd_da")):
        assert s.count(old) == 1, "%s 锚点不唯一（%d 处）" % (tag, s.count(old))
        s = s.replace(old, new, 1)
    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    import ast
    ast.parse(io.open(P, encoding="utf-8").read())
    print("已加 3 个 _v4 内核 + 派发（逐位一致，未对齐时退回标量）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
