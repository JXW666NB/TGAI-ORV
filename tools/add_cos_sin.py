# -*- coding: utf-8 -*-
"""加 cos / sin 算子（P100 报告点名：缺它们导致 RoPE 类模型迁移成本高）。

四处改动：
1. cuda.py 内核源码：`ewise_cos` / `ewise_sin` / `ewise_cos_bwd` / `ewise_sin_bwd`
2. cuda.py rt 方法：`cos` / `sin` / `cos_bwd` / `sin_bwd`
3. __init__.py `_cos_cuda` / `_sin_cuda`（设备路径）
4. __init__.py `Tensor.cos` / `Tensor.sin`（含 CPU 路径）

反向公式：
  d/dx cos(x) = -sin(x)  ⇒  dL/dx = g * (-sin(x))
  d/dx sin(x) =  cos(x)  ⇒  dL/dx = g * cos(x)
"""
import ast
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CU = os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py")
INIT = os.path.join(ROOT, "python", "tgai_go", "train", "__init__.py")

cu = io.open(CU, encoding="utf-8").read()
init = io.open(INIT, encoding="utf-8").read()
n = 0

if "ewise_cos" in cu:
    print("cuda.py 已含 cos 内核，跳过内核部分")
else:
    # ── 1. 内核：接在 ewise_abs_bwd 之后 ──
    anchor = "extern \"C\" __global__ void ewise_abs_bwd("
    i = cu.find(anchor)
    if i < 0:
        print("找不到 ewise_abs_bwd 锚点")
        sys.exit(1)
    j = cu.find("\n}\n", i) + 3
    KERNELS = '''
/* cos / sin：RoPE 等位置编码需要。反向分别是 -sin(x) 与 cos(x)。
   与 numpy 一致：用 cosf/sinf，不做快速近似（__cosf 精度不够）。 */
extern "C" __global__ void ewise_cos(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = cosf(a[i]);
}

extern "C" __global__ void ewise_sin(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = sinf(a[i]);
}

/* d/dx cos(x) = -sin(x) */
extern "C" __global__ void ewise_cos_bwd(const float* a, const float* g, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = -sinf(a[i]) * g[i];
}

/* d/dx sin(x) = cos(x) */
extern "C" __global__ void ewise_sin_bwd(const float* a, const float* g, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = cosf(a[i]) * g[i];
}
'''
    cu = cu[:j] + KERNELS + cu[j:]
    n += 1

if "def cos(self, a):" not in cu:
    # ── 2. rt 方法：接在 abs_bwd 之后 ──
    anchor2 = "    def relu(self, a):"
    k = cu.find(anchor2)
    if k < 0:
        print("找不到 relu 锚点")
        sys.exit(1)
    METHODS = '''    def cos(self, a):
        out = self.empty(a.shape, dtype=np.float32)
        if a.n:
            grid, block = self._ewise_grid(a.n)
            self.launch("ewise_cos", grid, block, [
                self._ptr(a), self._ptr(out), (int(a.n), ctypes.c_int),
            ])
        return out

    def sin(self, a):
        out = self.empty(a.shape, dtype=np.float32)
        if a.n:
            grid, block = self._ewise_grid(a.n)
            self.launch("ewise_sin", grid, block, [
                self._ptr(a), self._ptr(out), (int(a.n), ctypes.c_int),
            ])
        return out

    def cos_bwd(self, a, g):
        out = self.empty(a.shape, dtype=np.float32)
        if a.n:
            grid, block = self._ewise_grid(a.n)
            self.launch("ewise_cos_bwd", grid, block, [
                self._ptr(a), self._ptr(g), self._ptr(out),
                (int(a.n), ctypes.c_int),
            ])
        return out

    def sin_bwd(self, a, g):
        out = self.empty(a.shape, dtype=np.float32)
        if a.n:
            grid, block = self._ewise_grid(a.n)
            self.launch("ewise_sin_bwd", grid, block, [
                self._ptr(a), self._ptr(g), self._ptr(out),
                (int(a.n), ctypes.c_int),
            ])
        return out

'''
    cu = cu[:k] + METHODS + cu[k:]
    n += 1

if "_cos_cuda" not in init:
    # ── 3. 设备路径：接在 _abs_cuda 之后 ──
    anchor3 = "    def _relu_cuda(self):"
    m = init.find(anchor3)
    if m < 0:
        print("找不到 _relu_cuda 锚点")
        sys.exit(1)
    DEVPATH = '''    def _cos_cuda(self):
        rt = _cuda()
        x = self._gpu
        buf = rt.cos(x)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad,
                               _parents=(self,))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.cos_bwd(x, g))
        out._backward = bw
        return out

    def _sin_cuda(self):
        rt = _cuda()
        x = self._gpu
        buf = rt.sin(x)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad,
                               _parents=(self,))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.sin_bwd(x, g))
        out._backward = bw
        return out

'''
    init = init[:m] + DEVPATH + init[m:]
    n += 1

if "    def cos(self):" not in init:
    # ── 4. Tensor 方法：接在 abs 之后 ──
    anchor4 = "    def sqrt(self):\n        return self ** 0.5"
    p = init.find(anchor4)
    if p < 0:
        print("找不到 sqrt 锚点")
        sys.exit(1)
    METH = '''    def cos(self):
        """余弦。为 RoPE 一类位置编码准备（P100 报告指出的迁移缺口）。"""
        if self.device == "cuda":
            return self._cos_cuda()
        out = Tensor(np.cos(self.data), requires_grad=self.requires_grad,
                     _parents=(self,))
        s = np.sin(self.data).astype(np.float32)

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, -g * s)
        out._backward = bw
        return out

    def sin(self):
        """正弦。为 RoPE 一类位置编码准备。"""
        if self.device == "cuda":
            return self._sin_cuda()
        out = Tensor(np.sin(self.data), requires_grad=self.requires_grad,
                     _parents=(self,))
        c = np.cos(self.data).astype(np.float32)

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, g * c)
        out._backward = bw
        return out

'''
    init = init[:p] + METH + init[p:]
    n += 1

io.open(CU, "w", encoding="utf-8", newline="\n").write(cu)
io.open(INIT, "w", encoding="utf-8", newline="\n").write(init)
print("改动处数:", n)
try:
    ast.parse(io.open(CU, encoding="utf-8").read())
    ast.parse(io.open(INIT, encoding="utf-8").read())
    print("语法 OK")
except SyntaxError as ex:
    print("语法错误:", ex)
    sys.exit(1)
