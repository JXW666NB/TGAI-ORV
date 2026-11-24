# 待做：融合 `tanh` 内核（2026-11-13 定位，未实现）

> 这是目前**收益最确定**的一项。本文把要做的事、验收标准、以及为什么不能用 exp 重写写清楚，
> 避免下次又从"要不要用 exp"开始讨论。

## 1. 现状与证据

| 事实 | 数值 | 怎么测的 |
| --- | --- | --- |
| `exp` 有真内核 | 16 MB 输入 → **16 MiB** | `tools/probe_exp_tanh.py`，每次清池 |
| **`tanh` 没有内核** | 16 MB 输入 → **176 MiB（11×）** | 同上 |
| `situ_glu_orv` 一次调用 | 756 MiB（含 3 次 tanh） | `tools/probe_moe_parts.py` |
| MoE 每层留存 | 450 MiB | `tools/probe_split_mem.py` |
| 8 层激活 | 1,980 MiB（PyT 805） | 峰值 − 参数 |

源码位置：`train/__init__.py:1821`

```python
def tanh(self):
    """组合式（2*sigmoid(2x)-1）⇒ CPU/CUDA 都能用。"""
    return _O.tanh(self)
```

`cuda.py` 里搜不到任何 `tanh` —— 从来没有过这个内核。

## 2. 为什么不能用 exp 重写（重要）

`tanh(x) = (e^{2x} − 1)/(e^{2x} + 1)`，当 `x` 较大时 `e^{2x}` 溢出成 `inf` ⇒ `inf/inf = NaN`。
ORV 没有 clamp/abs 的通用保证，训练路径上**一次溢出就毁掉整次训练**，比慢更糟。
`situ_glu` 的三个参数恰好都带 `1/β` 缩放（0.25 / 0.5 / 0.04），"大概率"安全 —— 但训练代码不接受"大概率"。

## 3. 要写什么

### 3.1 前向

```cuda
// out[i] = tanhf(x[i])；一次分配输出，无 workspace
__global__ void tanh_fwd(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanhf(x[i]);
}
```

Python 侧：`def tanh(self)`（或 `runtime.tanh(buf)`）返回新缓冲区，
`Tensor._from_gpu(..., requires_grad=x.requires_grad, _parents=(x,))`，并挂 `_backward`。

### 3.2 反向

```cuda
// dout[i] = g[i] * (1 - y[i]*y[i])　　y 是前向输出（tanh 值），不需要重算
__global__ void tanh_bwd(const float* g, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float t = y[i]; out[i] = g[i] * (1.0f - t * t); }
}
```

反向复用**前向输出** `y`，不再多存一份 `x`。

### 3.3 接入点

* `Tensor.tanh()` → 走新内核（CUDA 分支）；CPU 分支保持原样（numpy 有 `np.tanh`）。
* 数值验收：与 `np.tanh` 比，`max|Δ| < 1e-6`；反向与 `1−t²` 比，`< 1e-6`。

## 4. 验收标准（全部要过）

| 项 | 判据 |
| --- | --- |
| 前向数值 | 与 `np.tanh` `max|Δ| < 1e-6` |
| 反向数值 | 与解析式 `g·(1−t²)` `max|Δ| < 1e-6` |
| `tanh` 单位开销 | 16 MB 输入 **≤ 32 MiB**（现在是 176） |
| `situ_glu_orv` 峰值 | 从 **756 MiB** 降到 **≤ 300 MiB** |
| 整模型 seq2048（8 层 E=8 flash）| 从 **5,820 MiB** 进到 **≤ 4,645 MiB**（即 ≤ PyT 的 1.0×）|
| 套件 | 762 passed 不退步 |

## 5. 风险

| 风险 | 对策 |
| --- | --- |
| NVRTC 编译失败 | 新内核单独一个 launcher，失败时**退回组合实现**并打警告，不静默算错 |
| 与既有 `_tanh` 调用点不一致 | 只改 `Tensor.tanh`，`_tanh()` 是它的薄包装，自动生效 |
| 反向依赖前向输出 | 组合实现的反向本就用输出；新内核保持一致，不额外存 `x` |
