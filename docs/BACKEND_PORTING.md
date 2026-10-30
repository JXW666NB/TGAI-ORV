# 后端移植指南（Vulkan / ROCm / 新硬件）

> 目标：**接一个新后端，不需要改训练代码一行**。所有算子调用都走
> `tgai_go.train.backend.get()`，新后端只要实现接口即可。

## 1. 接口分三层（先看这个，别被 79 个方法吓到）

| 层 | 数量 | 说明 |
| --- | --- | --- |
| **核心 core** | **54** | dense 模型训练必做。少一个就跑不起来 |
| **可回退 fallback** | 8 | 融合核（silu_mul / fused_rope / attn_mask_softmax）。可先不实现，走组合路径，会慢一些 |
| **可选 optional** | 17 | MoE 专用。不做 MoE 可以完全不理会 |

实测（2026-10-30）：`python tools/backend_port.py numpy_ref` → 核心 **36/54**
（剩下的都是计算密集项：rmsnorm/attention/embedding/ce/adamw）。

## 2. 最小接入步骤

```python
# 你的后端模块
class MyBackend:
    name = "vulkan"          # 或 "rocm"
    def empty(self, shape, dtype): ...
    def h2d(self, arr, dtype): ...
    # ... 按 backend.core_methods() 逐个实现

def make():
    return MyBackend()

# 登记
from tgai_go.train import backend
backend.register("vulkan", make)
```

然后：

```bash
python tools/backend_port.py my_backend:make --verbose   # 看还差什么
TG_BACKEND=vulkan python -m tgai_go.train.doctor         # 算子级自检
TG_BACKEND=vulkan python tools/accept.py --quick         # 端到端验收
```

## 3. 语义契约（每条都必须一致，否则数值对不上）

* 缓冲区是**连续行优先**，默认 float32；`h2d/d2h` 同步；`sync()` 之前其余操作异步；
* `sgemm(A,B)` = `A @ B`；**`gemm_nt(A,B)` = `A[M,K] @ B[N,K]^T`**；`gemm_tn(A,B)` = `A[K,M]^T @ B[K,N]`；
* `*_strided_batched` 的 batch 维在**最外层**；
* `softmax_lastdim` 沿最后一维、数值稳定（先减最大值，我们的 CUDA 版就是这么做的）；
* `read_scalar` 只能读 size==1 的缓冲；
* 反向算子的梯度布局与前向输出一致。

**参考实现**：`python/tgai_go/train/backends/numpy_ref.py`
（36/54 核心方法，纯 numpy，语义已用测试对齐 —— 照抄结构换掉底层调用即可）。

## 4. 各后端的具体差异

### CUDA（已完成，作为基准）
cuBLAS 做 GEMM，NVRTC 运行时编译融合核，CUDA Graph 捕获整个 step。

### ROCm / HIP（最省力的一个）
| 项 | 改动 |
| --- | --- |
| 运行时 | `libcuda.so` → `libamdhip64.so`，`cuLaunchKernel` → `hipLaunchKernel`（ctypes 层换个库名 + 函数名） |
| GEMM | `libcublas` → `librocblas`；`cublasSgemm` → `rocblas_sgemm`（注意 row/col-major 约定相反，转置标志要翻） |
| 编译 | NVRTC → **hipRTC**（API 形状几乎一样） |
| 内核 | 我们现有 CUDA 内核多数可直接用 HIP 编译；**注意 wavefront 64**（AMD）而不是 warp 32 |
| 显存 | HIP 的 `hipMalloc`/`hipMemcpy` 对应关系一一对应 |

**风险点**：我们内核里写死了 `__shfl_xor_sync(mask, ..., 32)` 的 warp 假设（softmax/attention 用了 warp-per-row），
AMD wave64 需要改成 `__builtin_amdgcn_ds_bpermute` 或按 64 重写规约。

### Vulkan（最费力的一个）
Vulkan 没有 cuBLAS/NVRTC，**一切都要自己来**：

| 项 | 需要做什么 |
| --- | --- |
| 内存 | `VkDeviceMemory` + staging buffer 做 h2d/d2h；需要自己管内存池（我们是 GpuBuf 引用计数） |
| GEMM | 自己写 tiled GEMM（shared memory + register blocking），或接第三方（如 clspv/glslang 编译的 SPIR-V 库） |
| 内核 | GLSL/SPIR-V。我们的 kernel 是 CUDA C，**需要重写或经 HIP→SPIR-V 工具链** |
| 调度 | 没有 stream 概念，用 command buffer + fence；`sync()` 对应 `vkWaitForFences` |
| 精度 | float32 一律可用；fp16/bf16 需要 `shaderFloat16`/`storageBuffer16BitAccess` 特性，且**累加必须手动升 fp32** |
| 现状 | 仓库里 `shaders/` 只有矩阵乘的骨架 ⇒ **从核心 54 个方法起步，先跑通 dense 前向，再谈融合核** |

**建议顺序**：Vulkan 先只做「核心里的 GEMM + 逐元素 + 归约 + softmax」，
用 `TG_BACKEND=vulkan python tools/backend_port.py` 看进度，能把 `accept.py` 的形状矩阵跑绿再往下做。

## 5. 验收标准（新后端要过这几关）

| 关卡 | 命令 | 判据 |
| --- | --- | --- |
| 接口齐 | `python tools/backend_port.py <spec>` | 核心方法 54/54 |
| 算子数值 | `TG_BACKEND=x python -m tgai_go.train.doctor` | 12/12 PASS |
| 端到端 | `TG_BACKEND=x python tools/accept.py --quick` | 6/6 PASS |
| 与 CUDA 对比 | 同输入跑 `tools/profile_step.py --gpu` | 每步 relerr ≤1e-5 |
| 性能 | 600 步计时 | 达到 CUDA 的 ≥50% 才值得继续优化 |

## 6. 不要做的事

* ❌ 不要为了新后端改训练代码（`examples/`、`__init__.py` 的算子调用点）——那正是这层抽象存在的意义；
* ❌ 不要在核心层里偷偷用 numpy 顶替（`numpy_ref` 是模板，不是产品后端）；
* ❌ 不要跳过 `doctor` 直接跑训练 —— 数值坏了很难回溯是哪一层。
