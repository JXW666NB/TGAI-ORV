# -*- coding: utf-8 -*-
"""C 端插件（**优先形态**）：原生库、C ABI、JSON 声明式清单。

## 为什么 C 端优先

用户明确要求：**插件优先 C 端，包括自定义 GPU 后端、算子插件等，其次才是 Python。**

理由很实际 —— 逐元素的 Python 胶水在训练热路径上是灾难。但「插件是 .py」这个
说法本身容易误解，先把事实说清：

> **.py 从来不是热路径。** 它只在**加载时**执行一次，做登记与绑定；
> 一旦进入训练循环，调用的是原生代码。所以 `.py` 的问题不是"慢"，而是
> "它把插件的**实现**也限制在了 Python 里"。

本设计让实现也能是原生的，Pyth 只剩一层可选外壳。

## 四条通道（按原生程度排序）

| 通道 | 形态 | 需要 Python 吗 | 热路径 |
| --- | --- | --- | --- |
| `plugin.load_manifest(json)` | 原生库 + **JSON 清单** | ❌ **完全不需要** | 原生 |
| `plugin.op_c(lib, symbol)` | 预编译库 + C ABI | 一行绑定 | 原生 |
| `plugin.op_native(source, entry)` | CUDA C 源码，**ORV 用 NVRTC 现场编译** | 一行绑定 | 原生 |
| `plugin.op(fn)` | 组合已有算子 | 是 | Python 胶水 |

### ① JSON 声明式（零 Python）

先用自己的工具链编译原生库：

```c
// myops.cu  →  nvcc -shared -o myops.so myops.cu
extern "C" void my_scale(float* y, const float* x, float a, int n);
```

再写一份 JSON 描述怎么调用：

```json
{
  "name": "myops",
  "version": "0.1.0",
  "api_version": "1.0",
  "library": "myops.so",
  "ops": [
    {"name": "scale", "symbol": "my_scale", "args": ["ptr:0", "ptr:1", "f32:2", "int:3"]}
  ]
}
```

```python
import tgai_go.train as T
T.plugin.load_manifest("myops.json")
T.plugin.native_info()["registered"]["scale"]     # {'entry': 'my_scale', 'kind': 'c-abi'}
```

参数声明支持：`ptr:N`（张量设备指针）、`f32:N` / `f64:N`（浮点）、`int:N`、
`raw:N`（原样传，仅标量）。

### ② 预编译库 + C ABI

```python
lib = plugin.load_lib("myops.so")
scale = plugin.op_c("scale", lib, "my_scale",
                    args=lambda y, x, a, n: [y._gpu.ptr, x._gpu.ptr, float(a), int(n)])
scale(y, x, 2.0, y.numel())
```

### ③ CUDA C 源码，ORV 自己编译

不装 nvcc 也行 —— ORV 内置 NVRTC，能现场编译：

```python
SRC = r'''
extern "C" __global__ void axpy_kernel(float* y, const float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = y[i] + a * x[i];
}
'''
launch = plugin.op_native(
    "axpy", SRC, "axpy_kernel",
    grid=lambda x, y, a, n: ((n + 255) // 256, 1, 1),
    block=(256, 1, 1),
    args=lambda x, y, a, n: [y._gpu.ptr, x._gpu.ptr, float(a), int(n)])
```

编译结果按 `(算子名, 架构)` 缓存，只编一次。

## 安全声明（必须讲清）

**加载原生库等于执行任意机器码。** ORV 不做沙箱、不校验来源、不检查签名。
只加载你信任的库 —— 这一点没有折中。

## 与「能力声明」的关系

C 端插件同样要遵守能力纪律：如果它只在某种硬件上正确（例如要求 `subgroup_ops`），
请在注册时声明，让分派在别的后端上拒绝它，而不是跑出错误结果。
见 `docs/PLUGIN_DISPATCH.md`。
