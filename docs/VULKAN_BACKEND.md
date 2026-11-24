# Vulkan 后端：现状与实测（2026-11-03）

## 1. 结论先行

| 项 | 结果 |
| --- | --- |
| 能不能跑 | **能**。AMD Radeon 核显（`integrated-gpu`，Vulkan 1.3.260）上 compute 前向 + 反向都跑通 |
| 数值对不对 | **逐位精确**：relu 前向 `max|diff| = 0.000e+00`，反向 `0.000e+00`（对 numpy 参考） |
| 快不快 | **不快**。比 numpy 慢 20~150×（同步提交 + 回读的必然代价，见 §4）|
| 能不能用 | **还不能**：要真正有收益，必须把数据常驻显存、多算子编进同一条命令缓冲 |

## 2. 用法

```python
from tgai_go.train import ops_vk_ops as V

FWD = """
#version 450
layout(local_size_x = 64) in;
layout(binding = 0) buffer A { float a[]; };
layout(binding = 1) buffer O { float o[]; };
void main() { uint i = gl_GlobalInvocationID.x; if (i < o.length()) o[i] = a[i] > 0.0 ? a[i] : 0.0; }
"""
BWD = """
#version 450
layout(local_size_x = 64) in;
layout(binding = 0) buffer G { float g[]; };
layout(binding = 1) buffer A { float a[]; };
layout(binding = 2) buffer D { float d[]; };
void main() { uint i = gl_GlobalInvocationID.x; if (i < d.length()) d[i] = a[i] > 0.0 ? g[i] : 0.0; }
"""
V.register_elementwise("vk_relu", FWD, BWD)   # 自动接 autograd
```

* GLSL 需要 `glslang`/`glslc`（用 `TGAI_GLSLANG` 指定路径），或者直接给 `.spv`。
* 内核约定与 `ops_cpu` / `ops_cuda` 一致：前向 `binding0..k-1` 是输入、`binding k` 是输出；
  反向是 `binding0 = g`、中间是输入、最后是输出。

## 3. 三个 bug 的定位过程（可复用）

1. **输出全 0**：先分清"命令没跑"还是"跑了没写"。用 `vkCmdUpdateBuffer` 写 0xAB 模式 →
   能读回 ⇒ **命令缓冲区确实执行**，问题在 compute 绑定层。
2. **哨兵法**：把输出预填 `-12345`，dispatch 后检查是否被改写 → 发现**一次都没写**，
   排除"写了但主机读不到"。
3. **真因**：`VK_PIPELINE_BIND_POINT_COMPUTE` 写成 **4**（正确 **1**）。bind point 非法 ⇒
   管线与描述符集都绑不上，**写入被静默丢弃**（Vulkan 的 robust access 不报错）。

顺带修正：
- `VkDescriptorPoolCreateInfo.flags` 传了非法值 0x2（FREE_DESCRIPTOR_SET 是 0x1，这里用 0）。
- 同步屏障掩码：`srcAccess=SHADER_WRITE(0x40) → dstAccess=HOST_READ(0x2000)`，
  stage `COMPUTE_SHADER(0x800) → HOST(0x4000)`。
- 所有 Vulkan 函数必须显式声明 `argtypes`（64 位句柄不声明会被当 int，直接 `OverflowError`）。

## 4. 性能实测（AMD 核显，elementwise `x*2`）

| n | Vulkan | numpy | 比值 |
| --- | --- | --- | --- |
| 1 024 | 0.305 ms | 0.005 ms | 61× 慢 |
| 65 536 | 2.741 ms | 0.018 ms | 153× 慢 |
| 1 048 576 | 29.659 ms | 1.496 ms | 20× 慢 |

* 缓冲池化把小块从 1.86 ms 降到 0.305 ms，但**每次调用的同步提交 + 回读**决定了下限。
* 想真正提速，需要：常驻显存 + 多条 dispatch 编进同一条命令缓冲 + 只在必要时回读。
  这属于"把算子接进训练图"的改造，不是单点优化。

## 5. 工程注意事项

* **测试必须放子进程**：AMD 驱动在同一进程里加载/卸载 `vulkan-1.dll`、或反复建销 instance，
  会让 pytest 进程退出时**段错误**。现在所有触碰 Vulkan 的测试都在子进程里跑，
  父进程只解析 stdout。
* `ops_vk.load_spirv()` 是早期骨架，已被 `ops_vk_ops` 取代；它不再枚举设备（避免崩测试），
  调用会明确指向新 API。

## 6. 与 `.tgop` 插件包的衔接

`manifest.json` 里写 `"vulkan": {"spirv": "x.spv"}`（反向用 `bwd_spirv`），
`ops_bundle.load()` 会调用 `ops_vk_ops.register_elementwise`。本机没有 Vulkan 设备时
自动降级到 `cpu` 或 `python` 档。
