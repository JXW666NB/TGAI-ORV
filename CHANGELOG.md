## [0.9.49] - 2026-11-11

**Muon 3B 整步 35.0 s → 5.78 s（6.05×）**，并**更正一个我说错的结论**。

### 之前的错判

我曾写「Muon 是 FLOP 受限、已到物理下限、没法修」。**这个结论不成立。**
实测拆解（`tools/muon_profile.py`，真 V100）：

| 项 | 耗时 |
| --- | --- |
| 裸 GEMM `[1024,4096]×[4096,1024]` | 0.815 ms（10.54 TFLOP/s）|
| 裸 GEMM `[1024,1024]×[1024,1024]` | 0.233 ms（9.20 TFLOP/s）|
| **`read_scalar(sumsq_sequential)`** | **101.258 ms ← 真凶** |
| 一轮 NS（3 GEMM + elementwise） | 1.143 ms |
| 完整 5 步 NS（单矩阵） | **111.406 ms** |

⇒ **91% 的时间花在「每个矩阵一次单线程顺序归约」上**，不是算力。
纯 GEMM 每矩阵只要 9.3 ms。

### 修法：顺序归约 → 并行归约

`sumsq_sequential` 是单线程 fp32 顺序累加（当年为了与 OpenBLAS 逐位一致）。
`fast=True` 本来就已放弃逐位一致（cuBLAS 单次约 3e-8），所以改用已有的
两趟并行归约 `sumsq_parallel`（`sumsq_partial` + `sumsq_final`）。

| 项 | 修前 | 修后 |
| --- | --- | --- |
| `read_scalar(sumsq_*)` | 101.258 ms | **0.062 ms（快 1633×）** |
| 单矩阵 5 步 NS | 111.406 ms | **10.128 ms** |
| 224 矩阵外推 | 25.0 s | **2.3 s**（纯 GEMM 下限 2.1 s，即下限的 110%）|
| **真 3B 整步（实测）** | **35.0 s** | **5.78 s（6.05×）** |

### 顺带一个**反直觉但重要的发现**

并行版**比顺序版更准**：

| 实现 | 420 万项求平方和（期望 1677.0546）| 与 float64 参考 |
| --- | --- | --- |
| `sumsq_parallel`（分块） | **1677.0546** | **完全相同** |
| `sumsq_sequential`（单累加器） | 1670.0114 | 差 **4.2e-03** |

fp32 单累加器在大 n 上丢了三位有效数字；分块求和误差更小。
所以这次改动**同时更快、也更准**。

### 现状评价（不含糊）

5.78 s/步 **仍比 AdamW fp16 的 709 ms 慢 8×**，所以 **3B 上仍建议用 AdamW**。
但「完全不可用」的标签不再成立：从 50× 落后缩到 8×，且已贴到纯 GEMM 下限。
Muon 更适合 <1B。

### 测试（4 条，`tests/test_muon_ns_numerics.py`，CUDA 专用）

`parallel` 对 float64 参考 ≤1e-5；`sequential` 的漂移记录在 ≤1e-2 内；
`fast=True` vs numpy 参考 ≤1e-5；`fast` vs `slow` 两条路径 ≤1e-5；
CPU 上的 Muon 不受影响。真 V100 上 **4 passed**。

套件：**660 passed / 33 skipped**。

---

## 文档合并：插件系统只有一套说法

之前 `PLUGINS.md`（0.7.x 的算子插件入门）与新增的四篇 `PLUGIN_*` 部分重复，
容易产生两套说法。现已明确分工，并在 `PLUGINS.md` 顶部与相关章节加了指向：

| 文档 | 角色 |
| --- | --- |
| **`PLUGIN_CHARTER.md`** | **规范**（不变量 / 承诺 / 兼容 / 安全 / 评审），冲突时以此为准 |
| `PLUGINS.md` | 算子插件**入门**（怎么写第一个、contrib、打包） |
| `PLUGIN_DISPATCH.md` | 能力模型与分派（实测开销、bind 纪律） |
| `PLUGIN_NATIVE.md` | C 端四通道 |
| `PLUGIN_DSL.md` | 插件语言与各后端代码生成 |

### 宪法新增第 3 条：生成优先、Python 兜底

这是为**插件语言**铺路的核心条款：

| 档 | 形态 | 谁把数学变成机器码 |
| --- | --- | --- |
| **B1 声明式内核** | 用户写数学（逐元素/归约/广播）| **ORV 生成各后端源码并编译** |
| B2 后端源码 | 用户写 CUDA C / GLSL / 厂商语言 | 该后端工具链 |
| B3 预编译库 | 用户编译好的 `.so`/`.dll` | 用户自己的工具链 |
| P 组合式 | 用已有算子拼（Python）| 不需要 |

硬要求：能由 B1 表达的自定义算子**不得**只提供 P 档；缺生成器的后端必须让该插件
**表现为不存在**，不得静默降级；P 档兜底**必须可统计**，禁止静默兜底。
另有一条待实现清单（§9）明确记录 `plugin.dump_code` 与 `plugin.stats` 尚未实现。

## [0.9.48] - 2026-11-11

**`ParameterGroup` 支持 CUDA**（偏移视图），并**修掉导出/移动文件时的两处误用**。

### 之前的状态

0.9.43 只有 CPU 可用，CUDA 直接报错 —— 当时判断是「`GpuBuf` 没有 offset 字段」。
实际不成立：`GpuBuf` 的视图可以**指定任意 `ptr`**，所以偏移视图不需要改 `GpuBuf`。

### 现在

新增 `CudaRuntime.sub_view(buf, shape, elem_offset)`：在底层缓冲内做偏移视图，
共享同一段显存（`view_of` 会挂 `keepalive`、清 `_token`，原缓冲先死不会被误释放），
越界报错不静默。

`ParameterGroup.to_cuda()` 用它把整组搬上设备，各视图落在正确偏移。

### 真 V100 实测（`tools/param_group_gpu_check.py`）

```
1) 偏移视图指针正确: True
2) 视图读到正确切片: True（最大偏差 0.0e+00 / 0.0e+00 / 0.0e+00）
3) 视图参与 GEMM 与独立张量一致: True（最大差 0.0e+00）
4) 原地改底层后视图同步: True
```

### 过程中连踩三个坑（都只有上真卡才暴露）

1. **默认设备是 cuda 时 `Parameter(ndarray)` 立刻建了设备副本**，`.data` 变成 None，
   `h2d(None)` 只得到 1 个元素 ⇒ `sub_view` 报越界。修：构造期间强制 CPU，
   并明确**`self._arr` 是唯一权威主机副本**。
2. **"复用已有的 `packed._gpu`"** 会读到陈旧内容（主机副本已被改）。已删除该优化。
3. **缺少安全写入整组数据的方法** —— `packed.data = ...` 是出式赋值、会断开视图。
   新增 `group.load(array)`：切片原地写、共享不断，已上设备时同步刷新。

### 测试

`tests/test_param_group.py` 增 3 条：`load` 原地写并保持共享、元素数不符报错、
`assert_shared_gpu` 在 CPU 上是无操作。套件：**655 passed / 33 skipped**。

### 附：仓库根目录整理

32 个测量输出（`.txt` / `.log`）原先散在仓库根目录 —— 它们是 README 与
`ROADMAP_3B_PERF.md` 里数字的**原始证据**，所以**移动**到 `docs/measurements/`
并生成索引，而不是删除。`.gitignore` 增加 `/*.txt`、`/*.log` 防止复发。

## [0.9.47] - 2026-11-11

**修复：容量因子在 CPU / CUDA 上丢弃的 token 不一致**（会真实影响用户）。

### 问题

0.9.40 用「截断每个专家的偏移量」实现容量限制，这依赖后端 `moe_compact` 的顺序。
实测两条路径丢的不是同一批 token：

| | 修复前 | 修复后 |
| --- | --- | --- |
| `cf=1.0` 相对差 | **7.57e-01** | **5.88e-07** |
| 丢弃 token | CPU=3，**GPU=0** | CPU=3，**GPU=3** |
| 集合一致 | **False** | **True** |

后果：用户在 CPU 上调试、在 GPU 上训练时，会看到**不该出现的分歧**。

### 修法：改用门控权重掩码

丢弃一个 (token, 专家) 对，等价于把该对的门控权重置 0。
掩码**只依赖 `idx`**（两条路径都从同一份 logits 选出），所以天然一致。

```python
def _capacity_mask(idx, n, cap, n_experts, top_k):
    # 稳定排序 ⇒ 同一专家内部仍是 (token, slot) 升序，与 CPU 扫描顺序完全相同
    order = np.argsort(flat, kind="stable")
    ...
```

同时**删掉**了 CPU 路径里旧的 `tok/slot` 截断 —— 否则两套规则叠加会再次分叉。

### 这个修法不需要新内核

之前判断"需要排序内核，暂不修"是过度悲观：`off`/`tok` 本就在主机侧，
用 `argsort(kind="stable")` 就够。已在 `docs/DESIGN.md` 的技术债表里更正。

### 测试（9 条，+3）

新增三条直接验证掩码规则：
`test_capacity_mask_keeps_first_cap_per_expert`（构造已知分布逐位核对）、
`test_capacity_mask_is_deterministic`（稳定排序）、
`test_capacity_mask_respects_cap_exactly`。

GPU 校验也恢复成**严格判据**（要求丢弃集合一致），不再是"限制下不要求一致"。
真 V100 结果：`cf=None/2.0/1.0` 三档全部 `集合一致=True`。

套件：**652 passed / 33 skipped**。

## [0.9.46] - 2026-11-11

**C 端插件优先**：插件可以是原生库，Python 只是可选外壳。

### 先纠正一个常见误解

「插件是 .py，是不是太慢？」——**.py 从来不是热路径。** 它只在**加载时**执行一次，
做登记与绑定；进入训练循环后调用的是原生代码。所以 `.py` 的问题不是"慢"，
而是"它把插件的**实现**也限制在 Python 里"。本版让实现也能是原生的。

### 四条通道（按原生程度排序）

| 通道 | 形态 | 需要 Python | 热路径 |
| --- | --- | --- | --- |
| `plugin.load_manifest(json)` | 原生库 + **JSON 清单** | ❌ **完全不需要** | 原生 |
| `plugin.op_c(lib, symbol)` | 预编译库 + C ABI | 一行绑定 | 原生 |
| `plugin.op_native(source, entry)` | CUDA C 源码，**ORV 用 NVRTC 现场编译** | 一行绑定 | 原生 |
| `plugin.op(fn)` | 组合已有算子 | 是 | Python 胶水 |

### 实测（真 V100，`tools/native_plugin_check.py`）

```
n=1000      与 numpy 相对差 4.39e-08  ⇒ 一致
n=1048576   与 numpy 相对差 6.94e-08  ⇒ 一致
再次调用未重复编译: True
原生内核 n=1M: 0.0192 ms
```

**NVRTC 现场编译通道已实证可用**，不是"设计好了"。

### 过程中连修三个真 BUG（都在真实 GPU 上暴露）

1. `_compile(source=...)` 收 **str**，我传了 `bytes` → `AttributeError`；
2. `_compile(arch=...)` 收 `"sm_70"` 字符串，我传了整数 `70` →
   `NVRTC: invalid value for --gpu-architecture`；
3. `_compile(want_cubin=True)` 返回的是 **cubin 字节**，不是 CUmodule →
   直接当 module 传会得到 `rc=400 invalid resource handle`；
   必须再 `cuModuleLoadData`。

另外 `launch(args=...)` 要 `(值, ctype)` **对**，不是裸值。

### 关于跨后端可移植性（重要，不是本版能解决的）

C 端插件**天生绑定后端** —— CUDA C 不可能跑在 Vulkan 或国产卡上，这是物理事实。
解法是三档可移植性 + DSL，已写成设计：`docs/PLUGIN_DSL.md`

| 档 | 形态 | 可移植 | 速度 |
| --- | --- | --- | --- |
| A 组合式 | 用已有算子拼 | ✅ 全后端 | 看拼法 |
| B 声明式内核（DSL） | 逐元素/归约/广播描述 | ✅ 全后端自动生成 | 近原生 |
| C 后端原生 | 手写 CUDA C / GLSL / 厂商语言 | ❌ 只该后端 | 原生 |

纪律：**C 档插件必须同时提供 A 或 B 档回退**，否则它在别的后端上应是"不存在"，
而不是"跑出错结果"。DSL 分 S1–S5 五阶段，**S1（逐元素 + CPU/CUDA 两路生成）尚未开工**。

### 安全声明

**加载原生库等于执行任意机器码。** ORV 不做沙箱、不校验来源。只加载信任的库。

### 测试（14 条，`tests/test_plugin_native.py`）

C ABI 包装与缺符号报错、`load_lib` 缓存与缺文件报错、参数声明解析（含越界/坏类型）、
JSON 清单字段校验与端到端绑定、`op_native` 源码校验与无 CUDA 时报错、文档存在性检查。

套件：**649 passed / 33 skipped**（+14）。

## [0.9.45] - 2026-11-11

**修正 0.9.44 的分发缺口**：`torch-compat` 示例插件原来只放在 `examples/plugins/`，
**sdist 与 wheel 都没带上** —— `pip install tgai-orv` 的用户拿不到，
这跟"示例要能让所有人跑起来"的初衷矛盾。

已把插件**移入包内**：`tgai_go/train/plugins/torch_compat.py`，随 wheel 与 sdist 一起分发。

```python
import tgai_go.train.plugins.torch_compat   # 导入即登记（只登记，不安装）
import tgai_go.train as T
T.plugin.activate("torch-compat")           # 显式激活
import torch                                # 拿到的其实是 ORV
```

`examples/plugins/` 保留纯算子示例（softplus / gelu_exact / swiglu / squared_relu）。

### 过程中撞到的一条项目纪律

用 PowerShell 的 `Set-Content -Encoding UTF8` 创建 `__init__.py` 时写入了 **BOM 与 CRLF**，
被项目既有的卫生检查（`test_hygiene.py` / `test_cpu.py`）立刻抓红：
"BOM 会让 Python 报 invalid non-printable character"。
已改用无 BOM + LF 重写。**写 Python 源文件不要用 PowerShell 的 Set-Content。**

套件：**635 passed / 33 skipped**。

## [0.9.44] - 2026-11-11

**`torch-compat` 示例插件**：把 `torch` 这个名字指向 ORV。
这是「**PyTorch 的 API 面本身可以是插件**」这个宣传点的**可运行证据**。

### 用法

```python
import tgai_go.train as T
T.plugin.load("examples/plugins/torch_compat.py")   # 从文件装
T.plugin.activate("torch-compat")                   # 显式激活（会改 sys.modules）

import torch, torch.nn as nn                        # 拿到的其实是 ORV
m = nn.Linear(4, 8)
opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
loss = (m(torch.tensor(x)) ** 2).mean()
loss.backward(); opt.step()                         # 闭环跑通
```

这条能成立，说明**核心不必背 PyTorch 的 API 面** —— 社区一个文件就能补。

### 安全边界（这是重点）

**绝不覆盖真的 torch。** 若 `sys.modules["torch"]` 已被真 PyTorch 占用，本插件
**拒绝安装**并报错，而不是悄悄替换 —— 否则用户会以为在用 PyTorch 而实际跑的是 ORV，
这种混淆比功能缺失危险得多。

这个边界在**全量测试里真实触发了**：套件中其它用例会 `import torch`（真的 PyTorch），
于是 `test_import_torch_yields_orv` 与 `test_pytorch_style_loop_runs` 被正确跳过
（skip 原因是"本进程已加载真 torch，插件会正确地拒绝覆盖"），而不是失败。

### 覆盖范围（诚实声明，不夸大）

| 已覆盖 | 未覆盖 |
| --- | --- |
| `torch.tensor / zeros / ones / randn` | `torchvision` / `torchaudio` 等生态包 |
| `torch.Tensor`、`float32/float16/int64` | `torch.compile`、`torch.jit` |
| `torch.nn`（Module/Linear/Embedding/LayerNorm/RMSNorm/Sequential/ModuleList/functional/MoE） | 二阶导数、`torch.func` |
| `torch.optim`（AdamW/SGD/Adafactor/LAMB/Lion/Muon） | 分布式（DDP/FSDP） |
| `torch.no_grad`、`torch.manual_seed`、`torch.cuda` | CUDA Graph 的公开 API |

**这是最小可运行闭环，不是完整兼容层。** 要更完整，社区在这个文件上继续加即可 ——
这正是插件系统的用途。

### 顺带修掉一个命名限制

`plugin.api` / `plugin.backend` 的名字原先要求是合法 Python 标识符，导致
`torch-compat` 这种最自然的叫法被拒。已放宽：只有会挂成属性的层/优化器/骨架
（`T.nn.<name>`）才要求标识符，登记键允许 `. _ -`。

### 测试（7 条，`tests/test_torch_compat_plugin.py`）

插件文件存在、只登记不自动激活、`import torch` 得到 ORV、**PyTorch 写法闭环跑通**、
**拒绝覆盖真 torch**、卸载复原 `sys.modules`、重复加载幂等。

套件：**635 passed / 33 skipped**（+7 条用例、+2 条正确跳过）。

## [0.9.43] - 2026-11-11

**共享存储参数**（`T.ParameterGroup`）：多个参数共用同一块底层缓冲。

### 为什么需要

MoE 专家权重若连续存成 `[G, N, K]`，一次 `SgemmStridedBatched` 就能算完全部专家；
否则每次都要先把 G 份权重复制成连续缓冲 —— 实测打包开销比省下的还多
（`docs/MOE_MODERN_ARCH.md` 3.1：V2.7 慢 1.30×、256 专家慢 2.22×）。
共享存储参数是「权重连续存」的数据模型基础。

```python
g = T.ParameterGroup([(4096, 1024)] * 8)        # 8 个专家权重
y = x @ g.views[3].T                            # 视图可直接用于前向
opt = T.optim.AdamW(g.parameters(), lr=1e-3)    # 优化器只拿底层那一块
```

### 契约（不遵守会静默出错，所以在**构造时**就拦）

1. **只有 `works_inplace=True` 的优化器能用于本组** —— 出式更新
   （`p.data = p.data - ...`）会把 `.data` 换成新数组、**断开视图**，
   于是"权重更新了、视图还是旧值"。用 `group.optimizer_class(name)` 取优化器时
   由 `plugin.require_inplace` 直接拒绝。
2. **视图不能单独交给优化器**：`group.parameters()` 只返回底层那一块；
   `group.assert_optimizer_ok(opt)` 会检查并报错。
3. `group.assert_shared()` 自检视图确实共享内存。

### 平台状态（如实，不含糊）

| 平台 | 状态 |
| --- | --- |
| CPU（numpy） | ✅ **可用**，视图是切片、天然共享 |
| CUDA | ❌ **本版不支持**：`GpuBuf` 支持共享 `ptr` 的视图，但**没有 offset 字段**；做偏移视图需新建 `ptr+offset` 缓冲，会触碰按 ptr 索引的缓存/提交逻辑。**明确报错，不静默返回错误结果。** |

CUDA 支持是「零打包 → 分组 GEMM 收益」的前置条件，属后续工作。

### 测试（8 条，`tests/test_param_group.py`）

视图共享内存、紧密排布、`parameters()` 不含视图、`works_inplace` 强制、
优化器拿到视图被拒、**实证"出式赋值会断开共享"**（用它来证明风险真实存在）、
输入校验、`repr`。

套件：**630 passed / 31 skipped**（+8）。

## [0.9.42] - 2026-11-11

**能力模型 + 算子分派**：一个插件文件适配全部后端。这是接 Vulkan 的架构前提。

### 机制

插件**针对能力写，不针对后端写**：

```python
plugin.op("my_silu", fast_impl, backend="cuda", caps={"async_copy": True}, min_cc=(8, 0))
plugin.op("my_silu", slow_impl, backend="*")     # 任何后端的通用回退
```

分派按当前后端的**真实能力**（`caps`）与计算能力（`cc`）选实现。
`backend="*"` 的回退参与任何后端，所以 Vulkan 在插件尚未提供专用实现时仍能跑（只是慢），
而不是直接崩。

### 实测分派开销（硬约束的量化）

`dispatch.benchmark_overhead()`（20 万次）：

| 项 | ns/次 |
| --- | --- |
| 直接调用 | 75 |
| 经分派（缓存命中） | 701 |
| **分派净开销** | **约 627** |

| 场景 | 单次耗时 | 分派占比 | 结论 |
| --- | --- | --- | --- |
| 大 GEMM（2048×2048×6144） | 约 400,000 ns | **0.16%** | 可接受 |
| 小 elemwise（[128,4096]） | 约 15,000 ns | **4.2%** | **超标** |

⇒ 因此提供 **`plugin.bind()` 静态绑定**：解析一次后直接调用、零分派开销。
**热路径必须用 bind**，这条写进了 `resolve_op` 的文档、README 与
`docs/PLUGIN_DISPATCH.md`，不是隐藏知识。

### 三条纪律

1. **单次 dict 查表**：键 `(op, backend, dtype, cc_major, cc_minor, caps)`，结果进缓存；
   不做多级 fallback 链、不用反射。
2. **能力不足必须报错**：抛 `DispatchError` 并逐条说明每个候选缺什么
   （缺哪个能力 / 计算能力不满足 / dtype 不符）。
3. **通配回退**保证新后端可用。

### 过程中抓到两个真 BUG

1. **缓存键漏了 `caps`**：先按 `{"async_copy": True}` 解析出快路径，再按 `{}` 查询时
   命中了同一条缓存，**错误地返回快路径**。已把 caps 加入键（字符串化后排序）。
2. **提前返回挡住了通配回退**：只登记了 `backend="*"` 的后端（如 Vulkan）被判为"无实现"。
   已改为先把具体后端与通配两类候选都取到再判空。

### 测试（12 条，`tests/test_plugin_dispatch.py`）

能力门控、计算能力范围、通配覆盖任意后端、dtype 过滤、优先级、缓存命中、
新登记使缓存失效、静态绑定、无实现时报错含原因、开销量化、以及
「条件化实现不进 functional 命名空间 / 无条件算子弟照进」。

套件：**622 passed / 31 skipped**（+12）。

## [0.9.41] - 2026-11-11

**插件系统统一入口：六条轴，一个 `plugin` 命名空间。** 这是对外宣传的卖点之一。

### 为什么现在做

用户提出的关键判断：**如果优化器也是插件，那么「把优化器改成原地更新」就不是动核心，
而是加一个插件。** 这解掉了一个两难 —— 之前为了拿到分组 GEMM 的 1.39×，
需要改优化器的更新方式（出式 → 原地），那会动所有优化器、风险很高；
有了优化器插件，这件事变成「核心提供共享存储参数原语 + 社区写原地优化器」。

### 六条轴

```python
from tgai_go.train import plugin

@plugin.op("softplus")                     # ① 算子（autograd 自动）
def softplus(x): return (x.exp() + 1.0).log()

@plugin.layer("MyBlock")                   # ② 层
class MyBlock(T.nn.Module): ...

@plugin.arch("my_model")                   # ③ 骨架
def build(d=512): return ...

plugin.optimizer("Lion2", Lion2, works_inplace=True)   # ④ 优化器
plugin.backend("vulkan", VulkanBackend, caps={...})     # ⑤ GPU 后端
plugin.api("torch-compat", install_torch_shim)          # ⑥ API 兼容层
```

登记后**立即可用**：`T.build("my_model")` / `T.optim.Lion2` / `T.nn.MyBlock` /
`T.functional.softplus`。

### 三条设计原则

| 原则 | 做法 |
| --- | --- |
| **简易** | 六条轴共用 `plugin` 对象；支持装饰器；**一个 .py 文件装完全部** |
| **通用** | 插件只组合核心算子 ⇒ autograd 自动、同一份代码适配 CUDA/Vulkan/ROCm |
| **安全** | 能力（`caps` / `works_inplace`）必须如实声明，不足时**当场报错** |

第三条是关键：声明为出式更新的优化器**不能**用于共享存储参数 ——
`p.data = p.data - ...` 会把视图换成新数组，导致权重与视图不同步。
这种 bug 极难查，所以注册阶段就拒绝，而不是等训练跑歪。

### 装配顺序（刻意设计）

- 算子 / 优化器 / 层 / 骨架：**纯登记**，注册后立刻可用（无副作用）；
- 后端 / API 兼容层：**只登记不激活**，需显式 `plugin.activate(name)`。

理由是后者会改 `sys.modules` 或切后端 —— 装插件不该在你不知情时改变进程行为。

### 复用已有的 `ops_plugin`，不造第二套

算子轴转发到已有的 `ops_plugin`（0.7.x 起就有）：组合式注册、自带反向、
`entry_points` 发现、契约校验（`api_version` 主版本匹配 + 声明的算子必须真注册 +
`test_inputs` 数值自检）、`manifest()` 带 sha256 可复现。

### 测试（17 条，`tests/test_plugin_system.py`）

含端到端：**一个 .py 文件把六条轴全用上**，然后验证算子真能算、骨架真能建。
另有能力拒绝、重名需 `replace=True`、非法名字/类型报错、`summary()` 单行等。

套件：**610 passed / 31 skipped**（+17）。

## [0.9.40] - 2026-11-11

**MoE 支持容量因子与残差旁路**（Switch / GShard / Mixtral / DeepSeek 的工程必配项）。

### 语义

```
C = ceil(α · T · k / N)        每个专家最多接收 C 个 token
```

超过容量的 token 在本层**拿不到该专家的贡献**（MoE 输出记为 0），
但它在 block 里的残差 `x` 仍原样流过去 —— 这就是 **residual bypass**。
Switch / Mixtral / DeepSeek 都是这个做法：丢的是「专家加工」，不是「整个 token」。

多卡 Expert Parallel 必须设（要静态分配缓冲）；单卡可不设。

### 新增 API

```python
T.nn.MoE(..., capacity_factor=1.25)   # 默认 None = 不限制，与旧行为逐位相同
```

### ⚠️ 一个必须知悉的限制（实测发现，已写进代码注释）

**开启容量限制后，CPU 与 CUDA 保留的 token 集合可能不同。**
两边都严格遵守 cap，但"丢哪些"不一致 —— 因为 CPU 的排序序与 CUDA `moe_compact`
的到达序不同。

实测（cap=24，96 token）：

```
截断前 counts = [17, 21, 30, 27, 26, 16, 26, 29]
截断后 counts = [17, 21, 24, 24, 24, 16, 24, 24]     ← cap 生效，两边一致
CPU 有 3 个 token 两个专家全被丢；CUDA 为 0
```

⇒ **用容量限制时请固定用同一个后端。** 要做到跨后端一致，需要按门控权重排序后再截断
（GShard 的优先路由），那需要一次排序内核，本版未做。

### 顺带修掉一个真 BUG

`_clamp_offsets` 里用了 Python 内置 `min`，但 **ORV 在模块命名空间里把 `min`
遮蔽成了张量 API**（`_api_min`），导致在 CUDA 上抛「max/min 尚未实现」。
已加 `_builtin_min = min` 别名（与已有的 `_builtin_max` 对称）。

这个坑值得记下来：ORV 模块内**不能直接用 `min`/`max` 做标量比较**。

### 测试（6 条，`tests/test_moe_capacity.py`）

1. `capacity_factor=None` 与旧行为逐位相同；
2. 容量公式 `ceil(α·T·k/N)` 正确（多组 T/α 交叉验证）；
3. 容量收紧时确实有 token 被丢（MoE 输出整行为 0）；
4. 被丢 token 的反向正常（无 nan 梯度）；
5. **共享专家不受容量限制**（它不参与路由）；
6. `capacity_factor ≤ 0` 报错。

套件：**593 passed / 31 skipped**（+6）。

## [0.9.39] - 2026-11-11

**MoE 支持 Z-loss**（ST-MoE, arXiv:2202.08906）—— router 的数值稳定手段。
这是现代 MoE 补齐清单的第 4 项（前两项是 0.9.37 共享专家、0.9.38 aux-loss-free）。

### 为什么需要

fp16 下 router logits 容易溢出（早期训练 router 权重摆动大），溢出会让 softmax
退化甚至 nan。Z-loss 惩罚 logit 的绝对幅值：

```
L_z = β · Σ_t (logsumexp(s_t))²
```

### 实现技巧（避免新增内核）

`log_softmax(x)_j = x_j − lse`，所以 `x_j − log_softmax_j === lse` **对每一列都成立**。
取均值即得 lse —— 复用已有的 `log_softmax`，不新写 logsumexp 内核，
也不会因为专家数 E 而损失精度。

### 新增 API

```python
moe = T.nn.MoE(..., z_loss_coef=1e-3)    # 默认 0.0 = 关闭，与旧行为一致
y = moe(x)
loss = task_loss + moe.z_loss            # 需要调用方自己加，ORV 不替你改 loss
```

**注意**：ORV 不把 z_loss 自动加进你的 loss —— 隐式改 loss 会让调试变难。
`moe.z_loss` 是一个带梯度的标量张量，未开启时为 `None`。

### 实测

| 项 | 结果 |
| --- | --- |
| 与手算 `β·Σ(logsumexp)²` 的相对差 | **6.48e-08** |
| CPU vs CUDA | **0.00e+00（逐位相同）** |
| GPU 上梯度到达 router | ✅ |
| 开/关 z_loss 的前向输出 | **逐位相同**（它只多返回一个 loss 项） |

### 测试（6 条，`tests/test_moe_z_loss.py`）

1. 默认关闭，`z_loss is None`；
2. 数值与手算一致（≤1e-6）；
3. logit 幅值变大时 z_loss 单调变大（这才是"压制幅值"的作用）；
4. 梯度能流到 router；
5. **不影响前向输出**（逐位相同）；
6. 与共享专家 + aux_free 同时开启也正常。

套件：**587 passed / 31 skipped**（+6）。

## [0.9.38] - 2026-11-11

**MoE 支持 aux-loss-free 负载均衡**（DeepSeek-V3, arXiv 2408.15664）。
这是现代 MoE 的第二块拼图，接在 0.9.37 的共享专家之后。

### 为什么补这个

纯 top-k 没有均衡机制，容易「路由塌缩」：少数专家被反复选中 → 更新更多 → 更易被选中。
传统做法是加辅助损失（Switch 的 `α·N·Σ f_i·P_i`），但**辅助损失以梯度形式注入 router**，
和主任务梯度打架。V3 改为在**图外**用符号更新调偏置，主梯度保持干净。

### 语义

```
选专家：   s_i(x) + b_i          ← 含 bias
门控权重： g_i(x)                 ← 用**原始**分数，bias 不参与
更新：     b_i ← b_i − u·sign(c_i − c̄_i)    （每步一次，图外）
```

`b_i` 是 **buffer 而非 Parameter**：不进梯度图、不进 `state_dict`、优化器看不到它。

### 新增 API

```python
T.nn.MoE(d_model, d_ff, n_experts, top_k=2,
         balance="aux_free",    # 默认 "none" = 与旧行为逐位相同
         aux_free_lr=1e-3)      # 符号更新的步长 u
```

### 实测

| 项 | 结果 |
| --- | --- |
| CPU vs CUDA 相对差（aux_free，固定 bias） | **5.04e-07** ⇒ 一致 |
| CUDA 上 bias 更新方向（过载↓ / 欠载↑） | ✅ |
| **bias 归零后与 `balance="none"` 的差** | **0.00e+00** ⇒ 证明 bias 不进门控权重 |
| 负载均衡改善（合成用例，30 步） | 最大/均匀负载比 **1.89 → 1.54**（约 −19%）|

### 测试（6 条，`tests/test_moe_aux_free.py`）

1. `balance="none"` 时 bias 恒为 0，输出与旧行为逐位相同；
2. **bias 只改变选择、不改变门控权重**（bias 归零 == 无 bias，逐位）；
3. 符号更新方向正确（过载降、欠载升）；
4. `b_i` 不在 `state_dict`、参数数量不变；
5. **确实能均衡负载**（用最大/均匀负载比量化，1.89 → 1.54）；
6. 非法 `balance` 值报错。

### 已知限制（写清，不藏）

`b_i` 是运行时 buffer，**不进 `state_dict`** —— 断点续训会丢掉它（与 V3 论文的处理一致，
它是控制量不是模型参数）。若需要严格续训，需另行保存。

套件：**581 passed / 31 skipped**（+6）。

## [0.9.37] - 2026-11-11

**MoE 支持共享专家**（`n_shared`）—— 现代主流 MoE 的标配，此前完全不支持。

### 为什么补这个

调研现代 MoE 配置（`docs/MOE_MODERN_ARCH.md`）：

| 模型 | 路由专家 | top-k | 共享专家 |
| --- | --- | --- | --- |
| Mixtral 8x7B | 8 | 2 | 无 |
| DeepSeek-V2 | 160 | 6 | 有 |
| **DeepSeek-V3** | **256** | **8** | **1** |
| Llama 4 Scout / Maverick | 16 / 128 | 1 | **1** |

ORV 之前 `shared` 在代码里出现 **0 次**，意味着训不了 DeepSeek-V3 / Llama 4 这类结构。

### 语义（DeepSeekMoE）

```
y = Σ_{i∈TopK} g_i · E_i(x)  +  Σ_j S_j(x)
    └─ 路由部分（加权）─┘      └ 共享部分（不加权、全 token）┘
```

共享专家**恒定激活、不参与路由、不带门控权重**，作用是吸收通用知识，
让路由专家专注特化。

### 实现选择：加在 `_forward_flat` 这一层

好处是 **CPU 与 CUDA 两条路径自动同时生效**，不会出现"一条实现、一条没实现"
的数值不一致（这是 ORV 一直守的纪律）。已实测确认。

### 实测（真 V100，`tools/shared_gpu_check.py`）

| 项 | 结果 |
| --- | --- |
| CPU vs CUDA 相对差（n_shared=1） | **6.09e-07** ⇒ 一致 |
| n_shared=0 前向 | 3.10 ms |
| n_shared=1 前向 | **3.25 ms（+0.15，约 +5%）** |
| n_shared=2 前向 | 3.35 ms |
| GPU 上共享专家拿到梯度 | ✅ |

**每个共享专家只增加约 5% 前向耗时**，代价很小。

### 新增 API

```python
T.nn.MoE(d_model, d_ff, n_experts, top_k=2,
         n_shared=1,          # 共享专家数量，默认 0 = 与旧行为完全一致
         shared_d_ff=None)    # 共享专家的 d_ff，默认等于 d_ff
```

### 测试（7 条，`tests/test_moe_shared_experts.py`）

1. `n_shared=0` 与旧行为**逐位相同**（向后兼容）；
2. 输出 == 路由输出 + 共享输出，且共享部分**不带门控权重**（相对差 ≤1e-6）；
3. 把路由专家清零后输出只剩共享部分 ⇒ 证明共享专家**看到全部 token**；
4. 梯度能流到共享专家；
5. 参数按每个共享专家 3 个矩阵线性增长；
6. `shared_d_ff` 可独立设置；
7. `n_shared < 0` 报错。

套件：**575 passed / 31 skipped**（+7）。

## [0.9.36] - 2026-11-11

补上分组 GEMM 的 **pytest 守卫**（验收原文要求「数值对照 ≤1e-5 且有 pytest 守住」）。
四条断言已在真实 V100 上全部 PASS。

### 新增 `tests/test_moe_grouped_fallback.py`

| 用例 | 锁住的保证 | 真 GPU 结果 |
| --- | --- | --- |
| `test_non_uniform_shapes_fall_back_and_match_exactly` | 形状不一致时回退，与逐次 `gemm_nt` **逐位一致** | PASS（差 0.00e+00） |
| `test_uniform_shapes_never_crash_and_stay_correct` | 形状一致时走分组，**无论成败不崩、数值 ≤1e-5** | PASS |
| `test_moe_grouped_switch_defaults_off` | `TGAI_MOE_GROUPED` 默认关闭 ⇒ 关闭时行为与加分组前一致 | PASS |
| `test_grouped_helpers_exist` | 运行时原语存在，避免静默回退掩盖缺失 | PASS |

无 CUDA 的机器上后两条自动 skip（本地实测 1 passed / 3 skipped）。

### 为什么必须锁住这几条

2026-11-11 之前 `_grouped_ready()` 按 Ex 版签名调非 Ex 版符号，
驱动把指针低 32 位当 `m` 读 ⇒ **必然非法访存（rc=700）**。
因为当时它默认关闭，所以**没人发现**。现在有四道断言挡住回归。

### 工具

`tools/run_grouped_tests.py` —— 在无 pytest 的机器上直接调用同一批断言函数
（GPU 机器原先没有 pytest，本版已安装 9.2.0，两条路都能跑）。

套件：本地 **568 passed / 31 skipped**（新增 1 pass + 3 skip）。

## [0.9.35] - 2026-11-11

**定论：Muon 在 3B 规模不可用，且这是算法固有限制。** 两种优化都实测无效。

### 实测

| 配置 | 3B 整步 |
| --- | --- |
| Muon，`ns_fast=False`（顺序 FMA，与 OpenBLAS 逐位一致） | 62.9 s |
| Muon，`ns_fast=True`（Newton-Schulz 走 cuBLAS） | **35.0 s**（1.8×） |
| Muon + 按形状分组批量 bmm（672 个同形张量） | **35.0 s（零收益）** |
| AdamW（fp16 动量 + 逐层即时更新） | **0.709 s** |

### 为什么批量也救不了（关键测量）

批量实现把 NS 的算子数从 `30×G` 降到 30（G=8 时约 5× 减少），**步时完全没变** ——
说明瓶颈不是算子数量，而是**总 FLOP**：

- 每个 `[1024,4096]` 矩阵正交化约 **96 GFLOP**（5 步 × 3 次矩阵乘）；
- 3B MoE 有 224 个这样的矩阵 ⇒ 约 **21.6 TFLOP/步**；
- V100 fp32 峰值 15.7 TFLOP/s ⇒ 理论下限 **1.4 秒**，实测 35 秒（带宽受限）。

**即使实现完美，Muon 在 3B 上也至少比 AdamW 慢一个数量级。** 这不是代码问题。

### 处理

- 批量实现（`batch_ns`）**已删除** —— 实测零收益，不留无效复杂度；
- `ns_fast` 保留（真实 1.8× 改进），文档写明它救不了 3B；
- README 的 Muon 段改成上述定论：适合 < 1B，3B 及以上用 AdamW。

套件：**567 passed / 28 skipped**。

## [0.9.34] - 2026-11-11

Muon 新增 `ns_fast` 开关，3B 上步时从 **62.9 s 降到 35.0 s（1.8×）**。
**但仍比 AdamW 慢 49 倍，未到可用范围** —— 如实记录，不做可用性宣称。

### 根因修正（推翻上一轮的猜测）

上一轮把 62.9 s/步归因为"逐张量发射开销"，并按"按形状分组 bmm"去规划。
本轮读代码发现更具体的原因：`_newton_schulz5_gpu` 默认用的是
**顺序 FMA 内核**（`sgemm_mode(..., 1)`），**故意不用 cuBLAS** ——
为了让 GPU 结果与 NumPy/OpenBLAS 的 float32 逐位一致。

于是加了 `ns_fast=True`：同样的 5 步迭代，矩阵乘改走 cuBLAS。实测：

| 配置 | 3B 整步 | 峰值显存 |
| --- | --- | --- |
| Muon，`ns_fast=False`（默认，逐位可复现） | 62.9 s | 18.93 GiB |
| **Muon，`ns_fast=True`（cuBLAS）** | **35.0 s** | 19.17 GiB |
| AdamW（fp16 动量） | **0.709 s** | 24.78 GiB |

### 为什么只快了 1.8 倍（说明不是 GEMM 慢）

3B 有 2550 个 2D 张量，每个要走 5 步 × 约 6 个算子 ≈ **7.6 万次算子调用**。
按 0.4 ms/次估算正好是 30 秒量级 —— 所以瓶颈是**算子数量**，不是单个 GEMM 的速度。
换 cuBLAS 只解决了其中"GEMM 本身慢"的一部分。

**要真正可用，需要按形状分组批量做**（672 个同形张量可以一次 bmm 算完），
把 7.6 万次调用降到万级以下。这是下一阶段的工作，本轮未做。

### 结论（写进 README 前先记在这里）

- `ns_fast` 是**真实的 1.8× 改进**，可以用；
- 但 **Muon 在 3B 上仍不可用**（35 s/步 vs AdamW 0.709 s）；
- 任何"Muon 已可用于 3B"的说法都是错的。

套件：**567 passed / 28 skipped**。

## [0.9.33] - 2026-11-11

README 新增**「未验证清单」**；修掉两个把容差设在噪声地板上的不稳定测试。

### 新增：README「未验证清单」

集中列出 6 项**没有实测支撑**的能力，避免被误读成"已验证"：

| 项 | 状态 |
| --- | --- |
| A100 / H100 上的 bf16 加速 | ❌ 无硬件，未验证 |
| 7B 以上热门模型微调 | ❌ 已验证上限 0.5B，未验证 |
| Muon 在 3B 规模可用 | ⚠️ 实测可用但 62.9 秒/步，不推荐 |
| 分组 GEMM 融合 | ❌ 未生效（libcublas 符号参数布局与文档不符） |
| 多卡 / DDP / FSDP | ❌ 不支持 |
| 自定义 `forward` 自动迁移 | ❌ 不做（原因见 C_END_AND_MIGRATION.md） |

### 修复：两个间歇性失败的测试（测试设计问题，非实现回归）

跑全量时偶发（约 1/16）：

```
FAILED tests/test_clip_accum.py::test_accumulation_sum_mode_matches_torch_sum_reduction
FAILED tests/test_muon.py::test_step_layers_matches_step_within_fp32_noise
```

两个都是**把容差设在了噪声地带上**：

- `test_clip_accum`：跨库（ORV vs torch）fp32 差异的固有量级是 1e-6~1e-5
  （该测试自己的注释就写着这个区间），而判据取 1e-5 —— 正好压在上界，偶发越界。
  改为 **3e-5**（观测上界的 3 倍余量），并保留 print 实测值；真超了就是回归，不是噪声。
- `test_muon`：分层更新 vs 统一更新的残差约 2e-06，判据 1e-5 余量偏小。
  改为**尺度相对判据**（差异必须 < 更新量的 1%），实测占比约 0.02%，天然稳健。

**没有为了让测试变绿而放宽容差**：两处都是修正"判据设错位置"，并写清了依据与观测值。

### 验证

连续 **5 次**全量 pytest：每次 **567 passed / 28 skipped**，无失败。

套件：**567 passed / 28 skipped**。

## [0.9.32] - 2026-11-11

修掉分组 GEMM 的**必然崩溃**，并把它改成**失败即安全回退**。性能收益尚未拿到，如实记录。

### 修了什么

`cuda.py::_grouped_ready()` 加载的是**非 Ex** 版 `cublasSgemmGroupedBatched`，但 `argtypes`
按 Ex 版声明成了指针数组：

```
真签名：... int m, int n, int k, alpha, A[], lda, B[], ldb, beta, C[], ldc, batchCount, groupSize[]
旧声明：... int* m, int* n, int* k, alpha, ...
```

驱动把指针低 32 位当 `m` 读 ⇒ 天文数字 ⇒ **必然非法访存（rc=700）**。
旧注释把它归因为"某专家行数 >64 越界"，是误判 —— 只要分组路径被走到就必崩。

现在：
- `argtypes` 改为正确的标量版；
- `_grouped_sgemm` 先检查**所有组 m/n/k 是否一致**（非 Ex 版的硬要求），不一致就安全回退；
- 任何非 0 返回码都**静默回退**到逐次 `sgemm`，只提示一次，绝不抛异常。

### 实测（`tools/grouped_fix_check.py`）

| 场景 | 修复前 | 修复后 |
| --- | --- | --- |
| 形状一致，走分组 | **rc=700 崩溃** | rc=7（参数校验不过）⇒ 自动回退 |
| 形状不一致 | **rc=700 崩溃** | 回退，**0.00e+00 逐位一致** |
| 库的可靠性 | 一调就崩 | **永不崩、结果永远正确** |

### 未达成的目标（如实）

**MoE 反向 332 ms → ≤200 ms 这一条没做到。** 分组路径现在返回 `rc=7
"parameter number 2 had an illegal value"`（第 2 个参数，即 `transa` 位置），
说明这个 libcublas 版本的非 Ex 版符号**参数布局与文档不一致**，还需要进一步定位
（怀疑：该版本要求 `m/n/k` 在 `transa/transb` 之前，或 `alpha/beta` 需设备指针）。

在调通之前，性能仍是逐专家循环的老路径，**不做任何性能宣称**。

### 顺带确认的事实

- 运行时 libcublas **只有** `cublasSgemmGroupedBatched`，没有 `cublasGemmGroupedBatchedEx`
  与 `cublasSgemmGroupedBatchedEx`（`tools/grouped_ex_probe.py` 实测），所以方案 B（Ex 版）
  在这台机器上不可用。

套件：**567 passed / 28 skipped**。

## [0.9.31] - 2026-11-10

文档收尾：把核心卖点从算术变成实测。

- **PyTorch + 真 AdamW 实测 OOM**：纯 PyT 进程、28 层 MoE（1.997B，比 V2.7 更小）、T=512、fp32、
  开 gradient checkpointing。仅 fwd+bwd 峰值 14.96 GiB；建 AdamW（m/v 需 14.88 GiB）后走一整步
  得到 `OutOfMemoryError`（31.40 GiB 已分配，剩 14 MiB）。而 ORV 在 2.97B 上跑完 **709 ms / 24.78 GiB**。
- **头对头四格补齐**（T=128/512 × fp32/AMP）：ORV 快 1.11~1.16×；但 8 层规模上**显存是 PyT 更省**
  （ORV 多 0.7~3.5 GiB，主要来自 AMP 的 fp16 权重影子缓存）。旧 README 的 1.41~1.74× 已按新口径下修。
- 修掉对拍脚本自身的两个 bug（两边算式不一致、批量替换误伤 PyT block）。

本版仅文档与工具变更，库代码与 0.9.30 相同。

## [0.9.30] - 2026-11-10

**3B 全参整步实测：AdamW 709 ms、峰值 24.78 GiB（纯显存，比 PyTorch Adafactor 基准 911 ms 更快）。**
同时实测出一个必须公开的坏消息：Muon 在 3B 上要 **62.9 秒/步**。

### 实测（V2.7 28 层 2.97B，V100 32GB，逐层即时更新）

| 配置 | 优化器状态 | 峰值显存 | 整步 |
| --- | --- | --- | --- |
| **AdamW fp16 动量** | 11.1 GiB | **24.78 GiB** | **709 ms** |
| Muon fp16 动量 | 5.5 GiB | 18.93 GiB | 62.9 s |

AdamW 比 PyT 的 Adafactor 步（911 ms）**快 1.28×**，且**不用主机内存**。
Muon 峰值低 5.9 GiB，但慢 **89 倍**。

### Muon 为什么慢（这不是实现缺陷，是算法代价）

Muon 对每个 2D 参数做 Newton-Schulz 正交化（5 次迭代矩阵乘）。小模型上可忽略，
但 3B 有 **2550 个参数张量**，逐张量正交化就是 O(张量数 × 迭代数 × 矩阵乘) 的固定开销。
要在大模型上用 Muon，需要按层合并正交化组或降低迭代次数 —— 属于后续工作。

**因此 README 把 AdamW 列为 3B 的默认选择，Muon 标注为"仅显存极度紧张时使用"。**
不把"省了 5.9 GiB"说成优点而不提 89 倍的时间代价。

### 修复：`Fp16MomAdamW.state_bytes()` 把 fp16 动量算成了 5.5 GiB

`m` 与 `v` 是**两个**缓冲：fp16 应是 2×2=4 字节/参数（3B ⇒ 11.1 GiB），
原实现写成 2.0（⇒ 5.5 GiB），会让人误以为还能再多塞一倍。已修正 `bytes_per_param()`。

套件：**567 passed / 28 skipped**。

## [0.9.29] - 2026-11-10

补齐 Muon 单测，修一个**影响所有绑定权重模型**的兼容性 BUG，并实测 AdamW 路线在 3B 上的显存。

### 修复：`Module.parameters()` 没有去重（绑定词嵌入会被更新两遍）

`tie_embedding`（GPT 类模型的常见做法）让 `embed.weight` 与 `lm_head.weight` 指向
**同一个 `Parameter`**，它在 `named_parameters()` 里出现两次。原 `parameters()` 原样返回两份
⇒ 优化器**每步把这个权重更新两遍**，且不报错。实测与"按对象身份更新一次"的结果差 **7.2e-03**。

PyTorch 的 `parameters()` 会去掉重复参数，本次对齐该语义。这个 BUG 影响所有使用绑定权重的
模型（不只是 Muon —— AdamW / SGD 同样中招）。

### 新增：Muon 单元测试（9 条）

`tests/test_muon.py` 覆盖上一版只能靠 GPU 实跑验证的行为：

- 状态**按形状**分配（不是每参数三份），并复核 README 里 2.97B 的 11.1 / 5.5 GiB 推算；
- `fp16_mom` **只把 2D 那部分**减半；
- CPU 更新路径（2D 走 Newton-Schulz、1D 走 Adam 式）；
- `fp16` 与 `fp32` 动量的结果接近（相对差 < 5e-3）；
- `parameters()` 去重 + 预设确实共享同一 `Parameter`；
- 分组覆盖面强制校验、`auto_rest` 补齐、`step_layers` 与 `step` 的差异在 fp32 噪声带内。

### 修正：分层更新**不是逐位一致**

上一版的说法过于乐观。实测残差约 **2e-06**（参数更新量级 ~1e-2，相对 ~2e-4）：
分层更新改变了参数更新顺序，Newton-Schulz 迭代上会累积 fp32 噪声。
测试判据因此取 ≤1e-5，并在注释里写明"这是同一算法的舍入差，不是逐位一致"。

### 实测：两条优化器路线都能在 32GB 上跑完 3B 全参

V2.7（28 层 / 2.97B / MoE），V100 32GB，逐层即时更新：

| 优化器 | 优化器状态 | 峰值显存 | 剩余 | 结果 |
| --- | --- | --- | --- | --- |
| **Muon**（fp16 动量） | 5.5 GiB | **18.93 GiB** | 12.93 GiB | ✅ |
| **AdamW**（fp16 动量） | 11.1 GiB | **24.40 GiB** | 7.46 GiB | ✅ |

Muon 多出 5.5 GiB 余量（可用于更长序列或更大 batch）；AdamW 通用性更好，适合稠密模型。

### 新增 `Fp16MomAdamW.state_bytes()` / `bytes_per_param()`

与 `Muon` 同名 API，便于在训练前判断"这个模型装不装得下"，而不是等 OOM 再猜。

套件：**567 passed / 28 skipped**。

## [0.9.28] - 2026-11-10

**3B MoE 全参训练在 V100 32GB 上跑通**（峰值 18.93 GiB，剩余 12.93 GiB，不用主机内存）。
关键是把 Muon 的状态占用从 33.3 GiB 降到 5.5 GiB。

### 修复：Muon 给每个参数分配了三份状态

旧实现无条件为**每个参数**分配 `buf + _adam_m + _adam_v`：

```
2.97B × 3 份 × 4 字节 = 33.3 GiB     （比 AdamW 的 22.2 GiB 还差，3B 上完全不可用）
```

现在按形状只分配真正用得上的：

| 参数形状 | 走哪条路 | 状态 |
| --- | --- | --- |
| 2D（投影、专家矩阵、embedding） | Muon 正交化 | **1 份动量** |
| 其它（1D 的 RMSNorm、标量） | AdamW | m + v |

再接上 `fp16_mom`（动量存 fp16，内核仍是 fp32 计算，逐张量转换）：

```
2.97B：fp32 动量 11.1 GiB → fp16 动量 5.5 GiB
```

同时新增 `state_bytes()` / `bytes_per_param()`，让"能不能装下"可以有可靠答案而不是估算。

### 新增：分层即时更新（`step_layers`）

反向是逐层的：层 i 的 `dL/dx_i` 算完之后，层 i-1 的反向**不再需要层 i 的权重**。
因此可以此时就用同一个 `t` 更新层 i 并释放它的梯度 —— 同时驻留的梯度只需一层（约 0.4 GiB
而不是 11.1 GiB）。

`set_layer_groups()` 会**强制校验覆盖完整**：绑定权重（`tie_embedding` 时 `embed.weight` 与
`lm_head.weight` 是同一对象）在参数列表里会出现两次，按出现次数更新会让同一参数被更新两遍
（实测差异 1.285e-03）；分组漏掉 `embed/norm/head` 则会**静默少训练**（实测差异 1.1e-02）。
两处都已在代码里拦住，并自动把未覆盖参数收进 `-1` 组。

### 实测（V2.7 28 层 2.97B，V100 32GB）

| 配置 | 优化器状态 | 峰值 | 结果 |
| --- | --- | --- | --- |
| AdamW fp32 动量 | 22.2 GiB | ≥46 GiB 需求 | 不可行 |
| AdamW fp16 动量 + 全梯度驻留 | 11.1 GiB | 33.6 GiB 需求 | **实测 OOM** |
| **Muon fp16 动量 + 逐层即时更新** | **5.5 GiB** | **18.93 GiB** | ✅ 跑通，剩 12.93 GiB |

### README 重写

按"① 这是什么 → ② 优势与疑问 → ③ 新手入门"三段式重写，全文使用项目第三人称
（此前的第二人称"你的显卡"不适合对外宣传）。同步更新显存实测表。

### 已知不足（本轮未完成）

- Muon 的**单元测试尚未补齐**（GPU 实跑验证了分配、fp16、`step_layers`，
  但仓库里还没有对应的 pytest）。下一次发版前补上。

## [0.9.27] - 2026-11-10

重写 README，并补上"从零开始训练自己的模型"这条链路的入口 API。

### README 重写

旧 README 还停留在 0.2.0（结论、数据、功能表全部过时）。新版围绕**一个问题**组织：
"我为什么用 ORV 而不用 PyTorch"，并逐条回答最容易被质疑的点：

- 数学对齐（8 步 loss 相对差 1.44e-07；算子级对照表）
- 能否微调热门模型（Qwen2.5-0.5B 与 HF 相对差 6.29e-06，**并明确标注 7B 以上未实测**）
- 能否训练自己的模型（10 行 + 四步流程）
- 速度（4/8/14 层比 PyT 快 1.41~1.74×；T=512 整步 2038→754 ms）
- **为什么省显存而不是省精度**（逐层重算 + fp16 动量；计算全程 fp32；fp16 动量的余弦相似度 1.000000）
- 与 PyT 的**8 条真实差异**（单卡、自定义 forward 不可自动迁移、`max(dim=)` 不返回索引、
  permute 是真拷贝、`to()` 断梯度链、无 compile、AMP 语义不同、无生态包）

### 新增：新手四步 API

```python
text  = T.load_corpus("my_corpus.txt")          # ① 导入语料（纯文本 / .jsonl）
tok   = T.train_tokenizer(text, vocab_size=4000) # ② 在你自己的语料上训分词表
ids   = tok.encode(text)
data  = T.tokens(ids, seq=128)                   # ③ 切块
model = T.preset.transformer(vocab=tok.vocab_size, d_model=256, n_layers=6, n_heads=8)
hist  = T.fit(model, data, steps=2000, lr=3e-4)  # ④ 预训练
```

- `T.load_corpus(path)`：纯文本或 `.jsonl`（读 `text`/`content` 字段）。
- `T.train_tokenizer(text, vocab_size)` → `T.Tokenizer`（有 `.vocab_size` / `.encode()`）。
- `examples/beginner_pipeline.py` 实跑（纯 CPU，约 140 秒）：**loss 5.5404 → 3.4252（62%）**。

### 补上缺失的顶层导出

`T.SftDataset` / `T.iter_jsonl_texts` / `T.ppo` 之前只在子模块里，README 写 `T.ppo.gae` 会 AttributeError。
现在都在顶层了，并加了 `tests/test_beginner_pipeline.py`（5 条）断言"README 里点名的 API 必须能点出来"。

### 一件必须说清楚的事（没写进卖点）

"V100 单卡 32GB 能全参 AdamW 训 3B" 这条**目前没有跑通实测**：
2.97B 的 fp32 梯度本身就要 11.1 GiB，加上权重 11.39 + fp16 动量 11.1 ⇒ 已超 31.86。
实测确实 OOM。README 里把它标成"尚未验证"，并给出算术，**不当卖点**。

套件：**558 passed / 28 skipped**。

## [0.9.26] - 2026-11-10

`bf16` 接入可编程 AMP（此前只有 `TG_BF16` 环境变量）。

### 变化

- `T.amp(dtype="bf16")` / `T.amp_on(dtype="bf16")` / `T.amp_off(dtype="bf16")` 均可用了。
- `T.amp_state()` 现在分别报告 `fp16` 与 `bf16` 两个开关。
- 退出上下文时**分别恢复**两个开关：原来开 fp16 就恢复 fp16，原来关着就保持关。

### 实现说明（不是新写内核）

GPU 侧 `_sgemm` 早就有 bf16 快路径（`_bf16_wanted` → `cublasGemmEx(CUDA_R_16BF, COMPUTE_32F)`），
只是入口只有环境变量。本次把它接到公开 API，并保留原有的**失败即回退**语义：
任何 cuBLAS 错误都会关掉 bf16 并只警告一次，之后的 GEMM 走原路径（fp16 或 fp32）。

### 实测（V100）

`tools/bf16_check.py`：

```
amp 状态: bf16=True fp16=False
结果与 fp32 **完全一致** ⇒ 运行期已回退 fp32
```

**V100 没有 bf16 硬件**，cuBLAS 报错后按设计回退，结果与 fp32 逐位相同 —— 不存在
"既不是 bf16 精度也不是 fp32 结果"的静默算错。**A100/H100 上才会真正加速，这一点我没有硬件实测**，
所以在 README 里只能写"API 已支持"，不能写"bf16 加速已验证"。

### 测试

`tests/test_amp_api.py` 改为断言 bf16 是合法 dtype、非法 dtype 仍报错。

## [0.9.25] - 2026-11-10

补齐 **PPO / RLHF**（此前只有 DPO/GRPO），修一个**会让策略往反方向学**的梯度 BUG，
并让 `T.preset` 支持 batch>1。

### 新增：PPO / RLHF

- `T.ppo.gae(rewards, values, dones, gamma, lam)`：广义优势估计（含 done 的 bootstrap 终止）。
- `T.ppo.ppo_loss(...)`：PPO 裁剪目标 + 可选的 value 损失与熵项。
- `T.ppo.discounted_returns(...)`：没有 critic 时的最简基线。
- `T.PPOTrainer`：actor-critic 训练器；`value_params` 会自动并进参数集合（否则 critic 静默不训练）。

于是四种训练模式齐了：**预训练 / SFT / DPO / GRPO / PPO(RLHF)**。

### 修复：min 平局导致梯度符号翻转（PPO 与 GRPO 都中招）

`min(s1, s2)` 原先用 `-max(-s1, -s2)` 组合实现。当 `ratio == 1` 时 `s1 == s2` 是**平局**，
`maximum` 把梯度**整份给第二个参数** ⇒ 代理损失对 logπ 的梯度符号翻转
（优势为正的样本，logπ 反而被压下去）。

实测（`tools/debug_ppo_grad.py`）：

| 情形 | 修复前 | 修复后 |
| --- | --- | --- |
| adv=+1, ratio=1 | 梯度 +1（错） | **-1**（对） |
| adv=-1, ratio=1 | 梯度 -1（错） | **+1**（对） |
| adv=+1, ratio=2（应被 clip） | — | **0**（对） |

现在改用**显式掩码**挑较小者（平局时两者相等，取哪边导数都正确）。
`align.grpo_loss` 里同一处写法一并修掉。

### preset 支持 batch>1

`T.preset.transformer(...)` 现在接受 `[seq]` 与 `[batch, seq]`，内部按 `[B*S, D]` 算、
注意力按 `[B*H, S, Dh]` 分组。测试里加了**批次独立性**断言：同一条序列在批量与单独前向中结果一致
（若 batch 维串了会立刻失败）。

### 测试

- `tests/test_ppo.py`（9 条）：GAE 与手写递推一致、done 终止 bootstrap、与 torch 参考的损失与梯度、
  clip 行为、value/熵项系数。
- `tests/test_ppo_trainer.py`（4 条）：step 跑通、critic 真的进参数集合、可学任务上优势高样本的 logπ 上升。
- `tests/test_preset_quickstart.py` 增 3 条批量用例。

套件：**553 passed / 28 skipped**。

## [0.9.24] - 2026-11-10

**T=512 整步 754 ms —— 达成 0.911 s 目标**（此前 1058–1098 ms）。根因是分头小 kernel。

### 根因

V2.7 的 attention 是**逐头循环 16 次**：每层约 80+ 次小 kernel
（16×(Linear + Linear + Linear + qh@kh.T + softmax + pr@vh + Linear)），
每次都不足以填满 V100；同时让每张 CUDA Graph 的 arena 膨胀到 459 MiB，
只捕得下 22–24/28 层。

（先前怀疑是 `permute` 真拷贝导致，实测**不是** —— `fwd_attn` 里根本没有 permute。）

### 修法：堆叠实现

每层用 `cat` 把 16 个 `[DH,D]` 权重拼成 `[D,D]`（反向经 `cat` 自动流回原 `Parameter`，
**不改权重布局、不影响 .tg 加载、不需要重训**），然后：

```
3 次全宽投影 GEMM  →  reshape/permute 成 [NH,S,DH]  →  1 次批量 SDPA（16 头一起）  →  1 次输出 GEMM
```

`fwd_attn` 默认走堆叠；`attn_impl="loop"` 可切回逐头版（保留做对照与回退）。

### 实测（V2.7 28 层，T=512）

数值等价（对照逐头实现）：前向 ≤5.47e-07、`dL/dx` ≤5.75e-07、权重梯度 ≤1.46e-06。

| 段 | 逐头 | 堆叠 |
| --- | --- | --- |
| 前向链 | 411.6 ms | **234.3 ms** |
| MoE 反向 | 347.9 ms | 329.8 ms |
| 注意力反向 | 638.4 ms | **208.0 ms** |
| 词表头 | 14.2 ms | 10.7 ms |
| 合计（逐算子） | 1412 ms | **783 ms** |
| 合计（开图） | 1058–1098 ms | **754 ms** |

单层注意力前向（seq=256）：**9.8 ms → 2.7 ms（3.66×）**。

⇒ 端到端：2038.5 ms（起点）→ **754 ms（2.70×）**，且**低于 0.911 s 目标 17%**。
loss 13.20335388（与逐头版 13.20335865 相差 3.6e-07 相对，属 fp32 求和顺序差异，非逐位一致）。

### 顺带修掉一个严重 BUG

`nn.CausalSelfAttention` 的 head 切分**用错了**（`reshape(B,S,H,Dh)` 直接 ravel 成 `[B*H,S,Dh]`
会把 head 轴与 seq 轴混在一起），实测与逐头参照相差 **9.61e-01**、且不报错、无测试覆盖。
已修为 `permute(0,2,1,3)` 后再 reshape，并加 `tests/test_causal_self_attention.py`（6 条，含 B>1/H>1
与反向梯度）。

## [0.9.23] - 2026-11-10

修一个**静默算错**的 BUG：`masked_fill` 在 CUDA 上返回 nan。

### 问题

```python
out = Tensor(np.where(m, value, self.data), ...)   # self.data 在 CUDA 上是 None
```

`self.data` 是 CPU 张量的 numpy 镜像；**CUDA 张量没有它**。于是 `np.where` 拿到 `None`，
产出 object 数组，最终变成 **nan** —— 既不报异常，也不算对。
这类"静默算错"比崩溃危险得多（用户的因果掩码会直接毁掉训练，而且看不出原因）。

### 修法

CUDA 上改成组合实现：`x*(1-m) + v*m`。只用已有算子，梯度自动只在未掩码处流，
不需要新内核。

### 实测（V100）

| 算子（CUDA vs CPU 同实现） | 修复前 | 修复后 |
| --- | --- | --- |
| `masked_fill` | **nan** | **0.00e+00** |
| `sqrt`（组合） | — | 5.92e-08 |
| `tanh`（组合） | — | 2.39e-07 |
| `gelu`（组合） | — | 7.78e-08 |
| `repeat_interleave` | — | 0.00e+00（且与 `np.repeat` 逐位一致） |

### 顺带做的系统排查

新增 `tools/audit_data_leak.py`：扫描所有 Tensor 方法，找出"用了 `self.data` 但没有 CUDA 分支"的。
结果只有 `masked_fill` 一处是真问题（`__init__` / `dtype` / `to` 属于正常用途）。
其余没有显式 CUDA 分支的方法（`sqrt`/`tanh`/`gelu`/`scale`/`repeat_interleave`）都是**组合实现**，
所以在 CUDA 上本来就可用 —— 上面表格是实测确认，不是推断。

`tests/test_masked_fill_cuda.py`（4 条）守住"CUDA 上不得出现非有限值"这条线。

## [0.9.22] - 2026-11-10

修一个会让 GPU **后续操作全部失败**的 BUG，并把 T=512 的图预算调到实测最优带。

### 修复：捕获失败没有 abort，流会一直停在 capture 状态

`LayerGraphBackward.capture()` 在 `graph_begin()` 成功之后若中途抛异常（例如 arena 用尽），
没有调用 `graph_abort()`。结果是 CUDA 流永久停在"捕获中"，之后任何
`cuCtxSynchronize` / `empty_cache` 都返回
`rc=900 operation not permitted when stream is capturing` ——
**整台机器后续所有 CUDA 操作都失败**，而报错信息完全看不出根源。

实测复现：一次 arena 用尽之后，接着的每一次捕获尝试（哪怕参数完全正确）都失败在 rc=900。

现在用 `try/except` 包住，失败必 `graph_abort()`。修好后同样的探测脚本能连续报出每个配置的真实结果。

### T=512 图数量边界（实测，V2.7 28 层，每张 480 MiB）

| 捕获张数 | arena 合计 | 整步 |
| --- | --- | --- |
| 16 | 7.5 GiB | 1165 ms |
| 20 | 9.4 GiB | 1140 ms |
| **22-24** | 10.3-11.3 GiB | **1058-1098 ms** |
| 26 | 12.2 GiB | 1311 ms（分配器抖动，反而慢） |
| 28 | 13.1 GiB | 失败 `rc=700` |

自动预算改为 `空闲 x 0.65 - 2 GiB`，在 28 层/T=512 上落在 22-24 张这一档。
**T=512 整步：2038.5 ms -> 约 1058-1098 ms（1.86-1.93x）**，loss 与 eager 完全一致（13.20335865）。

### 未达成的目标（如实记录）

判据里的 **"T=512 整步 <= 0.911 s" 未达成**。整步分解显示注意力反向占 638 ms（45%）；
28 层全上图的估算是 862 ms（可达标），但每张图 arena 需 459 MiB，受压只能捕 22-24 层。

**进一步的方向已经定位**：CUDA 上 `permute/reshape` 是**真拷贝**而非视图，
使 arena 有约 3 倍冗余（理论上这一层只需 ~100-150 MiB，实测 459 MiB）。
视图化后 28 张约 4.4 GiB 可全部捕获，预计落到 862 ms 附近。

## [0.9.21] - 2026-11-10

**开箱即用层**：10 行以内开始训练自己的模型；同时扩宽 PyT 迁移面、修掉两处误导性文案。

### 新增：`T.preset` / `T.tokens` / `T.fit`

```python
import numpy as np
import tgai_go.train as T
ids   = np.resize(np.array([1, 5, 9, 13, 17, 21, 25]), 4000)
data  = T.tokens(ids, seq=32)
model = T.preset.transformer(vocab=32, d_model=32, n_layers=2, n_heads=4)
hist  = T.fit(model, data, steps=80, lr=3e-3)
```

- `T.preset.transformer(...)`：pre-LN 的 decoder-only Transformer（RMSNorm/LayerNorm 可选、
  SwiGLU FFN、GQA 未启用、词嵌入与 lm_head 可共享），是**普通 `nn.Module`**，随时可拆开改。
- `T.tokens(ids, seq)`：最小数据入口（定长切块，label 为右移一位）。
- `T.fit(model, data, steps, lr)`：内含 AdamW + cosine(warmup) + 梯度裁剪 + 可选 AMP，返回 loss 历史。
- `examples/ten_lines.py` 就是这条路径：**loss 3.4698 → 0.0264（80 步）**。

### 新增：可编程 AMP

`with T.amp():`、`T.amp_on/amp_off/amp_state/amp_invalidate`、`T.GradScaler`。
`T.amp` 同时保留旧的 bf16 工具语义（可调用 + 属性代理），旧代码不破。

| 项 | 结果 |
| --- | --- |
| GEMM 2048×2048×6144 | fp32 4.33 ms → amp **0.67 ms（6.47×）** |
| 真实 8 层步 | loss 4219.40 → 4219.45（1.25e-05） |
| 退出上下文 | 状态复原 |

### 修复：PyT 迁移面

- `T.nn` 之前**没有** `Conv1d`/`Conv2d`（只在顶层），PyT 代码 `nn.Conv2d(...)` 直接 AttributeError。
  现补进 `nn`，且与顶层是同一对象。
- `from_torch` 新增：`Conv1d`、`Dropout`、`BatchNorm1d/2d`（带 running 统计与 affine 权重）、
  `MaxPool1d/2d`、`AvgPool1d/2d`。
- 修掉两处**误导性文案**：interop 文档与报错都曾写"LayerNorm 不支持，因为 ORV 没有 LayerNorm"，
  这是过时信息（`from_torch` 一直在转 LayerNorm）。报错现在会列出已支持类型。

### 测试

`tests/test_preset_quickstart.py`（7 条，含"loss 必须降到 70% 以下"与"实质代码 ≤10 行"两条硬判据）、
`tests/test_interop_migration.py`（9 条）、`tests/test_amp_api.py`（7 条）。

套件：**531 passed / 24 skipped**。

### 已知环境问题（不是本库缺陷）

`tests/test_rope_scaling.py` 在**全量**运行时会因本机 `transformers → scipy` 的
`FileNotFoundError(WinError 206, scipy.libs)` 而失败（单跑该文件 4 passed）。已改为**显式跳过并打印原因**，
避免把"机器环境坏了"误报成"RoPE 实现错了"。该 scipy 交互未根因定位。

## [0.9.20] - 2026-11-10

把 `TG_FP16` 环境变量升级成**可编程 AMP**：`with T.amp():`，不用再重启进程。

### 新增

- `T.amp(enabled=True)` 上下文管理器；`T.amp_on() / T.amp_off() / T.amp_state()`；`T.amp_invalidate()`。
- `T.GradScaler`：为 PyT 迁移代码保留（`scale_loss / unscale_ / step(opt) / update()`），
  但 **ORV 默认 AMP 不需要它** —— 见下面的设计说明。
- `T.amp` 同时保留旧语义：它以前是 bf16 工具模块（`T.amp.to_bf16_bits`、`T.amp.BF16_KERNELS`），
  现做成"可调用 + 属性代理"的命名空间，**旧代码不破**（`tests/test_amp_bf16.py` 全绿）。

### 设计与 PyT autocast 的差别

| 项 | ORV | PyTorch autocast |
| --- | --- | --- |
| 参数 / 激活 / 梯度 | **全程 fp32** | 激活可为 fp16 |
| GEMM 输入 | fp16（带权重影子缓存） | fp16 |
| GEMM 累加 | fp32 | fp32 |
| 需要 loss scaling | **不需要** | 需要 |

因为梯度和激活始终 fp32，所以不存在下溢；`GradScaler` 只是兼容层。

### 实测（V100）

| 项 | 结果 |
| --- | --- |
| GEMM 2048×2048×6144 | fp32 **4.33 ms** → amp **0.67 ms（6.47×）** |
| 真实 8 层训练步 | loss 4219.40 → **4219.45**（相对差 1.25e-05），梯度有限 |
| 退出上下文 | 状态正确复原（`on=False`，缓存关闭） |
| 数值（大 GEMM） | 相对 fp32 误差 2.78e-04~2.88e-04，与 torch AMP 同值 |

### 边界

- 只支持 fp16（V100 无 bf16；`_sgemm_bf16` 内核已在，接进 Tensor 层属于后续工作）。
- `amp()` 是**全局开关**（GEMM 级），不是 PyT 的逐算子策略；conv/attention 走同一套 GEMM 路由。
- 自定义优化器更新参数后要调 `T.amp_invalidate()`（内置 `Fp16MomAdamW` 已自动）。

## [0.9.19] - 2026-11-10

修正 fp16 GEMM：**大形状真正用上 tensor core**（此前默认阈值把它们全挡回 fp32），并修掉影子缓存里
一处 `hit[0]` 的 BUG。

### 两处根因

1. `_FP16_MAX_K` 默认 **1280** —— K 超过就退回 fp32。而 tensor core 的收益恰恰在**大 K / 大输出**上。
   实测（V100，fp16 输入 + `CUBLAS_COMPUTE_32F` 累加）：

   | 形状 (M×K×N) | fp32 | fp16（修正后） | 加速 |
   | --- | --- | --- | --- |
   | 512×1024×3072 | 0.28 ms | **0.08 ms** | 3.5× |
   | 2048×2048×6144 | 3.86 ms | **0.67 ms** | 5.8× |
   | 2048×2048×8192 | 5.30 ms | **0.94 ms** | 5.6× |
   | 2048×4096×11008 | 13.89 ms | **2.13 ms** | 6.5× |
   | 2048×2048×32000 | 19.24 ms | **3.15 ms** | 6.1× |

   同样形状下 PyTorch AMP 是 0.10 / 0.71 / 0.97 / 2.33 / 3.57 ms ⇒ **ORV 六项全部不慢于 PyT AMP**。

2. `Runtime._cast_f16_cached` 里写成 `return hit[0]`（`hit` 已是 `GpuBuf`，不是元组），
   抛 `TypeError` 后被"fp16 不可用 → 回退 fp32"的兜底吞掉。**上一版据此得出的"fp16 打平 fp32"
   其实是 fp32 在跑**，本次一并更正。

### 数值

fp16 输入 + fp32 累加，相对 fp32 的最大相对误差 **2.78e-04 ~ 2.88e-04**，与 **torch AMP 完全同值**。

### 测试

`tests/test_fp16_thresholds.py`：守住 `_FP16_MAX_K` 默认 >= 1<<30、`_FP16_MIN_OUTPUT` 保持量级，
防止把默认值改回去（读源码断言，因为 `cuda.py` 是 exec 进私有命名空间的）。

## [0.9.18] - 2026-11-10

T=512 的逐层图反向**打通**（此前 `rc=700`），并给出根因与自动预算。

### 根因（实测定位）

同样的脚本、同样的 T=512，只是捕获张数不同：

| 捕获张数 | arena 合计 | 结果 |
| --- | --- | --- |
| 4 | 1.9 GiB | ✅ 通过 |
| 14 | 6.6 GiB | ✅ 通过，整步 **1187 ms** |
| 16（自动预算选的） | 7.5 GiB | ✅ 通过，整步 **1165 ms** |
| 28 | 13.1 GiB | ❌ 重放后读梯度 `cuMemcpyDtoHAsync rc=700` |

⇒ **不是 T=512 的图机制有问题，而是 arena 太多把显存挤爆**（随后前向链/MoE/梯度需要工作集）。
独立驱动 28 张图（不跑 MoE）时是**逐位一致 0.00e+00** 的，进一步佐证。

### 新增

- `LayerGraphBackward.max_affordable(n_layers)`：按当前空闲显存估算能安全捕获多少张；
  规则保守——`空闲 × 0.5 − 2 GiB` 作为可用额度。
- `LayerGraphBackward.capture(xs, max_graphs=None)`：`None` 时自动截断；
  新增 `count` 属性与 `has(i)`，调用方对没有图的层回退逐算子。
- `tests/test_layer_graph_budget.py`：守住"只捕获 N 张 / `has()` 边界 / 重放与逐算子一致"。

### 实测（V2.7，28 层，seq=512）

| 路径 | 整步 | loss |
| --- | --- | --- |
| 逐算子 | 2038.5 ms | 13.20335865 |
| 图反向（自动 16 张） | **1165 ms（1.75×）** | 13.20335865 |

### 代价/边界

- 图数量受显存限制，深模型长序列下**不会**全部层都吃到图加速（本例 16/28 层）。
- 每张图仍然固定占 `arena_bytes`（默认 256 MiB，T=512 建议 480 MiB）。

## [0.9.17] - 2026-11-10

CUDA 上的**组卷积**（`groups>1`），conv 面最后一个缺口。

### 新增

- `conv2d` / `conv1d` / `Conv2d` / `Conv1d` 在 CUDA 上支持 `groups>1`：
  按组切片 → 复用已与 torch 对照到 2.39e-07 的单组路径 → 沿通道维拼接。
- `groups<=0`、或 `groups` 不整除通道数时明确报错并给修法。

### 实测（V100 vs `torch.nn.functional.conv2d`）

| 配置 | fwd | dx | dw | db |
| --- | --- | --- | --- | --- |
| groups=2, s=1, p=1 | 1.11e-07 | 7.81e-08 | 1.55e-07 | 6.64e-08 |
| groups=2, s=2, p=1 | **0.00e+00** | 6.35e-08 | 6.33e-08 | 1.01e-07 |
| groups=4, s=1, p=1 | 7.38e-08 | 5.59e-08 | 2.11e-07 | 1.02e-07 |
| groups=4, s=2, p=1 | 1.01e-07 | 4.21e-08 | 6.95e-08 | 2.39e-07 |

判据 1e-5，全部通过。测试：`tests/test_conv_cuda.py` 新增 groups=2/4 的契约用例
（无 CUDA 时跳过，CPU 侧套件保持全绿）。

### 代价（C3）

组卷积按组串行执行 g 次 gather+matmul，**不是**融合的分组内核；大 groups、大 batch 时明显慢于 cuDNN。

## [0.9.16] - 2026-11-10

把最后 4 个"CUDA 上只能走 CPU"的算子沉到 C 端，并补高层偏好训练器。

### C 端内核（新增 7 个）

- `ewise_abs` / `ewise_abs_bwd`、`ewise_relu` / `ewise_relu_bwd`、
  `ewise_clamp` / `ewise_clamp_bwd`、`ewise_div`。
- 结果：`_cpu_only` 调用点从 4 个降到 **0**，源码里不再有"CUDA 上偷偷回退 CPU"的算子。
- 实测（V100 vs torch）：abs / relu / clamp 的**前向与反向均 0.00e+00**；log 前向 7.50e-08、反向 0.00e+00。

### 新增 API

- `Tensor.to(device)` / `.cuda()` / `.cpu()`：叶子级设备搬运（用于常量、输入、参考 logp）。
- `Tensor.minimum(other)`：此前**完全没有**；CUDA 上用 `0.5*((x+y)−|x−y|)`。
- `Tensor.maximum(other)` 的 CUDA 分支同样改为组合实现（原来走 `.data`，CUDA 上直接崩）。
- `DPOTrainer` / `GRPOTrainer`：高层训练器，负责冻结参考、清零、反向、裁剪、步进、统计。
  用户只提供 `(ids, mask) -> logπ`。

### 修复

- `align.grpo_loss` 在 CUDA 上会算出 **nan**：常量（old/ref/mask）没有对齐设备，且统计用了
  `.data`（CUDA 无主机镜像）。现在常量 `.to(p.device)`、统计走 `to_numpy()`。
  实测修复后 GPU 上 GRPO 一步 loss 有限、梯度范数 0.367。

### 实测（V100 smoke）

| 项 | 结果 |
| --- | --- |
| DPO 20 步（GPU） | loss 0.688 → **0.213**，margin 0.0995 → **14.398** |
| GRPO 一步（GPU） | loss 有限、adv 正确、裁剪范数 0.367 |
| abs/relu/clamp 前反向 | **0.00e+00** |
| CPU 侧 pytest | 508 passed / 20 skipped |

## [0.9.15] - 2026-11-10

补齐 PyTorch 常用归约：`Tensor.max / min / var / std / norm`（前向 + 反向，CPU 与 CUDA）。

### 新增

- `x.max(axis/dim, keepdims/keepdim)`、`x.min(...)`：用 **argmax 的 one-hot mask** 做乘加实现，
  梯度只落到被选中的位置（不需要新内核）；并列最大值取第一个（与 torch CPU 一致）。
- `x.var(axis, unbiased=True)`、`x.std(...)`：两遍法（先均值、再中心化后平方求均值），
  `unbiased` 与 torch 默认一致（除以 n-1）。
- `x.norm(p=2, axis)`：只支持 p=2，其他 p 明确抛 `NotImplementedError` 并给修法。

### 实测（V100 与 CPU 对照）

| 项 | 相对差 |
| --- | --- |
| max / min / var / norm（CUDA vs CPU） | **0.00e+00** |
| std（CUDA vs CPU） | 2.02e-08 |
| max 反向（CUDA vs torch） | **0.00e+00** |
| CPU 侧 10 条 pytest（含 4 种 axis、unbiased 开关、keepdims 形状、p≠2 拒绝） | 全绿 |

### 代价（如实）

`max/min` 在 CUDA 上需要一次 `to_numpy()` 取 argmax，再拷回 one-hot mask —— **每次调用一次主机往返**，
频繁调用/大张量时慢于原生归约。这是"可用且可验证"的实现，不是最优实现。

## [0.9.14] - 2026-11-10

补齐自定义模型常用层：Dropout、MaxPool/AvgPool（1d/2d）、BatchNorm（1d/2d），以及
`Module.train()/eval()`。另附 `docs/REVIEW_2026-11-10_vs_PyTorch.md` 全面审查。

### 新增

- `nn.Dropout(p)`：inverted dropout，训练按 `1/(1-p)` 缩放；CUDA 走 `h2d + _from_gpu` 落同设备。
- `nn.MaxPool1d/2d`、`nn.AvgPool1d/2d`：用 `gather_flat`（前反向已在阶段 B 与 torch 逐位一致）实现，
  **CPU 与 CUDA 同一路径**；`AvgPool` 等价 `count_include_pad=True`（torch 默认）。
- `nn.BatchNorm1d/2d`：两遍法（均值→中心化后方差），只用已有 `mean/mul/add`，CPU/CUDA 通用；
  训练更新 running 统计，`eval()` 用 running 统计。
- `Module.train(mode=True)` / `Module.eval()`：递归到子模块，`self.training` 默认 True。

### 实测（V100，与 torch 对照）

| 项 | 相对差 |
| --- | --- |
| MaxPool2d 前向（CUDA vs CPU / vs torch）与反向 | **0.00e+00** |
| AvgPool2d 前向 / 反向 | 4.76e-08 / 9.69e-08 |
| BatchNorm2d 前向（CUDA vs CPU）与 running 统计 | **0.00e+00** |
| Dropout（CUDA） | 训练均值 0.9990、零比例 0.301、eval 恒等 ✓ |
| CPU 侧 13 条 pytest（含 BN 反向、1d 变体、坏参数拒绝） | 全绿 |

### 已知边界（不隐瞒）

- `AvgPool` 只支持 `count_include_pad=True`；`MaxPool` 在 CUDA 上要一次 `to_numpy()` 取 argmax
  （**主机往返**），并列最大值取第一个。
- `BatchNorm` 只支持 `track_running_stats=True`；`affine=False` 可用。
- CUDA 池化索引用 Python 循环按形状构造，大形状有主机开销；不是融合实现。

## [0.9.13] - 2026-11-10

CUDA 上的卷积（阶段 C2 收口）：**没有新写 CUDA 内核**，用 `gather_flat` + matmul 组合实现。

### 新增

- `conv2d` / `conv1d` / `Conv2d` / `Conv1d` 支持 **CUDA 张量**（`groups=1`）。
  `groups>1` 在 CUDA 上明确抛 `NotImplementedError` 并给修法，不静默算错。
- `_im2col_index_np`：主机侧构造 im2col 平坦索引（列序与 CPU 实现一致）。

### 实测（V100，与 `torch.nn.functional` 对照）

| 项目 | 最差相对差 |
| --- | --- |
| conv2d 前向 / dx / dw / db（6 组配置，含 padding、stride=2、dilation=2） | **2.39e-07** |
| conv1d 前向 / dx / dw / db（3 组配置，含 stride=2、dilation=2） | **1.26e-07** |
| groups>1 | 正确报错 |

判据 1e-5，全部通过。索引本身先用主机 `_im2col` 的输出作基准比对 6 组配置（0.00e+00）。

### 性能代价（C3，如实说明）

- 多一份补丁矩阵（`N*OH*OW*C*kh*kw`，与主机 im2col 同量级）
- `gather_flat` 是**逐元素** gather，不是融合卷积
- 索引在主机侧用 Python 循环构造，大形状有主机开销
- ⇒ **大 kernel / 大 batch 时明显慢于 cuDNN**。这是"可用且可验证"的实现，不是最优实现。

### 测试

- `tests/test_conv_cuda.py`：4 组形状/参数的前向与 dx/dw/db 契约 + groups 拒绝；
  无 CUDA 时自动 skip（CPU 侧套件保持 479 passed / 17 skipped）。
- `tools/conv_index_check.py`、`tools/conv_cuda_parity.py`、`tools/conv_cuda_extra.py` 为可复跑的对照脚本。

## [0.9.12] - 2026-11-10

逐层 CUDA Graph 反向 API（实验性，默认关闭），阶段 A 收口。

### 新增

- `tgai_go.train.layer_graph.LayerGraphBackward`：按层捕获 attention 段（含重算前向）的反向图，
  训练时逐层重放。含 `capture(xs)` / `replay_layer(i, gw)` / `grads_numpy()` / `close()`。

### 实测（V2.7，seq=128，V100）

| 项目 | 数值 |
| --- | --- |
| 单层 attention | 19.32 ms → 4.49 ms（**4.3×**） |
| 28 层 attention | 1016.3 ms → 140.5 ms（**7.24×**），梯度合计 0.00e+00 |
| 真实 28 层反向阶段 | 952.6 ms → 159.8 ms（**5.96×**），梯度 0.00e+00 |
| 整步（同 harness） | 1843.2 ms → 721.3 ms（**2.56×**），loss/梯度 0.00e+00 |
| **生产形状整步** | **1151.2 ms → 664.4 ms（1.73×）**，**目标 ≤1.0 s 达成** |
| API 自测（2 层，参数梯度 + `dL/dx`） | **0.00e+00** |

### 必须知道的边界

- **优化器不在上面的生产形状计时内**：`Fp16MomAdamW` 的 fp16 动量占 11 GiB，与 28 张图的 arena
  同开会 OOM（`live 24.22 GiB` / 31.86 GiB）。优化器单独实测 **47 ms**，合计约 **711 ms**，仍在目标内。
- 图路径**默认不启用**；不启用时行为与 0.9.11 一致。
- `v27_train_run.py` **尚未接线**（`LayerGraphBackward` 已具备，接线是几行的事）。
- 逐位一致的前提：同一图 + 固定缓冲 + 每步 `cuMemcpyHtoD` 写入上游梯度；已实测"换缓冲→结果随之改变"。

## [0.9.11] - 2026-11-10

CUDA Graph 多图支持（逐层捕获/重放），以及 `V27Trainer` 前向拆分。

### 新增（图路径）

- `CudaRuntime.graph_count()` / `graph_launch_slot(i)` / `graph_release_all()`：可同时持有并重放**多张**图。
- 捕获期分配**只走 arena**、不再从池取（`_alloc`），`free()` 跳过 arena 指针（`_in_arena`）：
  否则图节点引用的缓冲会被后续捕获/复用/`empty_cache` 改写或释放（实测表现为第二张图结果不对、
  或 `cuMemCtxSynchronize rc=700`）。
- `reserve_capture_arena()` 支持**多块 arena 堆叠**（一图一块），`release_capture_arena()` 统一回收。
- `_graph_destroy()` 在有 slot 引用时不再释放 arena / exec。

### 新增（模型）

- `V27Trainer.fwd_attn(x, i, cos, sin)`：`ln1 + 16 头注意力 + 残差`，**不含 MoE**（无 D2H，可捕获）。
- `V27Trainer.moe_part(y, i)`：`ln2 + MoE + 残差`（保持逐算子）。
- `fwd()` 现在是 `moe_part(fwd_attn(x))`，算术不变。

### 实测（V2.7，seq=128，单卡 V100）

| 项目 | 数值 |
| --- | --- |
| 单层 attention 图重放 vs 逐算子 | 19.32 ms → **4.49 ms（4.3×）** |
| 28 层 attention 图重放 | 1016.3 ms → **140.5 ms（7.24×）**，梯度合计 **0.00e+00** |
| **真实 28 层反向阶段**（含每步梯度写回） | 952.6 ms → **159.8 ms（5.96×，省 793 ms）**，梯度 **0.00e+00** |
| 每图 arena 占用 | 约 116.5 MiB（256 MiB 预留足够） |

### 边界与未完成（如实）

- 图重放**逐位一致**的前提是：同一图、同一缓冲地址；上游梯度每步用 `cuMemcpyHtoD` 写入固定缓冲，
  已验证"换缓冲后结果随之改变、且与 eager 逐位相同"。
- **训练循环尚未接线**：`v27_train_run.py` 目前仍全逐算子。接线做法已写成
  `tools/graph_28_backward.py`（前向 eager、反向逐层重放），整步 loss 对照与 event 计时待做。
- 图路径**默认不启用**；不启用时行为与 0.9.10 完全一致。

## [0.9.10] - 2026-11-09

embedding 路径设备化（省 160 ms/步，loss 逐位一致），以及若干测量工具。

### 新增 API

- `V27Trainer.x0_from_ids(ids)`：用 `gather_rows` 在设备上构造 `x0`，不再把整张 `[T,D]` 传一遍主机。
- `V27Trainer.add_embed_grad_device(id_buf, g_buf)`：`emb_bwd` 后与 `head.weight.grad` 在设备上相加，
  不再把 `[V,D]`（V2.7 下 128 MB）D2H 相加再 H2D。
- `V27Trainer.forward_all_dev(x0_buf, cos, sin)`：从设备缓冲起步的逐层前向。

### 实测（同一 28 层 3B、seq=128、其余条件完全相同）

```
loss      numpy 路径 14.45316887   设备路径 14.45316887   相对差 0.00e+00
单步时间   numpy 1.158 s            设备 0.998 s           省 160 ms（13.8%）
```

省下的时间与校准对照自洽：旧路径有三次 ~128 MB pageable 传输（约 4 GB/s、每次约 40 ms）。

### 工具（诊断用，非库 API）

- `tools/kernel_count.py`：按内核名统计一层前向/反向的发射次数
  （实测 208 + 334，×28 = 15,176 次/步；elementwise 家族占前向 140/208、反向 261/334）。
- `tools/moe_d2h_cost.py`：MoE 主机往返成本（实测 0.062 ms/层、1.7 ms/步）。
- `tools/zeros_check.py`：校准 memset / h2d / 清零语义（用于判断计时是否可信）。
- `tools/full_step_embed.py`、`tools/embed_timing2.py`、`tools/graph_probe.py`、`tools/graph_layer_test.py`。

### 未完成（与 0.9.9 相同）

A2 图捕获（阻断点：MoE 的 D2H）、GPU conv、单步 ≤1.0 s（当前约 1.20 s）。

## [未发布] - 阶段 A/B/C/E 的后续工作

### 已完成并有实测证据

- **embedding 路径设备化**：`V27Trainer.x0_from_ids` + `add_embed_grad_device` + `forward_all_dev`。
  同一 28 层 3B、seq=128、其余完全一致的两条路径对比：
  `loss 14.45316887 vs 14.45316887（相对差 0.00e+00）`，单步 **1.158 s → 0.998 s（省 160 ms，13.8%）**。
- **A1 融合 fp16-动量 AdamW**：逐位一致，优化器步 **0.375 s → 0.047 s（8×）**。
- **LayerNorm**（0.9.9 已发布）：CPU 与 **CUDA**（前向 4.38e-08 / 反向 9.29e-07）均实测。
- **gather/index_select/scatter_flat**（0.9.6 已发布）：GPU 前向/反向 **0.00e+00**。
- **conv**（0.9.6 已发布，CPU）：9 组配置前向与 dx/dw/db 均 ≤1e-5。
- **DataLoader**（0.9.7 已发布）：顺序批次与 torch **逐元素相同**。
- **torch 互操作**（0.9.8/0.9.9）：logits ≤1e-5，往返参数逐位相同。

### 测量结论（用于指导下一步，不是完成项）

- 一步 **15,176 次 kernel 发射**（一层前向 208 / 反向 334 × 28）；单 elementwise 家族占前向 140/208、
  反向 261/334，计数与 16 个 head 一一对应 ⇒ 发射膨胀来自**按 head 拆开写**。
- MoE 的主机往返（每层一次 D2H）实测只有 **0.062 ms/层、1.7 ms/步**（占 ~1.2 s 的 0.14%）
  ⇒ 它**不是**提速项，而是**图捕获的结构性阻断**（捕获窗口内禁止 D2H）。
- 当前单步 ≈ **1.20 s**（原 1.691 s，−29%），**仍未达 ≤1.0 s**。

### 未完成（如实列明）

- **A2 CUDA Graph**：未达成。驱动/管线可用，阻断点定位到 `MoE._combine_cuda` 的 D2H
  （`train/__init__.py:4059`），需先做 MoE 设备化（固定 grid 的分组合并内核）。
- **C2 的 GPU conv**：未实现（im2col 在主机侧），CUDA 张量会明确抛 `NotImplementedError`。
- **单步 ≤1.0 s**：未达，缺口约 200 ms，方向是减少 per-head 小 kernel（一次全宽 GEMM + 批量 elementwise）。

## [0.9.9] - 2026-11-09



补齐 LayerNorm，并为 CUDA Graph（阶段 A2）做前置修复与诊断。

### 新增

- `T.nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)`：用现有可微算子**纯组合**
  实现（无新内核）；CPU 与 CUDA 均已实测。
- `T.from_torch` / `T.to_torch` 支持 `nn.LayerNorm`（weight/bias/eps/elementwise_affine）。

### 实测

| 路径 | 前向 | 反向 |
| --- | --- | --- |
| CPU（对 `torch.nn.LayerNorm`）| ≤1e-5 | 对输入/γ/β 均 ≤1e-5 |
| **CUDA（V100）** | **4.38e-08** | **9.29e-07** |

互操作往返（含 LayerNorm 的 Sequential）参数**逐位相同**。

### 修复与诊断（为 A2 铺路）

- `Tensor.backward()` 结尾的 `rt.sync()` 在**图捕获期间**跳过（捕获中同步非法，rc=900）。
- 新增 `tools/graph_probe.py`：捕获期间一旦发生 `sync`/`d2h` 就打印调用栈。
  用它定位到 A2 的真正阻断点：**MoE 的 `_combine_cuda` 把每专家偏移量 D2H 回主机**
  （`train/__init__.py:4059`）以驱动 Python 侧的按专家循环。
- 结论（如实）：**逐层图捕获当前不可行**，因为 MoE 的算子是数据相关的（每专家行数不同，
  启动形状要靠主机侧偏移量）。可行方向是把 MoE 改成容量因子（capacity factor）+ 掩码的定形计算，
  或只对稠密部分做图捕获。仓库里 `graph_train.GraphStep` 对**稠密模型**已有 1.71× 实测先例。

### 版本策略

1.0 之前一律 `0.9.N` 递增；`1.0.0` 留给"单卡可替代 PyT 训练自定义模型"的里程碑。
## [0.9.8] - 2026-11-09

阶段 E：与 `torch.nn` 的**权重级**互操作。

### 新增（`tgai_go.train.interop`，导出为 `T.*`）

- `T.from_torch(module)`：torch 层/`nn.Sequential` → ORV 等价层（参数逐位复制）
- `T.to_torch(module)`：反向转换
- `T.state_dict_from_torch(m)` / `T.state_dict_to_torch(o)`：名字**原样保留**的 state_dict 互转
- `T.OrvSequential`：顺序容器；`T.InteropError`：不支持类型的明确报错

支持层：`nn.Linear`、`nn.Embedding`、`nn.RMSNorm`、`nn.Conv2d`。

### 实测（`tests/test_interop.py`，7 项全过）

| 项 | 结果 |
| --- | --- |
| Sequential(Linear, RMSNorm, Linear) 同输入 logits | ≤1e-5 |
| Embedding / Conv2d 单层 logits | ≤1e-5 |
| `to_torch(from_torch(m))` 参数 | **逐位相同** |
| state_dict 键名（`0.weight`/`0.bias`/…）| 双向保持，值逐位相同 |
| 自定义 Module、LayerNorm | 明确 `InteropError`（含"修法"）|

### 已知缺口（不掩饰）

- **`nn.LayerNorm` 不支持**：ORV 目前没有 LayerNorm 实现。遇到会抛 `InteropError`，
  并有测试守住这条路径。这是与 PyT 对齐的**真实缺口**，下一轮补。
- 容器只支持 `nn.Sequential`/`nn.ModuleList`；自定义 `nn.Module` 子类的 forward 无法自动复现。
- 这是**权重级**互操作：不能在 ORV 图里直接调用 torch 层，也不共享 autograd。
## [0.9.7] - 2026-11-09

阶段 D：数据加载抽象。

### 新增（`tgai_go.train.loader`，导出为 `T.DataLoader` 等，并挂到 `T.data.*`）

- `Dataset` / `IterableDataset` / `Sampler` / `SequentialSampler` / `RandomSampler` / `BatchSampler`
- `default_collate`：numpy 数组堆叠、标量转数组、dict/list 递归，字符串原样保留
- `DataLoader(dataset, batch_size, shuffle, sampler, drop_last, collate_fn, num_workers, seed)`

### 实测（`tests/test_loader.py`，对 `torch.utils.data.DataLoader`）

| 项 | 结果 |
| --- | --- |
| 顺序批次（bs=1/2/3 × drop_last True/False）| 内容与顺序**逐元素相同** |
| 显式索引分组 | 与 torch 的 `batch_sampler` **完全一致** |
| `len()`（bs=1/2/3/7/10 × drop_last）| 与 torch 相同 |
| `shuffle=True` | 是合法排列（不重不漏）、同 seed 可复现、不同 seed 不同 |
| `num_workers=4` vs `0` | 批次完全相同 |
| `default_collate`（dict + ndarray + 标量）| 与 torch 一致 |
| 空数据集 / 流式数据集 / 错误提示 | 覆盖（报错含"修法"）|

### 与 torch 的差异（如实标注）

- `shuffle=True` 用本库自己的 `np.random.Generator`，**同一 seed 下的顺序不等于 torch**；
  需要严格复现 torch 顺序时请传 `sampler=`。文档与模块说明都写明了。
- `num_workers>0` 用**线程**而非进程（受 GIL 限制，纯 Python 预处理不会真并行）；
  好处是零拷贝、无序列化、顺序天然稳定。

### 过程记录

本轮我先用 `SimpleNamespace` 覆盖了 `T.data`，而它原本是「数据规范化」模块，直接打挂 12 个
数据层测试；已改为把加载 API **挂到同一个模块对象上**，`T.data` 原有函数保持不变。
## [0.9.6] - 2026-11-09

阶段 C：卷积算子（CPU）。

### 新增

- `T.conv2d(x, weight, bias, stride, padding, dilation, groups)` / `T.conv1d(...)`，含反向
  （对输入、权重、偏置都正确）。
- `T.Conv2d` / `T.Conv1d` 模块（参数名与 torch 一致：weight/bias）。
- 实现方式：**主机端 im2col + matmul**；`Conv1d` 用可微 reshape 视图委托给 2D 路径。

### 实测（对 torch.nn.functional）

| 配置 | 前向 | dx | dw | db |
| --- | --- | --- | --- | --- |
| stride=1 / 2 | ≤1e-5 | ≤1e-5 | ≤1e-5 | ≤1e-5 |
| padding=1（配合 stride 1/2）| ≤1e-5 | ≤1e-5 | ≤1e-5 | ≤1e-5 |
| dilation=2 | ≤1e-5 | ≤1e-5 | ≤1e-5 | ≤1e-5 |
| groups=2 | ≤1e-5 | ≤1e-5 | ≤1e-5 | ≤1e-5 |
| bias=False | ≤1e-5 | ≤1e-5 | ≤1e-5 | — |
| Conv1d | ≤1e-5 | ≤1e-5 | ≤1e-5 | ≤1e-5 |

### 未完成（如实标注）

- **CUDA 卷积未实现**：im2col 在主机侧构造，GPU 路径需要新的 im2col/col2im 内核。
  当前传 CUDA 张量会抛 `NotImplementedError`（含"修法"），**不会静默算错**；
  测试覆盖了这条报错路径。因此目标 C2 的"GPU 也测"一条**未达成**。
- 性能：主机 im2col 比 cuDNN 慢，大 batch/大 kernel 时差距更明显。
- `groups` 走逐组循环实现（正确但非融合）。
## [0.9.5] - 2026-11-09

阶段三：证明"能用 ORV 训**用户自己的模型**"（不依赖 `.tg`，TGAI 只是样例）。

### 新增

- `examples/custom_model_quickstart.py`：纯 CPU、秒级可跑的自定义 Transformer 训练示例，
  演示 `AdamW + CosineLR(warmup) + Accumulator + clip_grad_norm_` 的组合用法。
- `docs/QUICKSTART_CUSTOM_MODEL.md`：安装、建模型、训练循环、一致性与能力边界。

### 实测（同模型、同初始化、同超参）

```
ORV vs PyTorch 各训练 8 步：逐 step loss 最差相对误差 = 1.44e-07   （判据 1e-4）
```

测试：`tests/test_custom_model_parity.py`。

### 能力边界（写进 QUICKSTART，未隐藏）

- 已支持：autograd、7 种优化器、5 种调度器、梯度累加/裁剪/激活重算、预训练/SFT/DPO/GRPO、
  单卡 32 GB 训 3B 全参 AdamW（torch 同卡 OOM）。
- 未支持：多卡；conv 与 gather 等通用算子；内置 DataLoader；与 `torch.nn.Module` 互操作。
## [0.9.0] - 2026-11-09

阶段二：对齐训练形态（DPO / GRPO）。torch 核心不含这些损失，参考实现按论文公式手写并逐项对照。

### 新增（`tgai_go.train.align`，导出为 `T.align.*` 与顶层函数）

- `logsigmoid(x)`：数值稳定（`−softplus(−x)` 形式）
- `seq_logprob(logits, labels, ignore_index)`：回答段 logπ **之和**，保持可微
- `dpo_pair_loss(pw, pl, rw, rl, beta)` / `dpo_batch_loss(...)`：DPO
- `group_advantages(rewards, eps)`：GRPO 组内归一化优势（总体标准差）
- `token_logprobs(logits, labels, ignore_index)`：**逐 token** logπ（可微）
- `grpo_loss(policy_logps, old_logps, ref_logps, advantages, masks, clip_eps, beta, kl)`：
  token 级 PPO 裁剪 + KL（k1 / k3），返回 (loss, stats)

### 实测对照（手写 torch 参考，独立实现同一公式）

| 项 | 判据 | 实测 |
| --- | --- | --- |
| `logsigmoid`（含 ±40）| 1e-6 | ≤1e-6 |
| `seq_logprob`（含 ignore_index）| 1e-5 | ≤1e-5 |
| DPO 单对 loss / 对两侧 logits 的梯度 | 1e-5 | ≤1e-5 |
| DPO batch 平均（beta ∈ {0.01,0.1,0.5}）| 1e-5 | ≤1e-5 |
| `token_logprobs` | 1e-5 | ≤1e-5 |
| GRPO loss / 梯度（beta=0 / 0.1·k3 / 0.1·k1）| 1e-5 | ≤1e-5 |
| GRPO 裁剪触发统计 | — | ratio 远离 1 时 clip_frac>0 |

### 已知限制（写进 docstring，未隐藏）

后端没有 gather 算子，因此：
- `seq_logprob` 逐序列调用（batch 内多几次 kernel）；
- `token_logprobs` 需要一份 [T, V] 的 one-hot，词表大时显存开销明显。

两处都留了"等 gather 就绪再向量化"的说明，接口不需要变。
## [0.8.0] - 2026-11-09

阶段一：训练基础设施对齐 PyTorch（每项都有与 torch 的数值对照测试）。

### 新增

- **学习率调度器** `T.optim.lr_scheduler`：`ConstantLR` / `LinearLR` / `CosineLR`（含 warmup）/
  `StepLR` / `LambdaLR` / `make_scheduler`。lr 序列与 `torch.optim.lr_scheduler` 逐 step 一致
  （float32 存储精度内，判据 1e-6）。
- **优化器** `T.optim.SGD`（momentum / dampening / nesterov / weight_decay）、`T.optim.Lion`、
  `T.optim.Adafactor`。
- **梯度裁剪**导出到 `T.nn.utils.clip_grad_norm_`（torch 同路径）与 `T.clip_grad_norm_`。
- **梯度累加** `T.optim.Accumulator`（tick/finalize/reset）与 `T.optim.scale_grads`。

### 与 torch 的实测对照

| 项 | 判据 | 实测 |
| --- | --- | --- |
| 调度器 lr 序列 | 1e-6 | 逐 step 一致（float32 存储精度）|
| SGD（5 组配置，4 步）| 1e-6 | ≤1e-6 |
| Lion（含 wd）| 1e-6 | ≤1e-6（对独立参考实现）|
| Adafactor（2D/1D/wd/scale_parameter/clip）| 1e-6 | ≤1e-6（对 `transformers.optimization.Adafactor`）|
| 梯度裁剪（3 组 max_norm）| 1e-6 | 范数与梯度均 ≤1e-6 |
| 梯度累加 4×(3 行) vs 1×12 行 | 1e-6 | **1.30e-06**（fp32 求和顺序，见下）|
| 激活重算 开/关 梯度 | 1e-5 | ≤1e-5 |

梯度累加未达 1e-6（实测 1.30e-06）：数学等价，差异来自 fp32 下 4 次小 GEMM 与 1 次大 GEMM 的
求和顺序。测试判据写明 1e-5 并记录实测值，未把"接近"当作达标。

### 修正的 torch 语义（对照测试抓出来的）

- **SGD 首步不做 dampening**：torch 用原始梯度初始化 momentum buffer，之后才乘 `(1-dampening)`。
- **`CosineAnnealingLR` 在 T_max 之后周期回升**（warm restart），不是单调降到 eta_min。
- **模块参数的 `p.data` 是 memoryview**：`p.data[...] = x` 会抛
  `memoryview: invalid slice key`，新优化器统一改用 `np.copyto(np.asarray(...))`。

### 显存（V2.7 3B，28 层，实测）

- AdamW（fp16 动量）：公式 **11.064 GB**，GPU 实测 **+11.072 GB**
- Adafactor（row/col）：公式 **0.021 GB**，GPU 增量小于 1 MiB（走了已预留的池）
- 约 **527×** 状态显存差距

## [0.7.0] - 2026-11-09

### 修复（与 PyTorch 数值对齐，实测 15 步 loss 相对偏差 1.00e-05）

- **注意力缺因果掩码**：`v27_train.py::fwd` 原为 `softmax((q@k.T)*scale)`，**没有 mask**，
  等价于双向注意力（能看到未来 token）。已加上三角 −1e9 掩码。这是语义级 bug：
  修复前单层 loss 与 PyTorch 差 2.56e-02，修复后 1.35e-07。
- **CUDA MoE 的 softmax 作用域错误**：`_combine_cuda` 在**全部专家**上做 softmax 再取 top-k，
  而语义（CPU 路径 / 引擎 `tgai_cuda.cu:301-307` / PyTorch）是**只在选中的 k 个上** softmax。
  现改用 `moe_topk` 返回的选中值 `val_buf` 直接 softmax。实测 scale=8 时 4.55e-02 → 1.42e-06。
- **训练路径丢失输入 embedding 梯度**（tied head）：训练喂 `x0` 用的是 numpy 常量，
  tied embedding 只拿到"输出投影"一侧梯度（实测差 34%）。新增
  `V27Trainer.add_embed_grad()`，`backward_and_step` 记录 `last_input_grad`，
  用 `rt.emb_bwd` 把输入侧梯度散射回 `head.weight.grad`。
- **`emb_bwd` 的 id dtype**：传 int64 会被内核按 int32 解读，梯度落到序列里不存在的行
  （实测 row 0 拿到 0.62，而该 token 根本不在输入里）。改为 int32 后总梯度误差 9.29e-01 → 5.85e-06。

### 新增

- `tgai_go.train.v27_train.V27Trainer.add_embed_grad(ids, g_x0=None)`：补齐 tied embedding 的输入侧梯度。
- `V27Trainer.last_input_grad`：逐层反向结束后保留 `d(loss)/d(x0)`。

### 对齐证据（可复现）

前向 1/2/4 层 loss 相对误差 1.35e-07 / 6.65e-08 / 0.00e+00；分组梯度 ≤7e-6；
tied-embed 总梯度 5.85e-06；AdamW 单步参数 ≤5.09e-08；逐层重算反向 vs 图反向 ≤3.71e-07；
ORV 自比噪声底（双进程）3.20e-07；真实语料 15 步曲线最差 1.00e-05。
全 28 层前向 loss：ORV 1.3444903 vs PyTorch 1.3444886（相对 1.2e-06）。

> 注意：`attention` 由双向改为因果会**改变已有模型的训练/推理行为**，
> 之前用旧版训练的 checkpoint 语义上等价于另一个模型。故按 minor 版本升级。

## [0.5.0] - 2026-11-03

这一版几乎全是**性能与可复现性**，每条都有实测数字。判据与证据见
`docs/GAP_VS_PYTORCH.md` §10~§22 与 `docs/PRODUCTION_READINESS.md`。

### 新增 · 真 AMP（激活存 fp16）
- `train/amp_mlp.py`：MLP 块内部激活存 fp16，**主权重 fp32 + fp32 累加**（标准 AMP 语义）。
- 数值（块级 vs fp32）：前向 **6.36e-04**、dh **7.56e-04**、dW1/2/3 **8.4e-04 ~ 1.17e-03**；
  端到端 3 步 loss 相对差 **1.54e-05**。
- 步时（V100，0.5B 形状，24 层，6 步）：fp32 527.2 → 仅 fp16 GEMM 354.0 → **AMP 263.5 ms**
  ⇒ 对原现状 **1.34×**、对 fp32 **2.00×**。
- 开关 `cfg["MLP_FP16"]`，**默认关**（fp16 有精度地板：1000 步任务 fp16 停在 1.2e-02、fp32 到 2e-06）。
- 过程中修掉的三个自有 bug 记录在 `docs/PRODUCTION_READINESS.md`。
### 新增 · 存储精度 float32 / float16 / bfloat16（新模块 	rain/dtypes.py）
- save_weights(path, weights, dtype="bfloat16") / load_weights(path)：
  检查点可按 fp16 或 bf16 存储（体积减半），文件内**逐张量记 dtype**，允许混合精度。
- bf16 用 uint16 存高位截断（round-to-nearest-even），**指数范围与 fp32 相同**：
  1e30 / 1e-30 都不溢出（fp16 会）。实测 bf16 相对误差 < 2^-8，且只有 8 位尾数。
- cast_weights() / 	o_storage() / rom_storage() 供训练侧按需 upcast。
- 注意：这是**检查点存储**精度；Tensor 的运行时存储仍是 float32
  （fp16/bf16 只用于 GEMM 内部计算）。真 AMP（激活存 fp16/bf16）仍未做。
- 新增 	ests/test_dtypes.py（7 项，纯 CPU）。

### 新增 · 可复现模式（`TG_DETERMINISTIC=1` 或 `rt.deterministic = True`）
- 三处 `atomicAdd` 全部换成确定性归约：rmsnorm 的 dw（按列升序）、
  词嵌入反向（按 id 独占一行）、`repeat_interleave` 反向（按 j 升序，新内核 `repeat_reduce`）。
- **实测：同一配置两次训练，参数最大绝对差 0.000e+00（逐位一致）**；默认模式 7.63e-06。
- 代价约 1%（717.3 → 724.3 ms/step，V100 / Qwen2.5-0.5B 形状 / B=4·S=512）。
- 与 fp16 加速**可同时开启**：492.9 ms/step 且参数逐位一致。

### 新增 · fp16 Tensor Core GEMM（`rt.fp16_gemm = True`，默认关）
- 零转换入口 `gemm_f16_nt_raw` / `gemm_f16_tn_raw` / `gemm_f16_nn_raw`
  （fp16 输入 + **fp32 输出** + **fp32 累加**），配权重影子与激活 scratch。
- 前向 GEMM 156.3 → **28.4 ms（5.5×）**；整步 754 → **489 ms（1.54×）**。
- 已知代价：**精度地板** —— 1000 步记忆任务 fp32 能到 2e-6，fp16 停在 1.2e-02。
  因此默认关闭，只适合常规训练。

### 新增 · fused AdamW（默认开）
- `adamw_update_multi`：290 个 kernel → **1 个**（设备指针表 + 前缀和 + 二分查找）。
- 隔离对拍 **0.000e+00**（与逐参数版逐位一致）；整步 **1.04×**。

### 新增 · CUDA Graph 捕获 arena
- `reserve_capture_arena()`：捕获期 bump 分配，绕开"捕获中禁止 cuMemAlloc"。
- 捕获成功（2441 节点），重放 **748.5 → 605.5 ms（1.23×）**。

### 修复 · 三处"看起来没问题但很致命"的实现
- `ce_row_stats` 是**非合并访问**（每线程连续分块，步长 V/256，仅 37 GB/s）：
  改跨步后 CE 前向 **67.5 → 3.6 ms（18.7×）**。
- `log_softmax_lastdim` 前向/反向是**线程-per-row 串行扫 k**：改 warp-per-row。
- 分配器两处**硬编码 12.2 GiB** 判据（工作集本就 13 GiB）导致每次未命中都整设备同步：
  改为看真实空闲显存 + 节流，步时 **1.125 → 0.898 s（-20%）**。

### 新增 · 算子
- `rope_half`（half-split RoPE 融合内核，前向/反向各 1 个 kernel）：步时 **-14%**。
  注意：现有 `fused_rope` 是**交错式**，与 Qwen/LLaMA 的 rotate_half 不等价（实测差 2.478）。
- `checkpoint_sequential(..., segments=)`（对齐 PyTorch 同名签名）。
- causal softmax 接线（含 n-D 支持）—— 但**实测更慢**（内核每行单线程串行），
  已改为 opt-in（`TG_SOFTMAX_CAUSAL=1`）。

### 工具
- `tools/profile_kernel.py`：CUDA event 级内核剖析（曾用它发现"只 instrument launch
  会漏掉 cuBLAS"的测量错误）。
- `tools/bench_gemm_fp16.py`：fp16 vs fp32 逐形状对照。
- `tools/tg_check.py`：`.TG` 表结构自检（哈希唯一、区间不重叠、越界、空洞）。

### 测试
- 新增 `tests/test_deterministic.py`（GPU-gated，断言"确定性模式下反向两次差 == 0"）、
  `tests/test_rope_half.py`、`tests/test_ckpt_segments.py` 等。
- 合计 **292 passed / 6 skipped**，`tools/check_package.py` PASS。

### 诚实结论（**2026-11-03 已纠正口径**，见 docs/BENCHMARK_RIGOR.md）
- ⚠ 更正：此前写的「对 PyTorch fp32 快 1.25×」**是错的** —— 我们的基准每步都读了一次
  loss（D2H + 同步），虚增了 9~12% 步时，而 torch 那侧没有。
- 同口径（循环内不读设备标量，只在末尾 sync）实测：
  我们 fp32 **683.9 ms** vs torch fp32 **621.0 ms** ⇒ 我们**慢 1.10×**；
  我们 fp16 **389.3 ms** vs torch AMP **219.5 ms** ⇒ 我们**慢 1.77×**。
- 我们 fp16 对 torch fp32：**快 1.60×**（389.3 vs 621.0）。
- torch AMP 的优势主要来自把**激活也存 fp16**；我们只把 GEMM 输入换成 fp16，
  真 AMP（激活 fp16 存储）尚未做。
- CUDA Graph 曾量到 1.23×/1.71×，但那是优化前；同口径重测**只有 1.01×**
  （见 docs/CUDA_GRAPH_FINDINGS.md），已不作为优化方向。
- L3 三项仍未全过（24h 长稳 21.8h、2 卡缺硬件、真 AMP 未做），
  因此对外标签保持 **「可用（自用/小团队）」**。

## [0.6.1] - 2026-11-05

### 修复 / 优化
- **`sum_axis` 占用率 bug**：通用 `sum_axis8` 是「一个输出一个线程」，bias 梯度只有 896 线程
  （不足 1 个 block）。新增 `sum_axis0_rows`（一列一 block、256 线程并行归约），
  **省约 14.5 ms/step**，数值核对 1.4e-07。
- **AMP 补上 `GradScaler`**（`amp_mlp.GradScaler`）：动态缩放 + 每步梯度有限性检查 + 溢出跳步。
  没有它时静态缩放两头不讨好（S=1 梯度下溢学不动；S≥32 先快后溢出尖峰）；
  有它之后 0.5B/1200 步只跳步 **2 次（0.17%）**、scale 自愈 256→8192、无 NaN。
  开销 **+1.2%**（221.4 vs 218.8 ms）。
- `TG_MLP_FP16` / `TG_SDPA_LOWMEM` 环境变量开关，便于基准与剖析工具使用。

### 实测（V100，同口径 V=32768 / B=4·S=512 / 24 层 / 6 步）
- 我们：fp32 **488.5 ms**、AMP **209~221 ms**；PyTorch：fp32 **500.0 ms**、AMP **182.9 ms**
  ⇒ fp32 **反超 1.02×**；AMP 仍慢 1.15~1.21×。

### 文档
- `README.md` 新增 **AMP 章节**：何时开（loss 1.5~4）、何时别开（要压到 <1e-2）、
  为什么必须配 `GradScaler`（附三档实测对照）。
- `docs/PRODUCTION_READINESS.md` 置顶「收官摘要」：判据三项 + 性能对照 + 三个关键发现 + 已证伪方向。
- `docs/ENGINEERING_STANDARD.md` §14「rc=700 先怀疑机器」、§13 文档一致性门禁、
  以及 Qwen3 对拍因主机内存不足失败的环境限制记录。

### 限制（如实）
- AMP 有**精度天花板**：同任务 1200 步 AMP 停在 0.79、fp32 到 5.3e-4。
- ③-2（2 卡）仍缺硬件；③-3（正面对决 ≥ PyTorch）仍未达标。
## [0.6.0] - 2026-11-05

### 新增 · 分块交叉熵（`CHUNK_CE`）—— 把 32 GB 卡跑不动的形状变成能跑
- `chunked_ce()`：lm_head 投影与 CE 按 token 分块，用 `T.checkpoint` 保证反向时重算，
  图里只保留块大小的 logits（不落地 `[N,V]`）。
- `cfg["CHUNK_CE"] = True`（可选 `CHUNK_TOKENS`，默认 8192）即在 `sft()` 生效。
- ⚠ **测量作废（2026-11-01 晚复核）**：先前记的"整块 29.92 GiB OOM / 分块 17.72 GiB"
  不可用 —— 当时 `T.checkpoint` 在 CUDA 上是**空操作**，且 `CHUNK_TOKENS` 默认 8192
  大于 N=2048（实际只有 1 块），因此两条路径的差异**不能归因于分块**。
  待用修好的 checkpoint 重测后再填数字。
- 数值等价：loss 差 4.8e-07、梯度差 <1e-5（`tests/test_chunked_ce.py` 4 项）。


### 新增 · `api.train` 支持 `pretrain`（从零预训练）
- 纯文本语料切定长块（`build_lm_sequences`，支持 `.txt` / 每行 `{"text"}`），
  **全 token 参与 loss**（不做对话模板、不遮 prompt）。
- `cfg` 给结构即可随机初始化（`make_random_weights`），CPU 上也可跑。
- V100 实测（V=151936, D=256, L=4, 400 块 × seq 256，120 步）：
  probe **12.0112 → 7.4026（−38.4%）** PASS。
- `dpo`/`grpo` 仍明确抛 `NotImplementedError`（不假装支持）。

### 修复 · 梯度检查点**真正接进训练路径**
- 之前 `T.checkpoint` 只有实现与单测（梯度逐位一致、中间张量 −96.1%），
  但 `Qwen25.forward` 里**没有调用** —— 即"有实现 ≠ 训练省显存"。
- 现在逐层前向抽成 `_layer()`，用 `cfg["GRAD_CKPT"]=True`（或 `model.use_ckpt`）包一层
  `T.checkpoint`；新增 3 项测试断言**前向与梯度逐位一致**（max|diff| = 0.0）。

## [0.4.0] - 2026-11-01

### 新增 · 十行代码训练 API（`tgai_go.api.train`）
- **进程内**训练：模型与 SFT 循环搬进 `tgai_go/train/qwen.py`（与示例共用同一份代码），
  `pip install` 之后不再需要仓库目录。
- `api.train(weights=…, data=…, method="sft", lr=…, batch=…, steps=…, out=…)`
  返回 `TrainingResult`（`probe_base/probe_final/drop_pct/verdict/ckpt/log`）。
- `method` 只支持 `sft`；`pretrain/dpo/grpo` **明确抛错并说明计划**，不假装支持。
- `tiny=True` 走 CPU 迷你权重，几秒跑通（CI 与首次上手用）。
- 实测（V100，Qwen2.5-0.5B × alpaca-zh，120 步）：`probe 2.5825 → 2.3771（−8.0%）`，
  0.5 s/step；同一配置走 CLI 为 −10.6%（**留出集划分不同，数字不可直接相减**，见文档 §11.7）。

### 新增 · `.TG` 加载体检与护栏
- `tools/tg_audit.py`：同一模型用 `.TG` 与 safetensors 各加载一遍比 CE，
  差 > 0.5 判 FAIL 并给出三个最可能原因。
- `load_tg` 加两道体检：线性层布局（`[in,out]` 自动转置或报错）、
  归一化权重是否像 gamma（**单点离谱即报错**并给修法，留 `allow_suspicious_norms`）。
  实测某份 `.TG`：2-D 权重正常，但 6/49 个归一化张量读错（`final_norm` 均值 7.14、
  `ln1` 0.033）⇒ CE 8.30 vs safetensors 3.66。

### ⚠ 更正（2026-11-01 晚）：先前"`.TG` 读错"的判断是**错的**

原判断：某份 `.TG` 的 6/49 个归一化张量"读错"（`final_norm` 均值 7.14、`ln1` 0.033），
并据此加了"**单点离谱即抛错**"的护栏。

对拍官方 safetensors 后推翻：

| 检查 | 结果 |
| --- | --- |
| 容器自洽性 | 290 个 hash **全唯一**、**零重叠**、**零空洞**、不越界 ⇒ 读到的就是文件里存的 |
| 官方 Qwen2.5-0.5B 的 `blocks.0.ln1` 均值 | **0.032**（`.TG` 是 0.033）—— 官方本来就这样 |
| 官方 `final_norm` 均值 | 约 **7.1** —— 也一致 |
| 逐张量相关系数 | v_proj 0.939 / ln1 0.886~0.910（**同模型、被扰动**，不是读错） |

⇒ 真因：**我是拿"归一化权重应≈1"的常识当判据**，而实测模型不满足它。
护栏已改为**只记录不抛错**；客观判据换成两个工具：
- `tools/tg_check.py`：容器自洽性（hash 唯一/无重叠/不越界）—— 纯结构，不猜
- `tools/tg_audit.py`：同模型两路加载**比 CE** —— 对比，不猜阈值

仍未收口：`.TG` 与 safetensors 的相关系数只有 0.89~0.94（不是 1.0），
说明它是**同一模型的扰动副本**（疑似量化/合并/微调来源），这解释了 CE 8.30 vs 3.66，
但源头在哪个转换器仍待查。

### 新增 · 数据层（`tgai_go.train.data`）
- `read_jsonl` 流式读（坏行报行号+修法）、`normalize`（原生/`text`/ShareGPT/Alpaca 四形态）、
  `from_hf`（鸭子类型接 datasets，不 import datasets）、`train_eval_split`（确定性划分）。

### 新增 · 对齐 PyTorch 的能力
- 梯度检查点：梯度逐位一致，保留中间张量 **1041 → 41（−96.1%）**。
- `LAMB` 优化器（仅 CPU；CUDA 上明确报错）。
- `Embedding(weight=)`：直接挂已有参数，省掉一张白分配的随机词表。

### 修复 · 会静默出错的三个问题
1. `api` 解析日志时取「训练 loss（不可比）」而非「探针 loss（可比）」⇒ 报出错误数字。
2. `load_tg` 对布局/归一化异常**静默放行** ⇒ 能训但训坏（现已响亮报错）。
3. `sweep_report` 的 gnorm 阈值按实测重新校准（旧规则会误杀最优配置），并修正排序。

### 度量 · V100 显存构成（Qwen2.5-0.5B / b2 / seq256 / fp32）
权重 2.56 + 优化器 3.46 + 一步激活 **6.85** = 峰值 **16.5 GiB**；
⇒ 激活改 bf16 存储理论省 ≈3.4 GiB。**V100 = sm_70，硬件无 bf16**，加速比必须等 sm_80+。


### 新增 · 梯度检查点 + LAMB（对齐 PyTorch 的两项能力）
- `T.checkpoint(fn, *inputs)` / `T.checkpoint_sequential(fns, x)`：
  前向不建图、反向重算。实测梯度与不检查点**逐位一致**，保留的中间张量
  **1041 → 41（−96.1%）**。限制写清：单输出、fn 必须确定性。
- `T.optim.LAMB`：逐层 trust ratio（`‖p‖/‖u‖`），大 batch 训练用。
  与测试内独立参考实现 5 步一致；满足 `‖Δp‖ = lr·‖p‖` 不变量。**目前仅 CPU**
  （CUDA 上明确报错而不是静默退化）。

### 新增 · RoPE 缩放（Llama-3 / Mistral-v0.3 类模型）
- 读 `config.json` 的 `rope_scaling`，支持 `linear` 与 `llama3`（分段缩放）；
  **yarn / longrope / dynamic / mrope 明确报错并给修法**，不再静默忽略。
- 与官方 `transformers.LlamaForCausalLM` 完整 logits 对拍：llama3 / linear 均 **rel < 1e-5**。
- 为什么重要：忽略 scaling 时短上下文"看着差不多"、长上下文**静默算错**且不报错。

### 工程 · 错误信息质量（算子 / 形状 / 修法）
- CPU 路径此前直接抛 numpy 裸异常，用户看到的是
  `operands could not be broadcast together with shapes (3,4) (7,)`、
  `index 9 is out of bounds for axis 0 with size 5` —— 看不出是哪个算子、怎么改。
  现在 `RMSNorm` / `Embedding` / `functional.rms_norm` 都在最前面做形状校验，
  消息统一为「算子: 形状不匹配 —— 输入最后一维 X，期望 Y（x.shape=…）。修法：…」。
- `cross_entropy` 现在**接受 Tensor 标签**（以前会报 `int() argument must be … not 'Tensor'`）。
- 新增 `tests/test_error_messages.py`：7 个故障场景断言消息里必须同时有
  **算子名**、**具体数字**、**修法**，且不得是 numpy 裸异常。9 项用例。

### 训练 · 真正可续训（存档带优化器状态）
- `save_params(model, path, opt=None, step=0)`：除参数外还存 `__step` / `__opt_t` /
  `__opt_m_*` / `__opt_v_*`。此前只存参数，AdamW 的动量一丢，**续训等于重启动量**，
  轨迹不连续（表现为续训后 loss 先跳一下）。
- `load_params(model, path, opt=None)`：恢复参数 + 动量 + 步号；**老存档（只有参数）仍可加载**。
- `sft(..., start_step=0, opt=None)`：取样改为**按 seed 预生成计划**（第 s 步固定用
  `plan[s*batch:(s+1)*batch]`），warmup 用真实步号 ⇒ 续训与连续跑对齐。
- `examples/qwen25_sft.py --resume <npz>`：一条命令接着训。
- 测试 `tests/test_resume.py`：**"3 步 + 存档 + 续 3 步" 的 loss 与 "一口气 6 步" 的末 3 步一致**。

### 工程 · 行尾与 BOM 规则自动化
- 全仓文本文件统一为 **UTF-8 无 BOM + LF**：`python/ tools/ tests/ docs/ .github/` 共 259 个文件
  （其中包树里 16 个 `.py`/`.c` 之前带着 CRLF 被打进了 wheel）。
- `tools/check_package.py` 新增**wheel 内** BOM/CRLF 检查 —— 检查的是"用户实际拿到的东西"。
- 新增 `tests/test_hygiene.py`：包树与 docs/tools/tests 出现 BOM/CRLF 会让 CI 变红。
- 已知欠账（如实记录）：仓库根目录的历史脚本（`tgai*.py` 等）仍是 CRLF，
  一次性全改会把这 259 个文件的 diff 混进来，留作单独提交。

## [0.3.0] - 2026-10-31

### 修复 · CUDA 地基（重要，影响所有用户）
- **ptx 加载失败 `rc=222` 的根因**：pip 的 NVRTC（12.8）产出的 PTX ISA **8.7** 高于驱动
  （CUDA 12.4）支持的 **8.4**，任何**新模块**都会被驱动 JIT 拒绝。
  此前基线"能跑"只是因为驱动 JIT 缓存里有旧 SASS —— **清缓存后引擎完全不可用**。
- **修复**：改为**优先加载 cubin**（`--gpu-architecture=sm_86` + `nvrtcGetCUBIN`，SASS 不过 JIT），
  PTX 作为兜底；两条都失败时报错给出修法（钉 `nvidia-cuda-nvrtc-cu12<12.5` 或升级驱动），
  并注明"降 arch 到 compute_75 修不了"（已实测）。
- `doctor` 现在打印**模块形态（cubin/ptx）+ NVRTC/驱动版本**，不匹配时主动警告。

### 新增 · 检查点（GGUF 直接拿去训练）
- `tgai_go.train.checkpoints`：`load_gguf` / `load_safetensors_dir` / `save_npz` / `load_npz`
  → 统一的 `Checkpoint(weights, config)`。
  - 未知架构 / 量化类型 / 缺必需权重一律**明确报错并给修法**（静默丢权重比崩溃更危险）；
  - 读 `config.json` 取 `head_dim` / `tie_word_embeddings` / `rope_theta` / `rms_norm_eps`；
  - `tie=true` 时**丢弃重复的 `lm_head`**（官方 Qwen3 safetensors 带它；Qwen2.5 GGUF 的
    `output.weight` 与词嵌入逐位相同），并在数据不一致时报错。

### 新增 · Qwen3 微调支持
- `Qwen25` 增加 `DH` / `QK_NORM` / `QKV_BIAS` / `TIE` 四个开关（默认值保持 Qwen2.5 行为）：
  - **`head_dim` 解耦**：Qwen3-0.6B 是 128 而 `hidden/heads`=64，硬编码必错；
  - **`q_norm`/`k_norm`**：每个 head 对 `head_dim` 做 RMSNorm，且必须在 **RoPE 之前**；
  - **无 qkv bias**；`params()` 按 `id()` 去重（tie 参数曾被优化器更新两次）。
- `tools/qwen3_gpu_parity.py`：与 transformers 现场对拍（**先关 TF32**，失败时用 fp64 归因，
  并做 `q_norm` 梯度冒烟）。

### 新增 · 性能
- **并行平方和内核**（两段归约）：1.36 亿元素 **2.665 ms**，原单线程 `sumsq_sequential`
  需 **7595 ms（2850×）**；且单线程版在 float32 下累积误差达 **63%**，并行版 5e-9。
  梯度裁剪因此恢复可用（`--clip` 默认 1.0）。

### 工程
- **CI 安装 CPU 版 torch**：此前 `ci.yml` 只装 numpy/pytest，导致 **117 项里 71 项
  （所有 torch 对拍）在 CI 里根本没被收集** —— 数值对齐等于没有门禁。
- 新增 `docs/FINETUNE_RESULTS.md`：微调实测的**唯一事实来源**（R1~R8 全部实验的命令、
  超参、probe 起止、判定、日志原文、8 个坑）。
- 测试 129 项（新增 checkpoints 18 项 / Qwen3 6 项 / SFT 冒烟 3 项）。

## [0.2.0] - 2026-10-30

这一版的主题是 **「能替别人的模型训练」**：算子面按 PyTorch 对齐、可插拔扩展、真机结构验证。

### 新增 · 算子
- `nn_ops` 33 个算子（与 `torch` 同名同义，全部有数值对照测试）：
  - 归一化/激活：`rms_norm` `layer_norm` `gelu(tanh)` `silu` `sigmoid` `tanh` `rsqrt` `pow`
  - 线性/形状：`linear` `embedding`（**带反向**）`stack` `split` `chunk` `triu` `tril` `arange`
  - 归约：`argmax` `amax` `amin` `norm` `sum/mean`
  - 损失：`cross_entropy` `nll_loss` `mse_loss` `log_softmax`
  - 优化：`clip_grad_norm_` `sgd_step`
  - 采样：`topk_indices` `multinomial` `randn` `manual_seed`
  - 张量收缩：`einsum`（1~2 操作数常见形式）
- `contrib` 官方插件集 20 个算子（`erf` `gelu_exact` `elu` `selu` `leaky_relu` `mish`
  `hardswish` `hardsigmoid` `log1p` `lerp` `sign` `floor` `ceil` `trunc`
  `flip` `roll` `flatten` `unsqueeze` `squeeze`）

### 新增 · 架构
- **算子插件系统**（`ops_plugin`）：组合式与自带反向两种写法、三种发现方式
  （手工 / `TG_OPS_PATH` / setuptools 入口点）、重名与覆盖核心均有保护。
- **后端三层接口**：core 54 / fallback 8 / optional 17，`backend.report()` 给分层体检。
- **NumPy 参考后端**（`backends/numpy_ref.py`）：接口活文档 + CI 无 GPU 可测 + Vulkan/ROCm 移植模板。
- `tools/backend_port.py`：一条命令看新后端还差哪些核心方法。

### 验证（可复现）
- **真实开源权重对齐**：Qwen2.5-0.5B（GQA 14:2 + qkv bias + RoPE θ=1e6）前向 vs HF
  `max|Δ|=1.278e-04`，相对误差 **6.29e-06**（判据 ≤1e-5）。
- **LLaMA 结构训练对齐**：RMSNorm+RoPE+SwiGLU+GQA、20 步，与 PyTorch 孪生
  逐步 loss 相对偏差 **1.59e-05**、训练后参数相对偏差 **4.90e-05**、loss 下降 99.1%。
- **算子覆盖率**：以 PyTorch 公开面为基准，Tier-1 训练必需 **63 项中 62 项具备（98%）**。

### 工程
- CI（GitHub Actions，CPU-only）：`pytest` + `build` + `check_package`，**已转绿**。
- `tests/conftest.py` 统一 sys.path，测试不再依赖"本地是否装过"。
- `tools/check_package.py`：发布前自检（文件齐 / 语法净 / 干净 venv 可 import / `.cu` 随包）。
- `docs/` 新增：`ENGINEERING_STANDARD.md` `PLUGINS.md` `BACKEND_PORTING.md`
  `OP_COVERAGE.md` `GGUF_TO_TG.md`。

### 修复
- `gguf_to_tg`：token 的 UTF-8 被 `[:255]` 硬截断切坏多字节字符 → 读 `.TG` 时崩溃。
- `gguf_to_tg`：`output.weight` 与 embedding 权重绑定，说明文档化（有意跳过）。
- `ops_plugin`：`kwargs` 未传给反向 → `elu(alpha=)`/`softplus(beta=)`/`lerp` 梯度静默算错。
- 形状算子：额外位置参数（`dims`）未进反向 → `flip`/`roll`/`flatten` 反向 `TypeError`。
- `squeeze`：对齐 torch 语义（非 1 维原样返回，而不是 numpy 那样报错）。
- 全仓库清理 16 个带 UTF-8 BOM 的源文件。

## [0.1.5] - 2026-10-30
- 新增 `tools/check_package.py`（发布前自检）；清理 BOM。

## [0.1.4] - 2026-10-30
- 补齐 `tgai_go` 顶层模块（`pack` `presets` `read_tg`）—— 修复 `--save-resume` / 导出时报
  `ModuleNotFoundError: tgai_go.pack`。

## [0.1.3] - 2026-10-30
- **开箱即用**：自动发现本机任何 Python 环境里的 `nvidia/*/lib`；`TGAI_CUDA_BIN` 支持多路径。
  实测：无 torch 的全新 venv 里 `pip install tgai-orv` → `doctor` 12/12 PASS。

## [0.1.1] - 2026-10-30
- 新增 `[cuda]` 可选依赖（`nvidia-cublas/nvrtc/runtime-cu12`）；`libcublas` 缺失的报错补上修法。

## [0.1.0] - 2026-10-30
- 首个公开版本：推理引擎 + 训练引擎（`tgai_go.train`）、CUDA 算子源码、示例、验收工具、
  `scripts/quickstart.*`、`docs/PRODUCTION_READINESS.md`。
