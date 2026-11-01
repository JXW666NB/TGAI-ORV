# 自研训练引擎 bf16 混合精度（AMP）落地方案

> 分析对象：`D:\DSH-Workspace\TGAI-对话档案\_tgai_orv`（只读分析，未修改任何仓库文件）
> 结论基于**读源码 + 本机只读探针实测**（`tools/amp_readiness.py` 跑通；另用一个临时脚本
> 打印了 `_device_source()` 的真实组成）。凡属估算的地方都标了【估算】，与实测分开。
> 目标机：**V100 = sm_70，硬件无 bf16**；cuBLAS `CUDA_R_16BF` 需 sm_80+。

---

## 0. 一句话结论

**在这台机器上能做的只有一件事：让"激活"以 bf16 驻留（2 字节），计算仍全程 fp32。
它省显存、不省算力；三个已有 bf16 内核算"存储半边"，不算 AMP，而且它们至今没有接进引擎。
GradScaler 我们不需要（bf16 指数位与 fp32 相同、梯度全程 fp32）。
"bf16 加速"这句话在 sm_70 上不可能成立，只能写"bf16 存储节省显存"。**

---

## 1. 先核对证据（防止把"编译通过"读成"能用"）

| 结论 | 需要的证据等级 | 仓库现有的 | 我的核对结果 |
| --- | --- | --- | --- |
| bf16 转换与 torch 位级一致 | CPU 对拍 | ✅ `tests/test_amp_bf16.py:18-28` | ⚠️ 实际只跑 **2×10⁴** 个值、判据是"≥99.9% 位相同"（`:21`、`:26`）；`docs/AMP_PLAN.md:83` 写的是"20 万个值 100.0000%"。**文档比测试强**，要么补测试，要么改文档数字 |
| 往返误差 ≤ 2⁻⁸ | 单元测试 | ✅ `tests/test_amp_bf16.py:31-37`（断言 `<= 2.0**-8`） | ✅ 与 `docs/AMP_PLAN.md:96-99` 的判据纠正一致。注意该文件**开头 docstring 第 9 行仍写 2⁻⁹**，与正文自相矛盾，建议一并改掉 |
| 3 个 bf16 内核可编译 | NVRTC rc=0 | ✅ `docs/AMP_PLAN.md:85`（PTX 489,270 B） | ⚠️ `tools/amp_compile_check.py:107` 默认 `--arch compute_86`，**这次编译证据是 sm_86，不是 sm_70**。位运算内核没有架构特性，`compute_70` 预期也能过，但那是预期不是证据 —— 建议补一条 `--arch compute_70` 的记录 |
| 3 个 bf16 内核**已接入引擎** | 出现在 `_device_source()` | ❌ 文档没这么说，但很容易被这么读 | ❌ **实测未接入**：`cuda.py` 全文无 `BF16_KERNELS`、无 `import amp`；`_device_source()`（`:1809-1812`）只拼 9 段，**不含** `amp.BF16_KERNELS`。实测 `_device_source()` 61,126 字符、**81 个 `__global__`**，`ewise_add_bf16`/`ewise_scale_bf16`/`rmsnorm_bf16` 三个都不在其中 |
| 引擎内已有 bf16 的能力 | — | — | ✅ 有两样：① `cast_f32_to_bf16` / `cast_bf16_to_f32`（`cuda.py:1557-1571`，属 `_FP16_KERNELS`，**已在** device source 里）；② cuBLAS bf16 GEMM（`:4473-4499`） |
| GPU 上执行 / 省显存 / 加速 | GPU A/B | ❌ 三项全无 | 与 `docs/AMP_PLAN.md:78-86`、`:140-152` 自述一致 |

**这张表的用法**：以后任何"bf16 已支持"的说法，先问它落在哪一行证据上。

---

## 2. Q1：`Tensor` 有没有 dtype 概念？能不能只做"激活 bf16 存储"？

### 2.1 现状：有三层 dtype，但**没有一层能表达 bf16**

| 层 | 位置 | 真实情况 |
| --- | --- | --- |
| ① 对外 API 的 `DType` | `__init__.py:349-375` | 有 `float32/float64/int32/int64`（`:370-375`），**没有 float16 / bfloat16**；`__all__` 也只导出这四个（`:270`） |
| ② `Tensor.dtype` | `__init__.py:635-639` | **CUDA 张量恒定报 `float32`**：`if self.data is None: return float32`（`:637-638`）。而 CUDA 张量的 `data` **永远是 None**（`:598`、`:606`）。⇒ 这是"名义 dtype"，与实际缓冲无关 |
| ③ 缓冲的 `GpuBuf.dtype` | `cuda.py:1887`（`__slots__` 含 `dtype`）、`:1898`（`np.dtype(dtype)`） | **真实**，而且 `empty/h2d/d2h/clone/full` 全都按 `itemsize` 算字节（`:3033`、`:3059`、`:3085`、`:3097`、`:3102`）。`rt.empty((n,), dtype=np.uint16)` 今天就能拿到一块 2 字节/元素的缓冲 —— `_cast_flat_ptr_bf16` 正是这么干的（`:4458`） |

`Tensor` 本身没有 dtype 槽位（`__slots__`，`:570-571`），CUDA 张量的存储一律在 `_gpu` 里；
而 `Tensor.__init__` 的 CUDA 分支写死了 `np.float32`（`:597-599`），`Parameter` 又必经它
（`:2828-2833`）⇒ **今天没有任何路径能造出一个 bf16 的 Parameter**（这是好事：master weight
天然 fp32，与 `docs/AMP_PLAN.md:55` 一致）。

另外，整个 6063 行运行时里**只有 2 处**校验 dtype（`cuda.py:3626`、`:3655`，均属 cat 拷贝）。
其余 79 个接口、81 个内核**拿到什么缓冲就用什么**。这条事实同时是"方便"和"危险"，见 2.4。

### 2.2 不引入 dtype，能不能只做"存 bf16、算 fp32"？

**能，但"不引入 dtype"只能理解成"不改 `Tensor` 的公共 dtype 语义"，不能理解成"存储层不需要类型标签"。**
理由：81 个内核的形参是 `float*`，编译器/驱动不会替你分辩；一块 2 字节缓冲喂给它们，内核会按
4 字节步长读到缓冲区之外（读 2 倍字节）并写坏邻居。所以**存储层必须有一个"这是 2 字节"的标签**，
而 `GpuBuf.dtype` 已经现成。最小方案 = 复用它，**不动 `Tensor.__slots__`**。

### 2.3 最少要动哪几处（按"必须/可选"分组）

| # | 文件 | 函数:行 | 改什么 | 为什么必须 |
| --- | --- | --- | --- | --- |
| M1 | `cuda.py` | 模块顶部 `:16-23` | `from . import amp`（`amp.py` 只依赖 numpy，无环） | 才能引用内核源码 |
| M2 | `cuda.py` | `_device_source()` `:1809-1812` | 加 `+ "\n" + amp.BF16_KERNELS` | **这是"接入"的唯一定义**。不加，三个内核永远编译不进引擎 |
| M3 | `cuda.py` | `launch()` `:3160` 或 `_ptr()` `:3287` | 加 fail-closed 闸门：非 float32 缓冲只允许进 `*_bf16` 内核，否则 `raise CudaError` | 见 2.4，这是安全前提 |
| M4 | `cuda.py` | 新增 3 个运行时方法（写在 `_sgemm_bf16` `:4473` 附近） | `add_bf16/scale_bf16/rmsnorm_fwd_bf16`：`self.empty(shape, dtype=np.uint16)` + `launch(...)` | 把内核变成可调用的算子 |
| M5 | `__init__.py` | `Tensor.dtype` `:635-639` | `cuda` 分支改成 `return _wrap_dtype(self._gpu.dtype)` | 否则 bf16 张量自称 float32，任何按 dtype 分支的代码都会错 |
| M6 | `__init__.py` | `_DTYPES` `:375`、`__all__` `:270` | 加 `bfloat16 = DType("bfloat16", np.uint16, True)` | `np.uint16` 目前会被 `_as_dtype` 判成**非浮点**（`:394`），`_is_float_dtype` 返回 False ⇒ 会被当成整数张量 |
| M7 | `__init__.py` | 边界 API（新增，放 `:1789` `tensor()` 附近） | `to_bf16_store(t)` / `t.bf16_store()`：显式"存成 bf16" | 引擎**没有 dispatch 层**，只能显式标注；见 2.5 |
| M8 | `__init__.py` | 要改驻留格式的那几个 `_forward_cuda` 闭包 | 例：`RMSNorm._forward_cuda` `:3214-3236`（闭包 `:3228-3235` 抓 `x._gpu`） | 真正省显存的地方在**闭包抓什么**：反向要用的张量，抓 bf16 影子而不是 fp32 |
| M9 | `backend.py` | `GROUPS` `:42-92`、`missing()` `:153-155` | 新增一组 "AMP 存储（可选）"，把 M4 的名字登记进去 | 不登记，`missing()` 会把新方法算进"必做"，其它后端接入时报缺 |
| M10 | `tools/amp_readiness.py` | `:43` | bf16 判定加 `_bf16` 后缀 / `unsigned short` | **现在这个工具永远扫不到位运算写法的 bf16 内核**：它只认 `"bfloat16"`/`"__nv_bfloat16"` 字样。⇒ `docs/AMP_PLAN.md:41` 那条判据"`amp_readiness.py` 里 bf16 内核数 > 0" **用当前实现无法达成**（我实测它现在报 81/0），必须先修工具 |

> 可选（S2 才需要）：`_sgemm_bf16` `:4473-4499` 里把 C 也改成 `CUDA_R_16BF`，省一次回写 cast；
> 这依赖 sm_80+ 的 cuBLAS 能力，V100 上无法验证。

### 2.4 一个必须先做、否则一定出事的闸门

现状：`add()` 无条件给输出分配 float32（`:3307`），`_vec4_ok()` 只查 16 字节对齐、**不查 dtype**
（`:3253-3265`），`_ptr()` 什么都不查（`:3287-3289`）。也就是说
**一块 uint16 缓冲今天可以被直接送进 `ewise_add_v4`** —— 内核按 `float4` 读，读的字节数是缓冲的 2 倍。

所以顺序必须是：**先加 M3 闸门，再接任何 bf16 内核**。闸门要做成"默认拒绝"（fail-closed），
而不是"记一条警告继续跑"——后者会让数值错误以"loss 有点怪"的形式出现，极难定位。

### 2.5 为什么必须"显式边界"，不能学 autocast 做隐式策略

`__init__.py` 里几百处 `rt.xxx(...)` 直接绑在具体运行时上（`backend.py:6-17` 自述），
没有 ATen 那种 dispatch key 层。要做隐式策略，就得在 79 个接口每个出入口都插判断；
而显式边界（M7）只需要在模型代码里标 4~6 处（`qwen.py:185-223` 的 `_layer`）。
**S1 选显式，代价是模型代码要知道 bf16；收益是每一步都能被单测钉住，且不会悄悄改到别的模型。**

---

## 3. Q2：三个已有 bf16 内核算不算"真 AMP"？

### 3.1 事实层面：它们现在**一个都没跑过**

- 不在 `_device_source()` 里（实测 `:1809-1812` 只拼 9 段）；
- `cuda.py` 从不 import `amp`（实测正则无命中）；
- 全仓引用只有 3 处：`tests/test_amp_bf16.py:76-81`（字符串检查）、
  `tools/amp_compile_check.py:117`（拼进去只做编译）、以及 `amp.py` 自己。
- ⇒ 它们目前是**"编译验证过的示例代码"**，对训练零影响。这一点写文档时必须说清楚，
  否则"bf16 内核已通过编译"极易被读成"bf16 已经能用"。

### 3.2 就算接进去，也不是"真 AMP"

"真 AMP"（`torch.autocast` 语义）是**算**用低精度：GEMM 走 Tensor Core、
中间激活 bf16、master weight 与敏感算子 fp32。
`amp.py:53-54` 自己写的约定是"**读 bf16、用 float 算、写回 bf16**"，
`tests/test_amp_bf16.py:80-81` 还把这条钉成了断言。
⇒ 这三个内核的算力与 fp32 版本**完全一样**，改的只是"字节数"。

准确的说法是：**它们是 AMP 的存储半边（S1），不是 AMP 的计算半边（S2）。**

### 3.3 收益与代价

**收益（三条，都只跟字节有关）**

1. **显存**：这些内核的输入/输出驻留字节减半。只对"活过内核边界"的张量有效。
2. **带宽**：读写字节减半。`rmsnorm_fwd`、`ewise_*` 这类**带宽受限**内核对 V100 也可能真变快
   —— 但这是"字节少了"，**不是"bf16 算得快"**（V100 的 fp32 单元一位没多）。这条必须单独 A/B 且标注。
3. **L2/缓存命中**：工作集减半，对大 batch 的 ewise 链有次级收益（不可直接归因，别写进结论）。

**代价**

1. **数值**：每次"存"都过一次 RNE，最坏相对误差 **2⁻⁸ = 3.9×10⁻³**（`tests/test_amp_bf16.py:31-37`，
   实测往返 3.85×10⁻³）。这不是"每步 0.4% 的噪声"那么温和——它会在
   ①**残差流**里逐层累积（24 层随机游走 ⇒ 量级可达 10⁻² 级【估算，最坏界很松，必须实测】）；
   ②**长 K 点积**里放大（K=896/4864，输入已被舍入 ⇒ 结果相对误差约 2u~4u）。
   ⇒ 现有判据 `docs/AMP_PLAN.md:39`「100 步 loss 相对差 ≤ 1e-3」**对 8 位尾数过于乐观**，见 5.2。
2. **kernel 次数**：
   - 若把 bf16 读写**融进**现有内核（`BF16_KERNELS` 的做法）：**次数不变**（1:1 替换）。
   - 若在算子边界**插 cast**：每个边界 **+2 次**（进、出）。好消息是 cast 内核**已经存在**
     （`cuda.py:1557-1571`），不用新写。
   - 到底多少次？`tools/profile_step.py:111-146` 已经能统计"次/步 + host ms/步"，S0 就是把这个基线记下来。
3. **复杂度**：每多一个 bf16 变体就多一条要维护的数值路径；`docs/AMP_PLAN.md:61-69` 的风险表已经列了 5 条，
   我建议再加一条"**池碎片会吃掉收益**"（见 5.2 的反判据）。

### 3.4 显存能省多少（我算的一笔账，供 S0 探针证伪）

Qwen2.5-0.5B 配置取自 `qwen.py:28-29`：`V=151936, D=896, L=24, NH=14, KV=2, FF=4864, DH=64`。
取 B=4 / S=512（N=2048 tokens，即 `docs/AMP_PLAN.md:182-187` 那个 17.72 GiB 的形状）：

| 每层张量 | 元素数 | fp32 | bf16 |
| --- | --- | --- | --- |
| `h`/`h2`（rmsnorm 输出）[N,D] | 1,835,008 | 7.00 MiB | 3.50 MiB |
| `q`/`k`/`v`/`o`（各一）[B·NH,S,DH] | 1,835,008 | 7.00 MiB | 3.50 MiB |
| `g`/`u`/`act`（SwiGLU）[N,FF] | 9,961,472 | 38.00 MiB | 19.00 MiB |
| `P`（物化 SDPA 概率）[B·NH,S,S] | 14,680,064 | 56.00 MiB | 28.00 MiB |
| **每层小计**（P 物化） | — | **212 MiB** | **106 MiB** |
| **×24 层** | — | **≈4.97 GiB**（5088 MiB） | **≈2.48 GiB**（2544 MiB） |

⇒ 语义激活集合本身约 5.0 GiB，全 bf16 化省 **≈2.48 GiB**。【估算】

但注意两件事，它们决定 S1 该先改哪里：

1. `docs/AMP_PLAN.md:117`、`:121` 实测"跑一步 +12.87 GiB、其中激活 ≈6.85 GiB"（B=2/S=256）。
   按上表同样口径，B=2/S=256 的语义激活只有 **≈1.1 GiB**（每层 46 MiB × 24）【估算】。
   ⇒ 语义激活只占实测 6.85 GiB 的 **约 16%**，即**约 5/6 是表达式级临时量**。这与代码对得上：
   `permute()` 是**真拷贝**（`cuda.py:3508` 分配新缓冲），`_rope_apply` 一次就造
   `cat`+两次乘+一次加共 4 个满尺寸临时量（`qwen.py:178-183`），
   `repeat_interleave` 又把 k/v 各放大 7 倍（`qwen.py:212-213`）。
2. 因此**S1 的优先级不该是"先把 q/k/v 变小"，而是"先把临时量最多的那几个算子（rope/cat/permute/silu_mul）的输入输出变小"**。
   这条是【推论】，需要用 5.1 的 S0 探针证实/证伪；如果探针显示语义张量才是大头，就按 5.2 的清单改顺序。

---

## 4. Q3：与 PyTorch 对比

### 4.1 `torch.autocast` 的机制（依据官方文档，见文末链接）

1. **它是一个 dispatch 层的策略，不是一种张量类型。** 进入 `autocast` 区域后，每个算子被查表决定
   用 `lower_precision_fp`（fp16/bf16）还是 fp32；**算子自己不知道自己被降精度了**。
2. **三张表**（CUDA）：
   - 走低精度：`__matmul__`、`addbmm`、`addmm`、`addmv`、`addr`、`baddbmm`、`bmm`、
     `chain_matmul`、`multi_dot`、`conv1d/2d/3d`、`conv_transpose*`、`GRUCell`、`linear`、
     `LSTMCell`、`matmul`、`mm`、`mv`、`prelu`、`RNNCell` —— **全是"大 GEMM/卷积"**。
   - 保持 fp32：`cross_entropy`、`nll_loss`、`log_softmax`、`softmax`、`softmin`、`layer_norm`、
     `group_norm`、`norm`、`normalize`、`rsqrt`、`sum`、`prod`、`cumsum`、`exp`、`log`、`pow`、
     `log1p`、`reciprocal`、各类 loss 与 `cdist`/`dist` 等 —— **全是"归约/超越函数/损失"**。
   - 按"最宽输入"提升：`addcdiv`、`addcmul`、`atan2`、`bilinear`、`cross`、`dot`、`grid_sample`、
     `index_put`、`scatter_add`、`tensordot`；而 `add` 这类二元算子**原生按最宽提升**
     ⇒ 只要有一个 fp32 输入，`add` 就在 fp32 里算并产出 fp32。
3. **只对 out-of-place 生效**：`a.addmm(b,c)` 会被自动降精度，`a.addmm_(b,c)` 或 `addmm(..., out=d)`
   **不会**。带显式 `dtype=` 的也不会。
4. **权重 cast 有缓存**（`cache_enabled`），不是每次前向都转一遍。
5. **反向跑在与对应前向相同的精度里**（官方原话："Backward ops run in the same type that autocast
   used for corresponding forward ops"），并且**不建议**把 backward 包进 autocast 区域。

**对照我们**：上面 5 条我们一条都不具备（无 dispatch、无表、无缓存、反向是手写闭包）。
这解释了为什么"落地方案"必须是**显式边界 + 逐算子 bf16 变体**，而不是"加一个开关"。

### 4.2 为什么 loss / softmax / norm 必须留在 fp32（四条，按重要性）

1. **它们都是"归约"**：softmax 要 `exp` 后求和、CE 要沿 V=151936 求和、RMSNorm 要求平方和。
   bf16 只有 8 位有效位，**大动态范围下相加会直接吃掉小项**（大数吃小数），
   而 fp32 的 24 位尾数让"求和"这件事本身有意义的。
2. **`exp` 的输入跨度大**：logits 减去最大值后仍在 [-20, 0]，`exp` 结果跨 9 个数量级；
   在 bf16 里存/算这些中间值，相对误差还是 2⁻⁸，但**权重被压缩到接近 0 的项会被舍成 0**。
3. **norm 的 `1/sqrt` 是病态的**：`eps=1e-6` 直接加在"均方"上（`qwen.py:29` 的 `EPS=1e-6`）。
   当某一行的均方接近 eps 量级时，eps 在 `ms + eps` 里的相对贡献变大，
   尾数少 16 位会把这点相对差异放大进 `rsqrt` 的输出 —— `docs/AMP_PLAN.md:67` 已经点到这条。
4. **性价比**：这三类算子**不是算力瓶颈**（GEMM 才是），降精度几乎不提速，却要付精度代价。
   ⇒ **"为什么不"比"为什么"更准确：没有任何收益，只有风险。**

### 4.3 `GradScaler` 何时必要？——只在 fp16 下必要，我们**不需要**

官方文档的机制（"Gradient Scaling" 一节）：fp16 的**指数位只有 5 位**（最小正规数 6.1×10⁻⁵、
最小次正规 6×10⁻⁸），小梯度会**下溢成 0**，对应参数的更新就丢了；于是把 loss 乘一个大系数，
反向后梯度同比放大、避开下溢，优化器 step 前再除回去。

**我们不需要 GradScaler，三条理由（任何一条单独成立）：**

1. **我们的 bf16 只用于"存"，梯度全程 fp32。** 前向是 fp32 计算、反向也是 fp32 计算，
   落到 `.grad` 的缓冲是 float32（`Parameter` 恒 fp32，`__init__.py:2828-2833`）。
   不存在"小梯度变成 bf16 后被 flush 成 0"的通道。
2. **bf16 的指数位与 fp32 完全相同（都是 8 位）**，动态范围 ~1.2×10⁻³⁸ ~ 3.4×10³⁸。
   bf16 丢的是**尾数**（8 位有效位），不是**范围**。缩放只能解决"下溢到 0"，
   **不能**解决"尾数不够"——这是 scaler 原理性的能力边界。
3. 官方文档自己就把两者解耦了：`torch.autocast` 与 `GradScaler` 是**模块化**的，
   并明确写"CPU 上用 bf16 的 AMP **只用** autocast"（不需要 scaler）。

**什么时候才需要**：只有当"梯度本身以 fp16 存储/累加"时才需要。
我们即便做到 S2（bf16 GEMM + fp32 累加，`cuda.py:4499` 的 `CUBLAS_COMPUTE_32F`），
梯度仍然是 fp32 ⇒ **S2 也不需要**。

> 顺带一条容易搞反的：`GradScaler` 不是"更安全的选择"，加上它反而引入一个需要调的超参
> （`init_scale`/`growth`）和一个"inf 就跳过 step"的分支。**在没有下溢风险时加它，是纯负债。**

### 4.4 我们的"bf16 存储"与 torch bf16 autocast 的差异矩阵

| 维度 | `torch.autocast(bf16)` | 我们的 S1（bf16 存储） |
| --- | --- | --- |
| GEMM 算在哪 | bf16 Tensor Core（sm_80+） | **fp32**（V100 上只能这样） |
| 激活驻留 | bf16 | bf16（目标） |
| master weight | fp32 | fp32（已如此） |
| softmax/CE/norm 内部 | fp32 | fp32 |
| 残差 `add` | 按最宽提升（多半 fp32 输出，但上游已是 bf16） | **显式 fp32 计算 + 可选 bf16 存** |
| 反向精度 | 与前向相同（bf16 GEMM ⇒ bf16 梯度路径） | **全程 fp32** |
| 需要 GradScaler | 不需要（fp16 才需要） | 不需要 |
| 省显存 | 是 | 是（幅度更小，因为不省 GEMM 计算） |
| 提速 | 是（Tensor Core） | **否**（带宽受限内核可能小幅受益，须单独标注） |

---

## 5. Q4：分阶段方案

前置：**先修 3 件事，否则后面的判据不可信**（都是小改，见 §7）。

### 5.1 S0 —— 不改行为，只加开关与统计

**目标**：把"要改哪里"从 41 个内核的清单，变成"按字节排序的 7~10 个分配点"。

| 项 | 内容 |
| --- | --- |
| 改哪里 | ① `amp.py:109-113` 扩成三态：`amp_mode() -> "off"/"stats"/"store"`，**保持 `amp_requested()` 语义不变**（`tests/test_amp_bf16.py:64-71` 钉着它）；② **新增 `tools/amp_storage_probe.py`**：照 `tools/profile_step.py:113-123` 的做法 monkeypatch `rt.empty`/`rt.launch`，按调用点（`sys._getframe(1)` 的 `co_filename:lineno`）聚合 peak/live 字节，输出"top-N 分配点 + 若转 bf16 可省字节"；③ 可选：修 `Tensor.dtype`（M5/M6），因为它不改任何计算 |
| 明确不做 | 不改 `_device_source()`（不接内核）、不改任何算子、不加 cast。⇒ **一步的 loss 必须与基线逐位相同** |
| 判据（数值） | `TG_AMP` 未设时：全量测试通过；同 seed 一步 loss **逐位相同**（`==`，不是 allclose） |
| 判据（显存） | 用 `tools/mem_profile.py`（`:28-66` 采样、`:106-116` 出 peak/`peak_delta`）+ `tools/profile_step.py:134-137` 的 `pool_stats()`（peak/live/cached/hit_rate/misses），把基线落成 JSON：**峰值 GiB、s/step、kernel 次/步、top-10 分配点字节** |
| 判据（速度） | 与 fp32 基线偏差 **< 1%**（证明 S0 零开销）；kernel 次/步与基线完全一致 |
| sm_70 可验 | **全部** |

### 5.2 S1 —— 激活 bf16 存储（存 bf16、算 fp32）

| 项 | 内容 |
| --- | --- |
| 改哪里 | ① M1~M4（接入 3 个内核 + 3 个运行时方法）；② **先 M3 闸门，再接内核**；③ M5/M6（让 dtype 说实话）；④ M7 显式边界 API；⑤ M8：按 S0 探针的榜单改对应闭包 —— **第一优先 SwiGLU 链 `F.silu(g) * u`（`qwen.py:221`）与 `rmsnorm`**（g/u/act 各 38 MiB，是本层字节最大的三项；`rmsnorm` 的 bf16 内核已写好），第二优先 `_rope_apply`/`cat`/`permute` 的临时量（`qwen.py:178-183`、`:203-205`），第三优先 `q/k/v` 这些"语义张量"；⑥ M9（backend 登记）、M10（修探针工具） |
| 图捕获注意 | `bf16 GEMM` 已经因为"捕获期间 cast 要新分配"而自我禁用（`cuda.py:4421-4428`）。**S1 必须照抄这个模式**：捕获期间走 fp32，并打印一次说明。否则 `TG_CUDA_GRAPH` 会直接图失败 |
| 判据（数值）—— 建议替换掉 `AMP_PLAN.md:39-40` 的旧判据 | ① **单步前向 loss 相对差 ≤ 1×10⁻³**（标量均值，误差大量抵消，这条容易过）；② **反向梯度相对 L2 差 ≤ 1×10⁻²**（★最要紧：bf16 存储的直接后果落在这里，比 loss 敏感一个量级）；③ **100 步 loss 轨迹**：末步相对差 ≤ **1%**（不是 10⁻³——8 位尾数撑不住 10⁻³）；④ **无 NaN/Inf**；⑤ 梯度范数在同一量级（0.5×~2×）；⑥ 与 fp32 基线**同 seed、同数据顺序**（否则①②③全部无意义） |
| 判据（显存） | 主判据（可归因）：S0 榜单里"已转换子集"的峰值字节下降 **≥ 40%**（理论 50%，扣 fp32 影子/对齐/池碎片）。次判据：`nvidia-smi` 设备峰值**不得上升**。**反判据**：若池 live 降了但设备峰值没降 ⇒ 收益被池缓存吃掉（`AMP_PLAN.md:193` 实测 `_pool_cached_bytes` 4.22 GiB 常驻），**必须先做池回收，否则 S1 白做** |
| 判据（速度） | 允许 ±10%。若慢 >10%：先看 kernel 次/步是否超预算（= S0 基线 + 新增边界数×2），再决定要不要融合 |
| sm_70 可验 | **全部**（bf16 存储不依赖 bf16 硬件；位运算内核在任何架构都能跑） |
| sm_80+ 才能验 | 无（S1 不含算力改动）。若想验证"带宽收益"，需在同一内核上做 fp32/bf16 存储 A/B 并**注明这是带宽不是算力** |

### 5.3 S2 —— GEMM 输入 bf16（需 sm_80+）

**现状（重要）**：`_sgemm_bf16`（`cuda.py:4473-4499`）已经在调 `cublasGemmEx(CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F)`，
门槛在 `_bf16_wanted`（`:4417-4429`）：`m*n ≥ 262144` 且 `k ≤ 1280`，由 `TG_BF16=1` 打开。
**但它没有架构检查** —— `_cc_arch()`（`:2454-2463`）存在却没用在这里；失败路径是
`_bf16_fail`（`:4406-4415`）捕异常 + **一行 stderr 警告**后回退 fp32。
⇒ **在 V100 上 `TG_BF16=1` 会静默降级**。这是"测量卫生"问题：任何人做完 A/B 都可能误以为"开了"。

| 项 | 内容 |
| --- | --- |
| 改哪里 | ① **前置**：`_bf16_wanted` 加显式能力闸门（`_cc_arch() >= "compute_80"`），启动时**明确打印**"bf16 GEMM: enabled (sm_86)" 或 "unavailable (sm_70, requires sm_80+)"；② 可选：`_sgemm_bf16` 里 C 也改 `CUDA_R_16BF`（省一次回写 cast，需 sm_80+ 实测确认）；③ 若权重也走 bf16，需单独评估（会改数值行为，`AMP_PLAN.md:123` 已标注） |
| 判据（数值） | 同 S1 的①②③；另加：**GEMM 级**单元对拍（`tools/qwen2_parity.py` / `qwen3_gpu_parity.py` 已有对拍框架）相对误差 ≤ 1×10⁻² |
| 判据（速度） | sm_80+ 上大形状 wall-clock **≥ 1.5×** 才算拿到 Tensor Core 收益；用 `tools/profile_step.py:148-198`（已有按 kernel 的 CUDA event 计时） |
| 判据（显存） | S2 本身**不省显存**（除非权重也 bf16）。不要把它算进显存预算 |
| sm_70 能验 | **只能验"回退是否诚实"**：能力闸门是否明确报"不可用"、dispatch 是否原样走 fp32、kernel 次/步是否与 fp32 完全一致。**不能验加速比** |
| sm_80+ 才能验 | 加速比、bf16 C 回写、真正的 AMP 收益、与 PyTorch AMP 的"正面对决" |

### 5.4 判据汇总（一页表）

| 阶段 | 数值判据 | 显存判据 | 速度判据 | sm_70 可验 |
| --- | --- | --- | --- | --- |
| S0 | 一步 loss **逐位相同** | 产出基线 JSON（不设目标） | 偏差 <1% | ✅ 全部 |
| S1 | loss ≤1e-3（单步）/ 梯度 L2 ≤1e-2 / 100 步 ≤1% / 无 NaN | 已转换子集 ≥40%，设备峰值不升 | ±10% | ✅ 全部 |
| S2 | 同上 + GEMM 对拍 ≤1e-2 | 不预期收益 | sm_80+ ≥1.5× | ⚠️ 只能验"诚实报不可用" |

---

## 6. Q5：在 sm_70 上不能声称"bf16 加速"——怎么写最不容易被误读

### 6.1 一句话模板（直接抄）

> **V100（sm_70）没有 bf16 硬件单元。本项目在此机器上只做"bf16 存储"：
> 激活以 2 字节驻留、计算全程 fp32，因此省显存、算力不变。**
> **任何吞吐/加速结论都必须在 sm_80+（A100/A10/4090）上实测，本机不提供也不外推。**

### 6.2 允许 / 禁止 对照表

| ❌ 会被误读（禁止） | ✅ 准确（照抄） |
| --- | --- |
| "支持 bf16" | "支持 **bf16 存储**（存 bf16、算 fp32）；**不支持** bf16 计算（需 sm_80+）" |
| "bf16 已就绪 / AMP 完成" | "AMP-S1（存储）完成；AMP-S2（计算）**未开始**，阻塞条件：sm_80+ 硬件" |
| "bf16 加速 1.8×" | "**未测**加速比；V100 上不存在 bf16 加速的物理基础" |
| "PTX 489,270 B 编译通过 ⇒ bf16 可用" | "3 个 bf16 内核在 **compute_86** 下 NVRTC 编译通过（**仅证明可编译**）；GPU 执行未验证" |
| "`TG_BF16=1` 就走上 bf16" | "`TG_BF16=1` 在 sm_80+ 上让 GEMM 用 bf16 输入；**sm_70 上会自动回退 fp32 并打一行警告**（`cuda.py:4406-4415`）" |
| "显存省一半" | "**已转换的那部分激活**字节省一半；总峰值省 X%（实测值待填）" |

### 6.3 三个"防误读"的结构性做法（比措辞更有效）

1. **两个开关，两张表，永不混称。**
   `TG_AMP` = **存储**（bf16 驻留）；`TG_BF16` = **计算**（GEMM 输入 bf16，需 sm_80+）。
   `amp.py:21-22`、`cuda.py:88` 已经是这么分的，`tests/test_amp_bf16.py:64-71` 还把"两开关不能混"钉成了测试
   —— **把这条测试的存在写进文档**，它就是"我们没混"的机器可验证证据。
   建议再补一张 2×2 交叉表（`TG_AMP` × `TG_BF16`），逐格写"允许声称什么"。
2. **每个数字都带机器名前缀。** 仓库已有正确先例：`CHANGELOG.md:60-62`
   （"V100 显存构成…**V100 = sm_70，硬件无 bf16**，加速比必须等 sm_80+"）、
   `docs/AMP_PLAN.md:140-152`（专列一节"为什么 bf16 的计算加速在 V100 上验不了"）。
   **把这条定为硬规则：任何 `X%`/`X×` 前面必须出现 `V100(sm_70)` 或 `A10(sm_86)`。**
3. **把"证据等级"写进文档**（照抄 §1 那张表）。误读往往不是因为措辞不好，而是因为
   "编译通过"和"跑起来了"被放在同一句里。

### 6.4 可执行的自检（放进 CI 或提交前跑）

```bash
# ① 任何"加速/speedup/倍/x"的提法，同行必须出现 sm_80/A100/A10/sm_86 或"未测"
grep -rniE "(bf16|amp).*(加速|speedup|[0-9.]+ ?x|倍)" docs/ README.md CHANGELOG.md

# ② 不允许出现无边界的"支持 bf16"
grep -rniE "支持 ?bf16|bf16 ?已" docs/ README.md CHANGELOG.md

# ③ 内核数必须与工具实测一致（现在应为 81 总 / 0 bf16）
python tools/amp_readiness.py --json
```

第 ③ 条现在会失败——因为工具本身扫不到位运算写法的 bf16 内核（见 M10）。
**在修工具之前，不要在任何文档里写"bf16 内核数 > 0"。**

---

## 7. 分析中发现的两个"非 AMP 但会毁掉 AMP 判据"的问题

这两条不在题目里，但会直接影响上面所有判据的可信度，建议优先处理。

### 7.1 `T.checkpoint` 在 CUDA 上是**空操作**

`__init__.py:2525-2528`：

```python
ts = [x for x in inputs if isinstance(x, Tensor)]
need = [x for x in ts if x.requires_grad and getattr(x, "device", "cpu") == "cpu"]
if not need:
    return fn(*inputs)              # 无可求导输入（或在设备上）：直接算，不折腾
```

⇒ 传进去的 `x` 是 CUDA 张量时，`need` 为空，**直接 `return fn(*inputs)`，一点内存都没省。**

影响面比听起来大：

- `qwen.py:245-248` 的训练路径就是 `T.checkpoint(self._layer, x, i)`；
- `qwen.py:638` 的 `chunked_ce` 也包在 `T.checkpoint` 里；
- `docs/AMP_PLAN.md:190-192` 把"大形状下打开检查点"列为**第二优先级的解药**；
- `tests/test_qwen_grad_ckpt.py:19-22` 的 fixture 是 `T.set_default_device("cpu")`
  —— **这条能力只在 CPU 上被验证过**，CUDA 上无人测。

**建议**：要么真做出 CUDA 版（反向重算，需在图里重建闭包），要么在 `use_ckpt=True` 且
device 为 cuda 时**响亮地打一行警告**："CUDA 上 checkpoint 未生效，显存不会下降"。
现在这种"静默无效"最危险——它会让人以为省了，从而把 bf16 的收益算错。

### 7.2 `tools/amp_readiness.py` 的 bf16 判定与"位运算实现"不兼容

`:43`：`has_bf16 = ("bfloat16" in a) or ("__nv_bfloat16" in a)`。
而我们的 bf16 内核形参是 `const unsigned short*`（`amp.py:70-84`，因为不能 `#include <cuda_bf16.h>`）。
⇒ 即使 S1 全部做完，这个工具**仍然报 0 个 bf16 内核**，
`docs/AMP_PLAN.md:41` 的验收判据 ③ 永远不通过。

**建议**：判定改成"名字带 `_bf16` 后缀 **或** 形参含 `unsigned short` **或** 源码里出现
`tg_bf16_to_f32`/`tg_f32_to_bf16`"，并给假阳性留一条注释（`cat_extract` 之类含 `bf16` 字样的要排除）。

---

## 8. 结论速览

1. `Tensor` **有** dtype，但 CUDA 张量恒报 `float32`（`__init__.py:637-638`），且 `_DTYPES` 里没有 bf16
   （`:370-375`）⇒ **没有可用的 bf16 dtype**；不过缓冲层 `GpuBuf.dtype` 是真的（`cuda.py:1898`），
   2 字节缓冲今天就能分配。
2. **可以只做"存 bf16、算 fp32"**，最小改动 10 处（§2.3 表），其中**必须最先做的是 launch 层的 dtype 闸门**
   （否则 2 字节缓冲会被 float 内核读越界，全仓只有 2 处 dtype 校验，`cuda.py:3626`/`:3655`）。
3. 三个 bf16 内核**至今没有接进引擎**（实测 `_device_source()` 不含它们），
   而且它们的语义是"读 bf16、用 float 算"，本质是**存储半边**，不是"真 AMP"。
4. **不需要 GradScaler**：bf16 指数位与 fp32 相同（丢的是尾数不是范围），且我们的梯度全程 fp32。
5. sm_70 上**只能**声称"bf16 存储省显存"；加速比必须 sm_80+ 实测，V100 不提供也不外推。
6. 两个必须先修的"判据污染源"：`T.checkpoint` 在 CUDA 上空转、`amp_readiness.py` 扫不到位运算 bf16 内核。

---

## 附录 · 引用清单

**源码**

- `python/tgai_go/train/amp.py`：`:21-22`（`TG_AMP`）、`:25-36`（`to_bf16_bits`）、
  `:39-42`（`from_bf16_bits`）、`:45-50`（往返误差）、`:53-106`（`BF16_KERNELS`，三个内核在
  `:70`/`:76`/`:83`）、`:109-113`（`amp_requested`）
- `python/tgai_go/train/cuda.py`：`:61`（`CUDA_R_16BF=14`）、`:88-100`（bf16 开关）、
  `:1492-1572`（`_FP16_KERNELS`；`:1557`/`:1567` 两个 bf16 cast 内核）、
  `:1809-1812`（`_device_source`，**不含** BF16_KERNELS）、`:1879-1932`（`GpuBuf`，`:1887` slots、`:1898` dtype）、
  `:2454-2463`（`_cc_arch`）、`:2683`（`pool_stats`）、`:3025-3035`（`empty`，`:3033` 按 itemsize）、
  `:3051-3129`（`h2d`/`d2h`/`clone`/`full` 均 dtype 通用）、`:3253-3265`（`_vec4_ok` 只查对齐）、
  `:3287-3289`（`_ptr` 无校验）、`:3305-3316`（`add` 输出固定 fp32，`:3307`）、
  `:3496-3508`（`permute` 是**真拷贝**）、`:3624-3627`/`:3652-3656`（仅有的 2 处 dtype 校验）、
  `:4393-4431`（`_bf16_state`/`_bf16_fail`/`_bf16_wanted`）、`:4442-4471`（`_cast_flat_ptr_bf16` 影子缓存）、
  `:4473-4499`（`_sgemm_bf16`，`:4499` 用 `CUBLAS_COMPUTE_32F`）、`:4501-4512`（`_sgemm` 分发）
- `python/tgai_go/train/__init__.py`：`:260-272`（`__all__`，`:270` 只有四个 dtype）、
  `:349-375`（`DType`/`_DTYPES`）、`:378-411`（`_as_dtype`/`_wrap_dtype`）、
  `:567-599`（`Tensor`，`:570-571` slots、`:597-599` CUDA 分支写死 fp32）、`:601-613`（`_from_gpu`）、
  `:635-639`（`dtype` 恒 float32）、`:2419-2436`（`scaled_dot_product_attention`）、
  `:2450-2477`（`linear_cross_entropy`）、`:2509-2559`（`checkpoint`，`:2525-2528` CPU-only）、
  `:2735-2750`（CE 反向原地复用 logits 缓冲）、`:2828-2833`（`Parameter` 恒 fp32）、
  `:3191-3236`（`RMSNorm`，`:3214` `_forward_cuda`、`:3228-3235` 闭包）
- `python/tgai_go/train/qwen.py`：`:28-29`（CFG）、`:178-183`（`_rope_apply` 造 4 个临时量）、
  `:185-223`（`_layer`）、`:226-253`（`forward`，`:245-248` 用 checkpoint）、
  `:615-642`（`chunked_ce`，`:638` 用 checkpoint）
- `python/tgai_go/train/backend.py`：`:21-23`（79 个接口）、`:42-92`（`GROUPS`）、
  `:96-102`（可选/回退组）、`:121-140`（`report`）、`:153-155`（`missing`）

**工具 / 测试 / 文档**

- `tools/amp_readiness.py`：`:26-34`（分类）、`:37-45`（扫描，`:43` 是误判点）、`:84-98`（输出）
- `tools/amp_compile_check.py`：`:58-102`（NVRTC 编译）、`:105-130`（`:107` 默认 `compute_86`、`:114-117` 拼 `BF16_KERNELS`）
- `tools/mem_profile.py`：`:28-66`（nvidia-smi 采样）、`:106-116`（peak/`peak_delta`）
- `tools/profile_step.py`：`:111-146`（kernel 次/步 + host ms，**S0 直接复用**）、
  `:134-137`（`pool_stats`）、`:148-198`（按 kernel 的 CUDA event 计时）
- `tests/test_amp_bf16.py`：`:18-28`（2×10⁴ 值、≥99.9%）、`:31-37`（2⁻⁸ 判据）、
  `:64-71`（两开关独立）、`:74-81`（无 `#include`、计算必须是 float）
- `tests/test_qwen_grad_ckpt.py`：`:19-22`（**CPU fixture**，CUDA 未测）
- `docs/AMP_PLAN.md`：`:11-13`（81 内核/0 bf16）、`:30-31`（第一步 41 个）、
  `:35-41`（S1 旧判据）、`:61-69`（风险表）、`:78-86`（S1 进展）、`:96-99`（2⁻⁸ 判据纠正）、
  `:108-138`（V100 显存构成）、`:140-152`（**为什么 V100 验不了加速**）、`:156-199`（分块 CE 与优先级）
- `CHANGELOG.md`：`:60-62`（显存构成 + "V100 = sm_70，硬件无 bf16"）

**外部依据（PyTorch 官方文档，用于 §4）**

- `torch.amp` 文档：autocast 机制、CUDA 三张算子表、`GradScaler` 的下溢原理
  —— <https://docs.pytorch.org/docs/stable/amp.md>
