# PyTorch FP16 全链路剖析（C 端 → Python 端）

> 源码：本机 `D:\DSH-Workspace\pytorch-src`，版本 2.16.0a0。
> 所有结论都标了 `文件:行号`，可逐条复核。

---

## 0. 全景：一次 `matmul` 在 autocast 下穿过哪几层

```
Python:  with torch.autocast("cuda", dtype=torch.float16):
             y = x @ w
             ↓
[1] Python torch/amp/autocast_mode.py       —— 只是设置/清理「autocast 状态」
             ↓
[2] Dispatcher: dispatch key = Autocast    —— aten/src/ATen/autocast_mode.cpp
    · 查算子属于哪一类：lower_precision_fp / fp32 / promote
    · 对每个 Tensor 入参调用 cached_cast(...)
             ↓
[3] cached_cast                            —— autocast_mode.cpp:122
    · 只对「叶子权重」做 fp32→fp16 并缓存（键 = TensorImpl*，值含 weakref）
    · 激活值一律现 cast、不缓存
             ↓
[4] 重新派发到 CUDA 真内核                  —— aten/src/ATen/native/cuda/Blas.cpp
    · addmm：有 bias → gemm_and_bias(cuBLASLt)；无 bias → gemm
    · 决定输出 dtype（res_scalar_t）与是否允许 reduced precision
             ↓
[5] cuBLAS 封装                             —— aten/src/ATen/cuda/CUDABlas.cpp
    · cublasGemmEx 或 cublasLtMatmul
    · 选 computeType：默认 fp32 累加，fp16 累加默认关闭
    · 选 reduction scheme：默认允许降精度（split-K 归约可用 fp16）
             ↓
[6] 硬件：Tensor Core（fp16 输入 × fp32 累加）
```

**关键点：整条链路里没有「损失缩放」。** 缩放完全在 Python（`GradScaler`）里，
用 ATen 的逐元素 kernel 做，与 GEMM 链路无关（见 §7）。

---

## 1. Python 端：`torch.autocast`

`torch/amp/autocast_mode.py`（CUDA 专用版本另有 `torch/cuda/amp/autocast_mode.py`）

职责只有三件：

1. 进 `__enter__`：调 `torch._C._enter_autocast(device_type, dtype, cache_enabled)`，
   把「autocast 开、目标 dtype、是否启用 cast 缓存」写进 C++ 的 thread-local 状态。
2. 出 `__exit__`：调 `torch._C._exit_autocast(device_type)`。
3. 提供装饰器 `custom_fwd/custom_bwd`，让自定义 autograd.Function 也能遵守 autocast。

**它自己不做任何 cast** —— 真正的 cast 发生在 dispatcher 里。
`cache_enabled` 这个参数直接对应 C++ 的 `thread_local bool cache_enabled`
（`autocast_mode.cpp:83`），可以关掉缓存来排查精度问题。

---

## 2. 派发层：Autocast dispatch key 与算子白名单

`aten/src/ATen/autocast_mode.cpp:181` 起，用 `TORCH_LIBRARY_IMPL(aten, Autocast, m)`
把一批算子注册到 **Autocast 这个 dispatch key** 上。算子按四种策略分类
（宏在 `aten/src/ATen/autocast_mode.h`）：

| 策略 | 含义 | 代表算子 |
|---|---|---|
| `lower_precision_fp` | 入参降到 fp16，**输出也是 fp16** | `mm` `addmm` `matmul` `bmm` `baddbmm` `linear` `einsum` `conv1d/2d/3d` `conv_transpose*` `_convolution` `addmv` `addr` `mv` `chain_matmul` `linalg_multi_dot` `prelu` `_thnn_fused_lstm_cell` |
| `fp32` | 强制以 fp32 执行 | `pow` `exp` `log` `softmax` `log_softmax` `layer_norm` `group_norm` `batch_norm` `alpha_dropout` `norm`(部分) `nll_loss` `cross_entropy` `sum` `prod` `cumsum` 等 |
| `fp32_set_opt_dtype` | 以 fp32 执行，并把可选 `dtype` 参数补成 fp32 | `softmax`(dtype 变体) `norm`(部分) 等 |
| `promote` | 按「最宽类型」提升（fp16+fp32 → fp32） | `cat` `stack` `index_copy` `add` `mul` 等逐元素/拼接类 |

**为什么 GEMM 归 `lower_precision_fp`、而 `softmax`/`layer_norm` 归 `fp32`？**
因为前者在 fp16 下有 Tensor Core 巨额提速，后者是**归约型**算子 ——
fp16 归约会丢有效位，PyTorch 选择让它们在 fp32 里算（这就是"混合精度"的由来）。

`promote` 的存在保证 `fp16_tensor + fp32_tensor` 不会静默降精度，而是升到 fp32。

**被禁用的算子**：`binary_cross_entropy`（`autocast_mode.cpp:160`）直接 `TORCH_CHECK(false)`，
因为它通常紧跟 sigmoid，fp16 下极易 NaN。

---

## 3. `cached_cast`：PyTorch 唯一的"cast 缓存"，与我们的 bug 正相反

`aten/src/ATen/autocast_mode.cpp:122`：

```cpp
Tensor cached_cast(at::ScalarType to_type, const Tensor& arg, DeviceType device_type) {
  if (is_eligible(arg, device_type) && (arg.scalar_type() != to_type)) {
    bool can_try_cache = (to_type == get_lower_precision_fp_from_device_type(device_type) &&
                         arg.scalar_type() == at::kFloat && arg.requires_grad() &&
                         arg.is_leaf() && !arg.is_view() && cache_enabled &&
                         !at::caching::is_cached_tensor(arg) &&
                         !c10::InferenceMode::is_enabled());
    if (can_try_cache) {
      auto it = get_cached_casts().find(arg.unsafeGetTensorImpl());
      if (it != end) return std::get<1>(it->second);
      ...
      get_cached_casts().emplace(arg.unsafeGetTensorImpl(),
                                 val_type{weakref_type(arg.getIntrusivePtr()), casted_arg});
```

缓存容器（`autocast_mode.cpp:41`）：
```cpp
ska::flat_hash_map<TensorImpl*, val_type> cached_casts;   // 键 = 张量对象身份
```

### 五条必须同时满足的缓存条件

| 条件 | 为什么 |
|---|---|
| `to_type == lower_precision_fp` | 只缓存"降精度"这一方向，反方向不缓存 |
| `arg.scalar_type() == kFloat` | 只缓存 fp32→fp16，不缓存 fp16→fp32 |
| `arg.requires_grad() && arg.is_leaf()` | **只缓存叶子 —— 实际就是模型权重** |
| `!arg.is_view()` | 视图不缓存（生命周期/别名不可控） |
| `cache_enabled && !is_cached_tensor && !InferenceMode` | 显式开关 + 排除 inference_mode（那里 `enable_grad()` 无法覆盖） |

### 三条"防悬挂"设计

1. **键是 `TensorImpl*`（对象身份），不是数据指针** —— 两个不同张量即便复用同一块显存，
   也不会命中同一条缓存。
2. **值是 `weakref`**（`weakref_type(arg.getIntrusivePtr())`）—— 原张量一死，条目随之失效，
   不会留下指向已释放内存的 cache。
3. **`clear_cache()`**（`autocast_mode.cpp:88`，Python 侧 `torch.clear_autocast_cache()`）
   —— 能整体清空，用于显式抛弃缓存。

> **这与 ORV 0.9.87 修的 bug 恰好是反面教材。**
> 我们当时把 fp16 影子缓存键成**裸数据指针**，于是：
> 激活值从内存池复用地址 → 反向命中前向留在该地址的**别的张量**的 fp16 拷贝
> → 梯度错 **4130 倍**。
> PyTorch 用「对象身份 + 只要叶子 + weakref + 可清空」四重保险避免了这件事。
> **ORV 的修法（只缓存"登记过的参数地址"）在语义上等价于 PyTorch 的 `is_leaf()` 限制。**

---

## 4. ATen CUDA 内核层：`addmm` / `bmm` 怎么选精度

`aten/src/ATen/native/cuda/Blas.cpp`

### 4.1 `addmm` 的两条分支（`:353` / `:507` / `:535`）

```cpp
// 有 bias：直接进 cuBLASLt 的 fused epilogue（bias 在 GEMM 内部加，省一次读写）
return at::cuda::blas::gemm_and_bias<scalar_t, res_scalar_t>(...);   // :365, :535
// 无 bias：走通用 gemm
at::cuda::blas::gemm<scalar_t, res_scalar_t>(...);                   // :507
```

`res_scalar_t` 决定**输出** dtype：autocast 下 fp16 输入 → `res_scalar_t = Half` → 输出 fp16。

### 4.2 `bmm` 的累加精度开关（`CUDABlas.cpp:604`）

```cpp
if (prop->major >= 7 && at::globalContext().allowFP16AccumulationCuBLAS()) {
    halpha = alpha; hbeta = beta;
    compute_type = CUDA_R_16F;              // ← 真·fp16 累加
    alpha_ptr = &halpha; beta_ptr = &hbeta;
}
cublasGemmStridedBatchedEx(..., compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// 输出：std::is_same_v<C_Dtype,float> ? CUDA_R_32F : CUDA_R_16F
```

`alpha`/`beta`（缩放系数）也随 compute_type 变成 `at::Half` —— 这是容易被忽略的细节：
**fp16 累加时，alpha/beta 本身也只剩 fp16 精度**。

---

## 5. cuBLAS 封装层：compute type 与 reduction scheme

`aten/src/ATen/cuda/CUDABlas.cpp`

### 5.1 cublasLt 路径（现代默认）

```cpp
const auto type_info = detail::getCublasLtTypeInfo<Dtype, C_Dtype>();   // :264
const cudaDataType_t abType    = type_info.ab_type;      // 输入
const cudaDataType_t cType     = type_info.c_type;       // 输出
const cublasComputeType_t computeType = type_info.compute_type;  // 累加
const cudaDataType_t scaleType = type_info.scale_type;
```

**对 `gemm<at::Half, float>`（fp16 输入、fp32 输出）：`computeType = CUBLAS_COMPUTE_32F`。**

### 5.2 两个全局开关，默认值已核实

| 开关 | 默认 | 位置 |
|---|---|---|
| `allowFP16AccumulationCuBLAS()` | **`false`** | `Context.h:535` `bool allow_fp16_accumulation_cublas = false;` |
| `allowFP16ReductionCuBLAS()` | **`AllowReducedPrecisionWithSplitK`**（枚举值 0） | `Context.h:531`、`Context.h:47` |

**这意味着默认配置下：**

* **累加在 fp32** —— 不是 fp16 累加，数值安全；
* **归约（reduction / split-K）允许降精度** —— 这是唯一"偷偷"用 fp16 的地方，
  对应 `CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK`（`CUDABlas.cpp:288-295`），
  可用 `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False` 关掉。

### 5.3 fp32 GEMM 的 TF32 开关（顺带澄清）

`float32_matmul_precision` / `allow_tf32` 只影响 **fp32 输入**的 GEMM
（是否用 TF32 Tensor Core），与 fp16 路径无关。两条路径互不影响，别混。

---

## 6. 数值后果：到底哪些位真的在 fp16 里

| 环节 | dtype | 依据 |
|---|---|---|
| 权重（参数本体） | **fp32** | autocast 只 cast 一份影子，`module.weight` 仍是 fp32 |
| GEMM 输入 A/B | fp16（cast 影子） | `cached_cast` |
| GEMM 乘加 | **fp32 累加** | `compute_type = CUBLAS_COMPUTE_32F` |
| GEMM 输出 | fp16 | `res_scalar_t = Half` |
| softmax / layernorm / loss | **fp32** | `AT_FORALL_FP32` |
| 梯度 | fp16（链式经过 fp16 输出） | 反向图沿用前向 dtype |
| 优化器状态 m/v | **fp32** | `AdamW` 在 fp32 参数上 |
| 主权重更新 | **fp32** | `param.add_(update)`，param 是 fp32 |

**结论：PyTorch 的 "AMP" 从来不是"fp16 训练"，而是"GEMM 用 fp16 输入 + fp32 累加"。
参数、梯度之外的优化器状态、以及所有归约算子都是 fp32。**

---

## 7. 损失缩放：在 Python 和逐元素 kernel 里，与 GEMM 无关

`torch/amp/grad_scaler.py`

| 动作 | 实现 |
|---|---|
| `scale(loss)` | 就是 `loss * scale`（乘法，`grad_scaler.py:187` 起的重载） |
| 反向 | 因 loss 被放大，整条链的梯度同比放大 |
| `unscale_(opt)` | 调 `torch._amp_foreach_non_finite_check_and_unscale_`（`:299`）—— **一个 C++ 融合 kernel**，同时做「检查 inf/nan」与「除以 scale」 |
| `step(opt)` | 若 `found_inf` 非零就跳过本步（`OptState` 状态机，`:50`） |
| `update()` | 动态调整 scale（连续成功则 ×2，出现 inf 则 ×0.5） |

**所以 loss scaling 与 §2–§5 的 GEMM 链路完全解耦。**
这也解释了为什么"fp16 GEMM 很快但不稳"和"loss scale 调不好"是两个独立问题。

---

## 8. PyTorch **没有**的东西（重要）

| 能力 | PyTorch 现状 |
|---|---|
| 纯 fp16 训练（参数本体 fp16） | **没有官方路径**。`model.half()` 会让优化器也进 fp16，Adam 的 v 直接下溢 |
| fp16 优化器状态 | **不支持**（Adam 的 `step` 对 half 参数会报错或数值崩） |
| fp16 累加 | 有开关，但**默认关闭** |
| bf16 优化器状态 | 也**没有**内置；社区做法是自定义（如 bitsandbytes 的 8-bit） |

> 我们实测 ORV 的 `--opt fp16mom --opt-state fp16` 第 2 步发散
> （loss 11.7 → 135，gnorm 1642），根因是 **Adam 的 v 典型 1e-8 而 fp16 最小正规数 6.1e-5**
> ⇒ v 下溢成 0。**这不是 ORV 的实现问题，是 fp16 表示范围决定的**；
> 换 bf16（指数范围同 fp32）后立即正常（loss 正常下降到 10.5，省 5.6 GiB）。
> PyTorch 之所以"没有这个问题"，是因为它**压根不支持** fp16 优化器状态。

---

## 9. 对 ORV 的启示（该抄什么、不该抄什么）

### 值得抄

1. **算子分级策略**：GEMM/conv 走低精度，归约（softmax/layernorm/loss）留 fp32。
   我们若做真 AMP，必须显式列出这三类白名单，而不是"全局 fp16"。
2. **缓存四重保险**：身份键 + 只叶子 + weakref + 可清空。（我们已用"只参数"达到等价效果。）
3. **`addmm` 走 fused epilogue**：bias 在 GEMM 内加，省一次全尺寸读写。
4. **开关默认值保守**：fp16 累加默认关、reduced-precision 归约可关。

### 不该抄 / 抄了也没用

1. **fp16 累加**：V100 上 `compute_type=R_16F` 会掉精度，且我们实测**整模型根本不吃 GEMM 算力**
   （20 层 fp16 全开反而慢 21%，因为瓶颈是 CPU 侧的派发与分配器）。
2. **把 fp16 当"提速手段"**：我们的 1.5B 训练里，GEMM 只占 GPU 时间的一部分，
   大头是逐元素链与 CPU 开销 —— 这与 PyTorch 的瓶颈结构完全不同，
   所以"PyTorch 用 AMP 有收益"不能推出"ORV 用 fp16 有收益"。
3. **fp16 优化器状态**：数学上不可行（v 下溢）。**要用就用 bf16**，我们已验证。

---

## 10. 索引（复核用）

| 主题 | 文件:行 |
|---|---|
| autocast 算子白名单 | `aten/src/ATen/autocast_mode.cpp:181-224` |
| 算子分类宏 | `aten/src/ATen/autocast_mode.h`（`AT_FORALL_LOWER_PRECISION_FP` / `_FP32` / `_PROMOTE`） |
| cast 缓存容器 | `aten/src/ATen/autocast_mode.cpp:41-45` |
| `clear_cache` | `aten/src/ATen/autocast_mode.cpp:88` |
| `cached_cast` | `aten/src/ATen/autocast_mode.cpp:122-158` |
| BCE 禁用 | `aten/src/ATen/autocast_mode.cpp:160` |
| `addmm` 派发 | `aten/src/ATen/native/cuda/Blas.cpp:353` / `:365` / `:507` / `:535` |
| `bmm` fp16 累加开关 | `aten/src/ATen/cuda/CUDABlas.cpp:604-620` |
| cublasLt type info | `aten/src/ATen/cuda/CUDABlas.cpp:264-268` |
| reduction scheme mask | `aten/src/ATen/cuda/CUDABlas.cpp:288-295` |
| fp16 累加默认关 | `aten/src/ATen/Context.h:535` |
| fp16 归约默认允许 | `aten/src/ATen/Context.h:531`、`Context.h:47` |
| `allowFP16AccumulationCuBLAS` 实现 | `aten/src/ATen/Context.cpp:806` |
| GradScaler unscale kernel | `torch/amp/grad_scaler.py:299` |
| GradScaler 状态机 | `torch/amp/grad_scaler.py:50` |
