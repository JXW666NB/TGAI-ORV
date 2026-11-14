# 设计参考：CUTLASS fp32/sm_70 的 fmha 是怎么做到 48 KB 以内的

> 来源：子智能体 `cutlass-research` 对 PyTorch v2.5.1
> `aten/src/ATen/native/transformers/cuda/mem_eff_attention/` 的源码核对（2026-11-15）。
> 文末标了"推算""未找到"的条目不要当事实引用。

## 0. 先纠正我们之前的三个错误前提

| 我们以为 | 源码事实 |
| --- | --- |
| fp32/sm_70 用 `32x128` | **DH ≤ 64 时用 `64x64_rf_sm70`**。`32x128` 只在 DH ∈ (64,128] 才轮到 |
| 反向也是 32x128 | 反向是 `64x64_k64_sm70`，`64x64` = (query 块 64 × **key 块 64**)，与 fwd 语义不同 |
| fp32 会用到 Tensor Core | **不用**。sm_70 落回 `OpClassSimt`，`InstructionShape<1,1,1>`，纯 SIMT FMA |

## 1. 它为什么只用 ~42 KB：**Q/K/V/dO 一个都不常驻 smem**

CUTLASS 只把 head_dim 的**一个 16 宽滑窗**放进 smem（`ThreadK=8 × 2 stages`）：

| 缓冲 | 我们 | CUTLASS |
| --- | --- | --- |
| Q | 64×72 = 4608 float（18 KB，常驻）| `BT × 16` = 1024 float（4 KB，滑窗）|
| dO | 4608 float（18 KB，常驻）| 同 Q 量级，滑窗 |
| K | 32×64 = 2048（8 KB）| `16 × BC` ≈ 4 KB |
| V | 2048（8 KB）| `16 × DH` 滑窗 |
| P | 64×32 = 2048（8 KB）| `BT × BC` = 16 KB（64×64）|
| dS | 2048（8 KB）| 与 P **union 复用** |

**关键认识**：head_dim 是**求和维**，沿它切片累加在数学上完全等价 ——
"分块不成立"只针对 **key 维**（softmax 归一化），对 head_dim 切分没有这个问题。

⇒ **砍掉 Q/dO 常驻就能从 68 KB 直接进 48 KB 以内，根本不用动 BT**。
这正是我前三次一直没做对的地方：我一直在动 BT，而正确的刀口在 head_dim。

## 2. 我们三次失败的**可能原因**（CUTLASS 的硬约束）

`kernel_forward.h` 里有两条 `static_assert`：

```cpp
static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");
static_assert(kQueriesPerBlock % kNumWarpsPerBlock == 0, "");
// kNumWarpsPerBlock = kQueriesPerBlock * kKeysPerBlock / (32 * 32)
```

代入：BT=64/BC=32 → warps=2 → `64 < 64` 失败；BT=32/BC=32 → warps=1 → `32 < 32` 失败。
**CUTLASS 合法的档位只有 64×64、32×128、64×128。**

它的 softmax 行归约用 `thread_id < BT` 初始化、`kLinesPerWarp = BT / kNumWarpsPerBlock` 分行 ——
**BT 一改而线程映射没跟着改，就是"编译通过、梯度全错"**，与我们观察到的症状完全吻合。

## 3. 其它值得抄的设计

| 做法 | 为什么 |
| --- | --- |
| **dK/dV 不用 atomicAdd、不放 smem** | 循环改成"key 在外、query 在内"，同一 CTA 顺序扫完该 key 块，跨块累加变顺序累加，epilogue `gmem += accum` 即可。源码里 `storeAtomicAdd` **定义了但从未调用** |
| **dQ 用 gmem workspace + 计数收口** | V100 走 split-key：`store → __threadfence() → __syncthreads() → release lock`（顺序不能改）|
| **P 存未归一化的 `exp2(...)`** | 归一化只在最后 epilogue 做一次 |
| **padding 只有 4 个元素** | `simt_transpose_padding(32, 8, 32) = 4`，加在 M 方向的行上；**不是**我们用的"行步长 = DH+8" |
| **smem 超限时静默跳过该 kernel** | PyTorch：`if (smem_bytes > sharedMemPerBlockOptin) return;` —— 这就是 P100 上"跑不起来"的直接机制 |

## 4. 下一步（在 `lab/attn32/` 里做，不碰生产）

1. **先砍 Q/dO 常驻**：改成沿 head_dim 的 16 宽滑窗（双缓冲），验证对拍
2. 若还超，再考虑 dS 与 P 的 union 复用（需一次 `__syncthreads()` 保证读完 P）
3. **不要动 BT**（除非同时按 `BT < warps*32` 与整除关系重排线程映射）

## 5. 未找到 / 未验证（不要当事实）

- `sizeof(SharedStorage)` 的**精确字节数没有公开值**；文中 22/25/42 KB 是按源码形状**手算**
- `MemoryEfficientAttentionSharedStorage` / `kSharedStorageSize` 这两个符号在 pytorch 中**不存在**
- epilogue 的 smem 字节数未核对（bwd 的 42 KB 有数 KB 不确定度）
- 想拿真值：`cuobjdump --dump-resource-usage libtorch_cuda.so | grep -A2 fmha_cutlassB_f32_aligned_64x64_k64_sm70`
