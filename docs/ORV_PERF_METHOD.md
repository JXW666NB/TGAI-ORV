# ORV 性能剖析方法与实测结论（2026-11-17）

> 这份文档记录**怎么定位**性能问题的（方法论），以及已经拿到的数字。
> 后续优化请照这套流程走，别再靠墙钟猜。

## 一、方法论：四层剖析，逐层下钻

| 层 | 工具 | 回答什么 |
|---|---|---|
| 1 | `lab/prof/phase_breakdown.py` | 一步的 fwd / bwd / opt 各占多久（**多次取中位数**）|
| 2 | `lab/prof/op_time.py` | **每个内核的真实 GPU 耗时**（CUDA event 包裹 `rt.launch`）|
| 3 | `lab/prof/cpu_profile.py` | CPU 侧热点（cProfile，**只 profile 第二个 step**）|
| 4 | 数组形状日志 | `h2d`/`d2h` 在搬多大的数组（本次一击命中）|

### 三个必须遵守的测量纪律

1. **单次测量噪声可达 60%**。同一配置单次测出 180 / 186 / 294 ms 都出现过。
   一律跑 5 次取中位数。
2. **`-m cProfile` 会把预热算进去**。build + NVRTC 编译约 5.5 s，
   必须用 `cProfile.Profile()` 显式包住"第二个 step"。
3. **墙钟时间 ≠ GPU 时间**。分不清这两者就会得出完全错误的结论（我犯过）。

## 二、本次定位过程（1.53× 提速的来源）

### 第 1 步：分段 → 发现 backward 异常
1 层 / seq256（5 次中位）：fwd 13.6 ms、**bwd 185.4 ms**、opt 8.7 ms。

### 第 2 步：CUDA event → 发现 97% 是 CPU

| 阶段 | GPU 时间 | 墙钟 |
|---|---:|---:|
| forward | 2.5 ms | 13.6 ms |
| backward | **3.8 ms** | **185.4 ms** |
| opt.step | 8.5 ms | 8.7 ms |

### 第 3 步：cProfile → 锁定 `numpy.array` + `h2d`
`numpy.array` 46 次 0.105 s；`h2d` 21 次 0.145 s（一步共 0.224 s）。

### 第 4 步：打印数组形状 → 一击命中
```
((96000, 1024), 'float32') × 2     ← 每步搬 393 MB × 2
```

### 根因与修复
`Tensor.__getitem__` 的**反向**（`__init__.py:1815`）走 CPU 中转：
D2H 梯度 → `np.zeros(shape)`（主机 393 MB）→ `np.add.at` 慢速散射 → 整块 H2D。
库内早有 GPU 版 `scatter_add_rows`（`indexing.py` 一直在用），改接过去。

**数值验证**：`scatter_add_rows` vs `np.add.at` 逐元素 `max_abs = 0.000e+00`，含重复索引同样逐位相等。

## 三、A/B 实测结果

| 配置 | 修改前 | 修改后 | PyTorch | 对比 |
|---|---:|---:|---:|---|
| 20 层 / seq512 | 0.647 s (792 tok/s) | **0.422 s (1213 tok/s)** | 0.831 s (616 tok/s) | ORV 快 **1.97×** |
| 4 层 / seq512 | 0.264 s (1940 tok/s) | **0.111 s (4600 tok/s)** | 0.160 s | ORV 快 **1.4×** |

CPU profile：一步从 0.224 s 降到 **0.056 s**。

## 四、已排除的方向（别再试）

| 方向 | 结论 | 证据 |
|---|---|---|
| fp16 GEMM 提速 | **无净收益**（任何门控阈值）| 262144 / 1e6 / 4e6 全试；门控全关≈fp32 ⇒ 慢在 fp16 GEMM 本身 |
| fp16 权重影子缓存 | 曾是**正确性 bug**（已修）| 指针缓存读到池复用地址，梯度错 4130× |
| `cublasSgemmGroupedBatched` | **不可用** | 调用报 "parameter number 2 had an illegal value" |
| 优化器 | 健康 | 1 个融合内核、8.5 ms |
| 融合 RMSNorm | **已在用** | `v3_ops.rmsnorm_orv_fused` → `rmsnorm_fwd` |

## 五、尚未达成的目标

**一步内核数 1226 → ≤700 未达成。** 当前分布（4 层 / seq256）：

```
ewise_mul_v4 201  broadcast_to8 196  fused_add_ip_v4 107  swap_last2 86
sum_axis8 77  ewise_scale_v4 54  ewise_pow 42  index_gather8 40
permute8_v4 40  index_scatter_add8 40  transpose2d 38
```

注意：嵌入修复后**每个内核的 CPU 成本只有约 10 µs**，1226 个合计约 12 ms ——
相对现在 111 ms / 422 ms 的步时，**继续减内核的收益已经不大**。
若要继续，优先级应是"融合大块的逐元素链"，而不是单纯减小内核数。

## 六、发布状态

`0.9.89` 的 wheel 已手工构建并校验（`dist/tgai_orv-0.9.89-py3-none-any.whl`，
102 项、`zip ok`、确认含修复），但**当时出口网络不通**
（`git push` → `schannel: failed to receive handshake`；
`twine upload` → `ConnectionError 10054`；代理 7890 端口在听但不转发）。

网络恢复后执行：

```powershell
cd D:\DSH-Workspace\TGAI-对话档案\_tgai_orv
git push origin main
$env:HTTPS_PROXY='http://127.0.0.1:7890'; $env:TWINE_USERNAME='__token__'
$env:TWINE_PASSWORD='<PyPI token>'
python -X utf8 -m twine upload --disable-progress-bar --non-interactive dist/tgai_orv-0.9.89-py3-none-any.whl
```

若 `python -m build` 因隔离环境装依赖失败（无网络时必现），用
`python -X utf8 lab/_build_wheel.py` 手工打包（纯 Python 包，不需要 build 后端）。

## 七、1.5B 生产配置的步时波动（2026-11-17 实测，未解决）

嵌入反向修复后，不同规模的稳定性差异很大：

| 配置 | 8~12 步的步时 | 稳定性 |
|---|---|---|
| 1.13B（20 层 / d_ff 2816）| **0.417 ~ 0.435 s** | 极稳，无趋势 |
| 1.5B（24 层 / d_ff 3328）| **0.527 ~ 1.164 s** | 波动 2.2x，且有缓慢上升 |

1.5B 的逐位重放结果（同参数跑两次，步时**逐位相同**）：

```
8 步 : 1.075 0.646 0.541 0.698 0.527 1.149 1.146 1.094
12 步: 1.068 0.648 0.547 0.649 0.631 0.857 1.150 1.097 1.164 0.943 1.038 0.765
```

**已排除的原因**：
- `--policy memory` / `--policy speed`：步时**完全一致**（连小数都相同）=> 不是池策略
- 显存泄漏：`peak_live` 恒为 27.08 GiB，不增长
- 随机噪声：两次运行逐位相同 => 确定性

**当前判断**（未最终证实）：步时是「step 序号的确定性函数」，来源可能是
(1) 不同 batch 的文档结构不同 => CSA 分块数 / seg 掩码规模不同；
(2) 分配器状态的确定性演化（碎片化）。

**影响**：1.5B 的中位步时约 0.83~0.90 s（450~970 tok/s），
所以「1.13B 上 1.53x 的提速」**不能直接外推到 1.5B**，那里中位约改善 19%。

**下一步该做的**：给 1.5B 引入梯度重算（gradient checkpointing）以换取显存余量
（当前 27.08 / 32 GiB，余量仅 4.8 GiB），既减少分配压力，也才可能支持 seq 1024。

### 七之补充：根因已定位（2026-11-17）

上一节说「CSA 分块或分配器演化」，**两个猜测都被实测否定了**：

- **CSA 分块不是原因**：打桩 `v3_csa.make_plan` 记录 `Tc`/`fallback`，12 步里
  `Tc` 恒为 128；`fallback` 只在第 5、7 步为 1，但第 6、8、10 步（`fallback=0`）同样慢。
- **不是总计算量变化**：每步 GEMM 调用数恒定 **1497**、总计算量恒定 **2778 GFLOP**。

**真正的变量是 MoE 每个专家分到的 token 数**（`_sgemm` 的 `n` 最小值）：

| step | 步时 (s) | GEMM 调用 | 总 GFLOP | 每专家最小 n |
| ---: | ---: | ---: | ---: | ---: |
| 0–4 | 0.537 ~ 1.068 | 1497 | 2778 | **128**（均衡）|
| 5 | 0.854 | 1497 | 2778 | 50 |
| 6 | 1.153 | 1497 | 2778 | 79 |
| 7 | 1.098 | 1497 | 2778 | 34 |
| 8 | 1.162 | 1497 | 2778 | 42 |
| 9 | 0.941 | 1497 | 2778 | 61 |
| 10 | 1.053 | 1488 | 2778 | 15 |
| 11 | 0.766 | 1479 | 2778 | 9 |

均衡时（每专家 128 token）步时 0.54~0.65 s；一旦某些专家只分到 9~79 个 token，
步时升到 0.85~1.16 s（**+70%**）。原因很直接：MoE 是**逐专家小 GEMM**，
`m=2816, n=9` 这种极瘦形状在 V100 上固定开销占绝对主导。

**结论**：1.5B 的吞吐波动**不是内存、不是分块、不是泄漏**，而是
**路由负载不均放大了瘦长 GEMM 的低效**。

**可行修法（按风险从低到高）**：
1. **把每专家的 token 维 N 向上对齐**（如补齐到 64/128 的倍数，补零并在 scatter 前清零）——
   计算量略增，但 GEMM 落到有效 tile 上，通常反而更快；
2. 加 capacity factor（限制每专家 token 上限 + 溢出丢弃/回退）；
3. 改成分组 GEMM —— **本机不可用**（`cublasSgemmGroupedBatched` 报 illegal value）。

这三条都还没实现，属于下一轮的工作。
