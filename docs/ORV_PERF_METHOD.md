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
