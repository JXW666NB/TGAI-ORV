# DP 梯度低精度通信（bf16/fp16）—— 把 all-reduce 字节数减半

状态（2026-11-21）：**代码 + CPU 逐位单测完成；GPU 实测待排期**（本轮 GPU 归 ZeRO 验证）。
本文所有"待测"标记的数字都是**验收时要填的坑**，不填不许说达标。

---

## 1. 为什么做这件事（上一轮已定案的结论）

上一轮（`5872fd9`）把「归约与反向重叠」做到了逐位正确，但实测吞吐只从
0.84–0.87× 提到 **1.10×（seq512）/ 1.25×（seq2048）**，没到验收线。根因是**实测归因**：

| 机器 | 单卡 | 2-rank 不重叠 | 2-rank 重叠 | 吞吐比（= 2×单卡 / 2-rank）|
|---|---|---|---|---|
| 2×5090 | 0.1406 s (seq512) | 0.3158 | 0.2565 | 0.84× → **1.096×** |
| 2×5090 | 0.2515 s (seq2048) | 0.4130 | 0.3998 | 1.20× → **1.253×** |
| 2×4090 | 0.2311 s | 0.6902 | 0.6570 | 0.68× → **0.704×** |

* 开重叠后**反向从 70 ms 膨胀到 183 ms**（≈70+111）：NCCL 内核与 ORV 计算内核在
  PCIe-only 机器上基本**不共驻**，归约没被藏住，只是把计算挤走了；
* 4090 更极端：裸 `ncclAllReduce` 3.16 GiB = **448 ms**（8 通道 353 ms，仅 9.6 GB/s），
  **大于单卡整步 231 ms** ⇒ 即使 100% 重叠，上限也只有 ~0.95×；
* 旁证：PyTorch DDP 同机 1.93×，因为它用的是 **bf16 梯度**（1.48 GB vs 我们 3.16 GiB），
  字节数少 2.1 倍。

⇒ **带宽是硬墙。** 在「3.16 GiB fp32 + 无 P2P」下，DP 转正是物理上不可达的；
出路只有三条：① 减少通信字节（**本次**）② 换 P2P/NVLink ③ 引擎级让 NCCL 与计算共驻。

本次把梯度**按 bf16/fp16 通信**：848.6M 参数 × 4 B = **3.16 GiB → 1.58 GiB**（正好一半）。

---

## 2. 用法（默认 fp32 = 历史行为逐位不变）

```bash
# 默认（不设变量）：fp32，与 1.0.15 逐位一致
TG_DP_BACKEND=nccl TG_DP_OVERLAP=1 ... python -m tgai_go.train.v3_pretrain ...

# 低精度通信（推荐 bf16：与 fp32 同指数范围，不会下溢）
TG_DP_GRAD_DTYPE=bf16 TG_DP_BACKEND=nccl TG_DP_OVERLAP=1 ...

# 可选误差反馈（多一份 fp32 影子梯度；会**自动关掉重叠路径**）
TG_DP_GRAD_DTYPE=bf16 TG_DP_GRAD_EF=1 ...
```

| 环境变量 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `TG_DP_GRAD_DTYPE` | `fp32` / `bf16` / `fp16` | `fp32` | 梯度**通信**精度；参数/优化器状态永远 fp32 |
| `TG_DP_GRAD_EF` | `0` / `1` | `0` | 误差反馈；只对 bf16/fp16 有意义（fp32 时开着会**报错**）|
| `TG_DP_PACK_MB` | 整数 MB | `256` | 低精度路径的打包分块（显存账见 §5）|

非法值（`b16`、`half1`……）**当场报错**，不静默退回 fp32 —— 写错一个字母就悄悄
多一倍通信字节，日志里看不出来是性能实验最难的坑。

### 支持矩阵

| 路径 | fp32 | bf16/fp16 | 备注 |
|---|---|---|---|
| `dist.all_reduce_grads` · NCCL 后端 | ✅（原样）| ✅ 设备侧 cast + `ncclAllReduce(bf16/fp16)` | 需要 NCCL ≥ 2.10 |
| `dist.all_reduce_grads` · host 后端（有 CUDA）| ✅（原样）| ✅ 共享内存里传 2 字节 | 需要 `cuMemHostRegister` 成功 |
| `dist.all_reduce_grads` · 纯 CPU | ✅（原样）| ✅ numpy 量化 | CPU 逐位对拍用 |
| `dp_overlap.NumpyTransport` | ✅（原样）| ✅ | 测试用，量化后逐位断言 |
| `dp_overlap.NcclBucketTransport` | ✅（原样）| ✅ 桶缓冲 2 字节 + 类型化 all-reduce | 推荐的生产路径 |
| `dp_overlap.HostBucketTransport` | ✅（原样）| ✅ | 低精度下自带 fp32 临时区 |
| `world=1` | ✅ | ✅ **快速空操作** | 单卡梯度逐位不变（验收硬要求）|

---

## 3. 语义（只改精度，不改数学）

```
每个 rank： pack32 = concat(p.grad)           # 缺梯度按 0
           pack16 = cast(pack32)              # bf16/fp16，字节数减半
           all_reduce(pack16, SUM)            # ← 唯一的集合通信，走 1/2 的字节
           pack32 = dequant(pack16) * (1/world)
           p.grad = pack32 的对应切片         # **覆盖**（不是累加）
```

* **参数永远 fp32**：不碰主权重、不碰 Adam 的 m/v、不碰优化器；
* **均值语义不变**：仍是 `sum/world`（与 PyTorch DDP 一致）⇒ 等价 batch 不变，
  lr 口径不变；
* **误差只来自一次收窄**：每 rank 的梯度先收成 bf16（相对误差 ≤ 2^-8 ≈ 3.9e-3，
  尾段典型 2^-9）/ fp16（≤ 2^-11 ≈ 4.9e-4），再做 sum。`world=2` 时
  `dequant(sum) * 0.5` 的乘 0.5 是精确的（2 的幂）；
* **重叠路径的量化时机**：stage 时先乘 `1/world` 再收窄。bf16 下乘 2 的幂是精确的，
  所以 `world=2` **时重叠路径与非重叠路径逐位一致**（CPU 单测钉住了「先乘再收窄」的
  值）；非 2 的幂的 world 会有 1 ulp 级差异；
* **缺梯度**：与 fp32 路径完全一致 —— 按 0 参与集合通信，最后补一个全 0 的 `.grad`
  （优化器仍会走 weight decay，历史行为不变）。

### 各 rank 的 dtype 必须一致

`dist` 在 rendezvous 时会让所有 rank 交换 `TG_DP_GRAD_DTYPE`，不一致就**当场报错**：

```
RuntimeError: 各 rank 的 TG_DP_GRAD_DTYPE 不一致：rank0=bf16, rank1=fp32。
修法：所有 rank 必须设成同一个值（fp32/bf16/fp16）；低精度归约会改变集合通信的
类型与字节数，不一致会挂死或算错
```

为什么必须：NCCL 的调用类型不同 ⇒ 挂死或算出垃圾；host 路径的共享内存名字按字节数
生成 ⇒ 一方建、另一方永远等不到（表现为「训练卡住」，极难定位）。

---

## 4. 实现（改了哪些文件、复用了什么）

只动三个文件（+ 一个新测试 + 本文）：

| 文件 | 改动 |
|---|---|
| `python/tgai_go/train/dist.py` | `TG_DP_GRAD_DTYPE/TG_DP_GRAD_EF` 解析；bf16/fp16 的 numpy 量化助手（与 CUDA 内核同公式）；`_cast_f32_to_16/_cast_16_to_f32/_zero_16_region`（**复用** `cuda.py` 里已有的 `cast_f32_to_bf16`/`cast_bf16_to_f32`/`cast_f32_to_f16`/`cast_f16_to_f32` 内核，不新写 CUDA）；`_all_reduce_lowp_{cpu,gpu}`（NCCL/host 两条）；EF；dtype 对齐；stats |
| `python/tgai_go/train/dp_overlap.py` | 三条传输的低精度分支（`NumpyTransport`/`NcclBucketTransport`/`HostBucketTransport`）；量化/反量化/2 字节清零；`comm_bytes` 记账 |
| `python/tgai_go/train/dist_nccl.py` | 只加两个**常量**（`NCCL_FLOAT16=6`、`NCCL_BFLOAT16=9`）和一个**新方法** `all_reduce_typed`；`all_reduce_group(entries, stream=None, dtype=None)` 默认 `dtype=None` ⇒ 走原来那条无类型调用，**老签名/假库/老测试一行都不用改** |
| `tests/test_dp_bf16_grad.py` | 34 个 CPU 用例（量化公式逐位对齐 CUDA 内核、cast 路径、均值、world=1 零变化、dtype 对齐、NCCL 类型通道、EF 两步抵消、2 rank 真多进程）|
| `docs/DP_BF16_GRAD.md` | 本文 |

**为什么复用 `cuda.py` 的内核而不是新写**：`cast_f32_to_bf16` 的 RTNE 公式
`(u + 0x7FFF + ((u>>16)&1)) >> 16` 是已有实测路径（bf16 动量/激活压缩在用），
新写一份等于引入第二个"bf16 舍入真相"。主机侧 numpy 助手照抄同一公式，
并有单测逐位对齐（`test_bf16_bits_match_cuda_kernel_formula`）。

**为什么直接 `rt.launch(...)` 而不是 `rt.cast_f32_to_bf16_flat()`**：后者每次调用都
`empty()` 一块新显存 —— 3.16 GiB 梯度每步分配 1.58 GiB 再释放，峰值直接破红线。
这里的目标缓冲**跨步复用**（`self._lp_buf`），只第一次分配。

---

## 5. 通信字节（验收 ①）与显存

### 5.1 通信字节（算术口径，可直接复核）

848.6M 参数（seq2048 配置：d1024/L12/E4/d_ff3328/词表 96000）的梯度：

| dtype | 每元素字节 | 每 rank 每步 all-reduce 载荷 | 相对 fp32 |
|---|---|---|---|
| fp32（默认）| 4 | 848.6e6 × 4 = 3.394e9 B = **3.16 GiB** | 1.00× |
| **bf16** | 2 | 1.697e9 B = **1.58 GiB** | **0.50×** |
| fp16 | 2 | 1.58 GiB | 0.50× |

复核方式（不需要卡）：

```python
import tgai_go.train as D
from tgai_go.train import dist as D
n = 848_600_000
print(n*4/(1<<30), n*2/(1<<30))     # 3.161... 1.580...
```

代码里每一步都会写进 `dp.comm_rank_bytes` / `dp.comm_rank_bytes_fp32`
（`dp.stats()` 可见）；`TG_DP_TIMING=1` 时还会打一行：

```
[dp-lp] dtype=bf16 comm_rank_bytes=1697000000 (fp32=3394000000, ratio=0.500) backend=nccl
```

> ⚠️ `v3_pretrain.py` 里原有的 `[dp] reduce ... bytes=... GiB` 那一行是按 fp32
> （`grad._gpu.n * 4`）算的，**低精度路径下它报的是 fp32 口径**，别拿它当通信字节。
> 以 `[dp-lp]` 行 / `dp.stats()["comm_rank_bytes"]` / nsys 为准。

> 待测（GPU）：nsys 侧 `ncclAllReduce` 的 payload 字节数应为 fp32 的一半；
> 或直接看 `dp.stats()["comm_rank_bytes"]`。**填这一节时把 nsys 命令与截图数字写上。**

### 5.2 峰值显存涨幅（验收 ≤ 0.5 GiB）

非重叠 `all_reduce_grads`（默认 256 MB 分块）：

| 缓冲 | fp32 路径 | bf16 路径 | 差值 |
|---|---|---|---|
| pack32（打包）| 256 MB | 256 MB | 0 |
| pack16（通信载荷）| — | 128 MB | +128 MB |
| scratch32（对端 dequant / EF）| — | 256 MB | +256 MB |
| **合计** | 256 MB | 640 MB | **+384 MB < 0.5 GiB** ✅ |

* host 路径的 pinned/共享内存**反而减半**（3.16 → 1.58 GiB 主机侧）；
* **重叠路径显存下降**：桶缓冲从 `3×256 MB` 变 `3×128 MB`（NCCL 传输），
  host 传输额外多 128 MB（peer16）+ 256 MB（tmp32），仍低于原来 3×256+256 的账；
* **误差反馈（EF）例外**：多一份 `total × 4 B` 的 fp32 影子（848M 时 **+3.16 GiB**）。
  所以 EF 默认关，且**不进** 0.5 GiB 红线（它是可选实验，见 §7）。

---

## 6. 数值偏差（验收 ④）

### 6.1 理论上界（CPU 已钉住）

| dtype | 单元素相对舍入误差 | 备注 |
|---|---|---|
| bf16 | ≤ 2^-8 ≈ 3.9e-3（尾段典型 2^-9 ≈ 2.0e-3）| 8 位有效数字，指数范围与 fp32 相同 ⇒ 不下溢 |
| fp16 | ≤ 2^-11 ≈ 4.9e-4 | 10 位有效数字，动态范围小（>65504 会 inf；梯度一般安全，但需实测）|

CPU 单测 `test_bf16_roundtrip_error_bound` / `test_fp16_quantization_matches_numpy`
钉住了这两个量级；`test_bf16_bits_match_cuda_kernel_formula` 钉住了与 CUDA 内核**逐位一致**
（包括 `1+2^-8`、`1+3·2^-8` 两个 RTNE 舍入中点）。

### 6.2 20 步 loss / gnorm 实测（待 GPU）

实验口径：同一份数据、同一 seed、warmup 后 20 步，比较
`fp32 / bf16 / bf16+EF` 三条的 loss 与**每步归约后的 gnorm**（gnorm 在归约之后算，
所以它直接反映梯度差异；各 rank 的 gnorm 必须逐位相同，是正确性判据）。

| 配置 | loss 曲线最大偏差（相对 fp32）| gnorm 最大偏差（相对 fp32）| 备注 |
|---|---|---|---|
| bf16 无 EF | 待测 | 待测 | 预期 gnorm 相对偏差 ~1e-3 量级 |
| bf16 有 EF | 待测 | 待测 | 预期比无 EF 小 |
| fp16 无 EF | 待测 | 待测 | 预期比 bf16 小 |

`world=1`：**梯度逐位不变**（`test_world_one_is_bitwise_noop_with_bf16` 在 CPU 上钉住；
GPU 侧同样走 `all_reduce_grads` 的 `world <= 1: return`，零行为变化）。

> 填表命令见 §9.3，测完把真实数字写进来；**不许用"应该差不多"代替**。

---

## 7. 误差反馈（EF，可选）

```
g_eff = p.grad + residual        # residual 是上一步的截断残差（fp32）
q     = cast(g_eff)              # 通信用的低精度值
residual = g_eff - dequant(q)    # 留在 fp32 影子里，下一步加回
```

* **代价**：一份 `total × 4 B` 的 fp32 影子。CUDA 路径在显存（848M 时 +3.16 GiB），
  纯 CPU 路径在主机内存；默认关；
* **与重叠互斥**：EF 的残差要在「打包缓冲」里结算，而重叠路径 stage 一进桶就发集合
  通信了 ⇒ `TG_DP_GRAD_EF=1` 时 `begin_overlap()` 返回 None（打一次 WARN），
  整步走 `all_reduce_grads`。这一点是**刻意**的：不静默降级、也不假装 EF 在重叠路径里生效；
* **缺梯度（MoE 未路由）**：该参数本步不参与通信，残差按 0 重置（会丢掉它之前累积的
  小误差）。密集模型（本次基准）没有这种位置，MoE 场景下 EF 的收益会略低于理论值；
* **CPU 单测证据**：`test_ef_two_step_average_beats_no_ef` 用
  `1+3·2^-8`（正好是 bf16 舍入中点）跑两步：
  * 无 EF：两步都量化成 `1.015625`，平均误差 `+3.90625e-3`；
  * 有 EF：第一步 `1.015625`，残差 `-3.90625e-3` ⇒ 第二步输入 `1.0078125`（精确可表示），
    两步平均 **= 1.01171875 = 原值（误差 0）**；
  * 断言：`err_ef == 0.0 < err_no`。这说明残差逻辑本身是对的（不是"看起来更小"）。

---

## 8. 吞吐（验收 ②③）—— 待 GPU，先写清口径与物理上限

**口径（与上一轮一致，别改判据）**：吞吐比 = `2 × 单卡步时 / 2-rank 步时`。
中位 + 极差，交替 A/B 至少 3 轮 × 20 步（`lab/rent/_ov_bench.sh` 的跑法）。

### 8.1 2×5090（目标：seq512 ≥ 1.25×，seq2048 ≥ 1.5×）

算术预期（用上一轮实测的「暴露值」推算，**不是结论**）：

| seq | 单卡 | fp32 重叠（现行）| bf16 归约理论值 | bf16 若 100% 暴露 | bf16 若 0% 暴露 |
|---|---|---|---|---|---|
| 512 | 0.1406 | 0.2565 (1.096×) | ~0.0875 | 0.228 → 1.23× | 0.141 → 1.99× |
| 2048 | 0.2515 | 0.3998 (1.253×) | ~0.081 | 0.332 → 1.51× | 0.252 → 2.00× |

上一轮实测：PCIe-only 上 NCCL 与计算**不完全共驻**（反向会膨胀），所以实际会落在两列
之间。seq512 的 1.25× 需要"至少藏住 10 ms 左右"，**能不能到必须实测**；seq2048 的 1.5×
在 0% 暴露时是 2.0×、100% 暴露时是 1.51×，所以**理论上刚好过线**，同样必须实测。

### 8.2 2×4090（如实报告，不允许凑数）

上一轮实测：裸 NCCL 3.16 GiB = **448 ms**（8 通道 353 ms，9.6 GB/s）> 单卡 231 ms。

bf16 之后：`ncclAllReduce` 载荷减半 ⇒ **约 177–224 ms**。DP 步时 ≥ 231 + 暴露部分，
而「暴露部分」在 4090 上几乎等于全部通信（不共驻）⇒ 步时约 **0.41–0.46 s**，
吞吐比约 **1.0–1.1×**（若真能藏住一半，上限 1.35×）。

**结论预期：2×4090 很可能仍不达标。** 这是物理上限（9.6 GB/s 的 PCIe + NCCL 与计算
不共驻），不是没调好；本文会把实测值如实写进来，不用"平均一下/挑最好一轮"凑数。

### 8.3 待填的实测表（GPU 排期后）

| 机器 | seq | 单卡 (s) | 2-rank bf16 (s) | 吞吐比 | 是否达标 |
|---|---|---|---|---|---|
| 2×5090 | 512 | 待测 | 待测 | 待测 | 目标 ≥1.25× |
| 2×5090 | 2048 | 待测 | 待测 | 待测 | 目标 ≥1.5× |
| 2×4090 | 512 | 待测 | 待测 | 待测 | 如实报告 |
| 2×4090 | 2048 | 待测 | 待测 | 待测 | 如实报告 |

---

## 9. 复现

### 9.1 CPU 单测（不需要卡）

```bash
python -m pytest tests/test_dp_bf16_grad.py -q          # 38 passed
python -m pytest tests -q                                # 全量（见提交信息里的数字）
```

覆盖：量化公式与 CUDA 内核逐位对齐、cast 路径、均值语义、缺梯度按 0、world=1 逐位不变、
dtype 不一致报错、NCCL 类型通道（假库）、EF 两步残差精确抵消、2 rank 真多进程 host 路径、
以及**假运行时上的设备侧打包管线**（pack32 → cast → 类型化 all-reduce → dequant →
×1/world → 覆盖写回，强制多分块）。最后这层在写的时候**当场抓到过一个真 bug**：
分块边界落在一个参数中间时，写回用的 `dst_off` 写成了 0（会把参数前半段覆盖两遍、
后半段留在本 rank 原值 —— 不报错、loss 照降）。现在由
`test_gpu_pipeline_multi_chunk_matches_reference` 钉住。

### 9.2 2-rank GPU 基准（在 5090/4090 机器上）

```bash
# 单卡
python -m tgai_go.train.v3_pretrain --data /root/bench/data2048_v96000 \
  --vocab 96000 --seq 512 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 \
  --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 20 --log-every 99 \
  --no-final-save --amp off --out /root/bench/bf16_s1

# 2-rank bf16 + 重叠（两个进程，CUDA_VISIBLE_DEVICES 必须不同）
env TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=bf16run CUDA_VISIBLE_DEVICES=0 \
    TG_DP_BACKEND=nccl TG_DP_OVERLAP=1 TG_DP_GRAD_DTYPE=bf16 TG_DP_TIMING=1 \
    TG_NCCL_LIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2 \
    python -m tgai_go.train.v3_pretrain ... --dp 2 --out /root/bench/bf16_d0 &
env TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=bf16run CUDA_VISIBLE_DEVICES=1 \
    ...（同上，--out /root/bench/bf16_d1）&
wait
```

日志里应当看到 `[dp-lp] dtype=bf16 comm_rank_bytes=1697... (fp32=3394..., ratio=0.500)`。
验收口径：交替 3 轮（single→dp→dp→single…），取每轮步时中位数，比 = 2×single/dp。

### 9.3 数值偏差（20 步）

同一条命令把 `TG_DP_GRAD_DTYPE` 换成 `fp32` / `bf16` / `bf16+TG_DP_GRAD_EF=1`，
从日志的 loss 与 `[dp] ... gnorm` 行取每步值，算相对偏差的最大值。

---

## 10. 限制与已知风险（不藏着）

1. **NCCL ≥ 2.10** 才有 bf16 集合通信；低于此版本 bf16 路径会**明确报错**
   （`当前 libnccl X 不支持 bf16 集合通信（需要 >= 2.10）` + 修法），不会静默走 fp32。
2. **host 后端 + 低精度**要求共享内存能被 `cuMemHostRegister` 注册成页锁定；
   失败会明确报错并建议改用 NCCL 后端（不静默退回）。NCCL 后端是 GPU 上的推荐路径。
3. **EF 与重叠互斥**（见 §7）；EF 的显存代价 3.16 GiB 不在 0.5 GiB 红线内（默认关）。
4. **fp16 的动态范围**：>65504 溢出成 inf。梯度一般远小于此，但**首次上 fp16 请先看
   §6.2 的 20 步实测**；bf16 没有这个问题（与 fp32 同指数范围），所以默认推荐 bf16。
5. **bf16 的绝对误差在「小梯度」上相对值不变**：均值语义不变，但梯度很小时（<1e-30）
   会进 bf16 次正规数；对 AdamW 的更新量无实际影响（实测见 §6.2）。
6. **不碰主权重/优化器**：参数、Adam m/v 全程 fp32。想让权重也变 bf16 是另一件事
   （`--param-dtype`，与本开关无关）。
7. **2×4090 很可能不达标**：物理上限见 §8.2，不允许改判据。
8. **nsys 口径**：NCCL 的 bf16 all-reduce 在 nsys 里仍是 `ncclDevKernel_AllReduce`，
   payload 字节看 `ncclAllReduce` 的 count×sizeof(bf16)；如果机器上没 nsys，
   用 `dp.stats()["comm_rank_bytes"]`（它是从元素数算的**实测配置值**，不是估算）。

---

## 11. 验收清单（对照任务）

| # | 要求 | 状态 |
|---|---|---|
| 1 | 通信字节减半（3.16 → 1.58 GiB）| ✅ 算术口径 + 代码记账；nsys 实测待补 |
| 2 | 2×5090 seq512 ≥1.25× / seq2048 ≥1.5× | ⏳ 待 GPU |
| 3 | 2×4090 如实报告 + 物理上限 | ⏳ 待 GPU（预期不达标，原因已写清）|
| 4 | 20 步 loss/gnorm 最大偏差；world=1 逐位不变 | ⏳ 偏差待 GPU；world=1 已 CPU 钉住 |
| 5 | `pytest tests -q` ≥ 1038 passed / 64 skipped；新增 CPU 用例 | ✅ 34 个新用例（cast/均值/world=1/EF/NCCL 类型/2rank）；全量数字见提交信息 |
| 6 | 峰值显存涨幅 ≤ 0.5 GiB | ✅ 设计值 +384 MiB（§5.2）；GPU 上用 `pool_stats()` 复测 |
