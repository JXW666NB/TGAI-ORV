# fp16 Tensor Core 注意力（V100 / sm_70）—— P0 结果

> 实测日期：2026-11-22。设备 **Tesla V100-PCIE-32GB**（cc 7.0 / 驱动 580.88 /
> ORV 1.0.20 / torch 2.6.0+cu124）。峰值口径与
> `docs/KERNEL_BASELINE_V100_2026-11-22.md` 一致：fp32 **14.0 TFLOPS**、
> fp16 张量核 **112.0 TFLOPS**；FLOPs 公式 `4*B*H*S*S*Dh`（causal 只算下半三角，
> 该公式是**上界口径**，两边同样偏大，便于和旧基线直接对比）。

## 0. 结论（先看这里）

| 项 | 结果 |
| --- | --- |
| **验收目标：fp16 注意力前向 ≥ 30 TFLOPS** | ✅ **达成：32.3~32.8 TFLOPS**（S=2048, B*H=36, Dh=64, causal；三次独立复测 32.78 / 32.71 / 32.26~32.81，最小值 32.85~36.37）。相当于张量核峰值的 **29%**、torch mem-efficient 的 **71%** |
| 相对旧 fp32 融合路径 | **2.85×**（32.8 对 11.46 TFLOPS，同为上界口径） |
| 正确性（fp16 输出 vs fp32 参考，相对误差） | **≤ 3.3e-4**（目标 ≤2e-3）：S=2048 时 2.64e-4；含 S=1/2/17/63/65/100/130 等非 64 倍数形状 |
| fp32 路径是否被影响 | **不受影响**：开关打开时 fp32 输入仍走原路径，与关闭时**逐位相同**（实测 `maxdiff=0.000e+00`） |
| 默认是否开启 | **不开启**。`TG_ATTN_TC=1` 才启用；默认连内核都不编译 |
| 反向 | ❌ **未实现**（进阶目标）。需要梯度时自动**精确回退**到 fp32 融合内核：结果正确（dq 相对误差 2.7e-4，差异只来自梯度写回时的 fp16 舍入），但速度是 fp32 水平 |

## 1. 先纠正两条会误导实现的事实

### ① V100 上**不能**用 `mma.m16n8k8 / m16n8k16`

PTX ISA 明确：`.f16` 类型的 `.m16n8k8`/`.m16n8k16` 形状 **要求 sm_75 或更高**
（Turing 起）。V100 是 sm_70，只有两条张量核路径：

| 指令 | 形状 | sm_70 可用 | 说明 |
| --- | --- | --- | --- |
| `mma.sync.aligned.m16n8k8/k16` | 16x8x8 / 16x8x16 | ❌ | 要求 sm_75+；这就是 FlashAttention 至今不支持 Volta 的原因 |
| `wmma.m16n16k16`（f16） | 16x16x16 | ✅ | 编成 4 条 HMMA.884；本实现用它做 QK^T |
| `mma.sync.aligned.m8n8k4`（f16） | 8x8x4 ×4 份 | ✅ | 一条指令 = 4 个独立 8x8x4 = 2048 FLOP；本实现用它做 PV |

已在真机逐元素验证 fragment 布局（`lab/rent/probe_tc_isa3.py` → `env_tc3.txt`）：
`mma.m8n8k4` 的 A/B/C 布局与 PTX 文档一致，相对误差 8.9e-8；
`wmma` 链路相对误差 1.1e-7。**指令纯吞吐**微基准（`probe_attn_tc_micro.py`）：

```
bench_wmma   (8 条独立 wmma/轮)  93.1 ~ 94.9 TFLOPS
bench_884    (8 条独立 884/轮)   96.7 ~ 100.2 TFLOPS
```

⇒ 指令本身能到峰值的 83~89%，慢的一定是内核的**数据搬运**。

### ② 旧基线里 "ORV fp16 = 11.45 TFLOPS" 量的是 **fp32 缓冲**

`CudaRuntime.h2d(arr, dtype=np.float32)` 的 dtype **默认是 np.float32**，旧探针
`rt.h2d(rng.normal(...).astype(np.float16))` 没传 dtype，于是 fp16 数组被再转成
fp32 上传 —— 量到的其实是"fp32 缓冲走 fp32 内核"。

而真正的 fp16 缓冲进旧 fp32 内核是**读错字节**（内核按 `const float*` 读 fp16 数据），
根本不能算一条可用路径。**结论：本次改造之前，ORV 没有可用的 fp16 注意力路径**，
16% 与 10% 的"fp16 峰值占比"是两个不同错误叠出来的假数字。

## 2. 实现

文件：

| 文件 | 内容 |
| --- | --- |
| `python/tgai_go/train/kernels/tgai_attn_tc.cu`（新增） | 内核本体（单独 NVRTC 模块） |
| `python/tgai_go/train/cuda.py` | 开关、源码定位、编译/加载、`attn_tc_fwd()` 发射、opt-in smem |
| `python/tgai_go/train/__init__.py` | `_sdpa_tc_ok()` 合同判定 + `_sdpa_tc()` 分发 + 精确回退 |

**为什么内核放在单独 `.cu`、单独编一个模块**：主模块 `_device_source()` 会被编到
compute_60/61（P100、GTX1060），那里没有张量核，`wmma/mma` 混进去会让**整个引擎**
在那些卡上编不过。这里按需编译，且 `cc<7` 直接拒绝。

### 算法（FlashAttention 式 online softmax，全 fp32 累加/softmax）

```
网格 (ceil(seq/BQ), B*H)，BQ=128，block 256 = 8 warp，每 warp 负责 16 行 query。
每个 warp:
  QK^T : 4 个 k 块 × 4 个 key 块 = 16 次 wmma.m16n16k16，fp32 累加
         → wmma.store 落 smem（这就是"不物化 P"里的 S）
  softmax: 每线程一行的一半（32 列），warp 内 2 lane 归约；causal 掩码在这一步做
  P(fp16) → smem
  PV   : 64 次 mma.m8n8k4（16 个 k 步 × 4 个 dh 块），fp32 累加
         C 的布局是**文档化的**，所以可以直接按行乘 alpha 做 online rescale
  O 全程 fp32 累加，最后一次除以 l 再舍入成 fp16 写回；同时写 m / lse 备用
```

关键取舍：**PV 用 `mma.m8n8k4` 而不是 `wmma`**，因为 `wmma` 的累加器 fragment 布局
是"unspecified"的，拿不到每个元素属于哪一行 ⇒ 无法按行做 `O *= alpha` 的
online rescale（除非每步把 O 写回 smem 再读出来，实测那样更慢）。`m8n8k4` 的
C 布局在 PTX 文档里写死了（`row = X`、`col = (i&4)+(lane&2)+(i&1)`），可以直接用。

### smem 预算（94,208 B，需要 opt-in 到 96 KB）

| 区域 | 大小 | 说明 |
| --- | --- | --- |
| `smQ` | 16,384 B | BQ=128 行 × 64 dh fp16 |
| `smK` | 10,240 B | 64 key × 80 half（行跨度 80 = padding） |
| `smVT` | 10,240 B | V 转置成 \[dh]\[key]，80 half/行 |
| `smS` | 36,864 B | 8 warp × 16 行 × 72 float |
| `smP` | 20,480 B | 8 warp × 16 行 × 80 half |

**行跨度为什么不是 64**：64 个 half = 128 B = 恰好 32 个 bank ⇒ 同一列的 16 行
全落在同一个 bank，`wmma.load`/B 片段读会撞成 8~16 路冲突。padding 到 80 half
（160 B = 5×32）后降到 2~4 路。S 是 f32，片段 32 B ⇒ 跨度必须是 8 的倍数，取 72。

### 契约与开关

* `TG_ATTN_TC=1`（1/true/yes/on）才启用；默认关。
* 只在 **q/k/v 都是 fp16 存储**、CUDA、causal、`dh==64`、三维同形状、cc≥7、
  且内核能编译+opt-in 成功时才走。任一不满足**安静回退**（fp32 路径逐位不变）。
* `dh!=64`、非 causal、带 `attn_mask`、CPU：全部回退，行为与改造前完全一致。
* 判定 fp16 必须看 `q._gpu.dtype`：`Tensor.dtype` 对 CUDA 张量**恒返回 float32**
  （`Tensor.dtype` 在 `self.data is None` 时直接 `return float32`），用它永远判不出来。

## 3. 正确性（实测）

参考口径：
* 小形状 → 主机 numpy **float64** 参考（同一份 fp16 输入 cast 到 fp64 计算），
  隔离"内核算错"和"fp16 舍入"；
* 大形状 → GPU 上 fp32 物化路径（同一份输入 cast 到 fp32）。

| 形状 | 相对误差 |
| --- | --- |
| BH=1 S=1 | 0.000e+00 |
| BH=1 S=2 | 2.49e-04 |
| BH=2 S=17 | 3.31e-04 |
| BH=3 S=63 | 2.41e-04 |
| BH=2 S=64 / S=65 | 1.83e-04 / 2.41e-04 |
| BH=2 S=100 / S=130 | 2.24e-04 / 2.01e-04 |
| BH=4 S=256 | 1.68e-04 |
| BH=4 S=512（vs GPU fp32 物化） | 2.44e-04 |
| BH=8 S=1024 | 2.82e-04 |
| **BH=36 S=2048** | **2.64e-04** |

目标 ≤2e-3，实测最好 1.7e-4、最差 3.3e-4 ⇒ 有 6 倍余量。
（fp16 尾数 10~11 位，理论上限约 5e-4；实测在预期内。）

其它验证：
* fp32 输入 + `TG_ATTN_TC=1`：与 `TG_ATTN_TC=0` **逐位相同**（S=256/1024，maxdiff=0）。
* 需要梯度时走回退：`dq` 与纯 fp32 路径相对误差 2.7e-4（差异只是梯度写回 fp16 的舍入）。

## 4. 性能（实测，median of 30，5 次 warmup）

| 配置（causal，B*H=36，Dh=64） | ORV fp16 TC（本改造） | ORV fp32 融合（基准） | torch fp16 mem-eff |
| --- | ---: | ---: | ---: |
| **S=2048** | **32.78 TFLOPS（29% TC 峰值）** | 11.43~12.69 TFLOPS | 46.14（41%） |
| S=1024 | 27.4 TFLOPS | 9.0~9.9 | — |
| S=512 | 16.3 TFLOPS | 5.8~6.7 | — |
| 需要梯度（回退 fp32） | 10.9 TFLOPS | — | — |

* 两次独立复测 S=2048：**32.78 / 32.71 TFLOPS**，最小值 32.93 / 36.37 ⇒ 稳定在 30 以上。
* 与基线文档对比：旧基线里 ORV fp32 lowmem 是 **11.46**（本次复测 11.43~12.69，V100 时钟有波动），
  torch fp16 mem-eff 是 **45.64**（本次 46.14，与基线一致）⇒ 口径可比。
* 注意 `4*B*H*S*S*Dh` 是**上界口径**：causal 真实计算量约为其一半，
  所以 32.7 TFLOPS 对应**真实约 16.4 TFLOPS = 张量核峰值 15%**。跟 torch 的 46.1（真实 23%）
  比，我们还差 1.4×。**报告里所有 TFLOPS 都按同一上界公式给，不要混用。**

### 优化过程（三步，都是实测）

| 版本 | 改动 | S=2048 TFLOPS |
| --- | --- | --- |
| v1 | BQ=64 / 4 warp / 无 padding，48 KB smem（2 block/SM） | 17.30 |
| v2 | smem 行跨度加 padding（K/VT/P 80 half、S 72 float）消 bank 冲突，57 KB | 24.61 |
| v3 | BQ 64→128、4→8 warp（8 warp/SM），94 KB smem（1 block/SM） | **32.78** |

v1→v2 的 +42% 来自消 bank 冲突（用 `CU_FUNC_ATTRIBUTE_NUM_REGS=148`、
`LOCAL_SIZE_BYTES=0`、`occupancy=2 block/SM` 排除寄存器溢出/占用率问题后定位的）。
v2→v3 的 +33% 来自每 SM 的 warp 数 4→8（1 块/SM 下延迟更容易被盖住）。

## 5. 失败过 / 踩过的坑（照实记）

1. **以为 V100 能用 m16n8k8**（任务书也是这么写的）。PTX ISA 一查：f16 的
   m16n8k8/m16n8k16 要 sm_75+。第一版设计全部作废，改用 wmma + m8n8k4。
2. **NVRTC 不认直接写的 PTX 语句**：`wmma.load...;` 必须包在 `asm volatile(...)` 里。
   第一次编译报 `identifier "wmma" is undefined` 就是这个。
3. **`wmma.load/store` 带 `.shared` 状态空间时操作数必须是共享窗口地址**
   （`cvta.to.shared`），直接传 C 指针会 illegal memory access。改成
   **通用寻址**（不写 `.ss`）即可，数值精确、性能没差。
4. **`rt.permute()` 硬编码 `dtype=np.float32` 输出**：fp16 缓冲过 permute 会按
   fp32 元素重排成垃圾。所以 TC 内核直接吃原生 `[B*H, S, Dh]` 布局（把 head/token
   步长作为参数传进去），**不做 permute**（顺带省掉一次 1.5 GB 级拷贝）。
   → **遗留风险**：`_sdpa_lowmem` 这条老路径对 fp16 输入仍然是错的（读错字节）。
   本次没改它（怕碰 fp32 既有路径），TC 路径绕开了。
5. **`rt.h2d()` 默认 dtype=np.float32**：见 §1②。真 fp16 缓冲必须显式传
   `dtype=np.float16`。旧基线的 fp16 数字全部作废。
6. **bank 冲突**：见 §4。行跨度 = 64 half 时 wmma load 掉到 17 TFLOPS。
7. **回退路径一开始写成"直接 cast fp32 就丢给 attn_nop_fwd_stats"**，忘了
   `attn_nop_*` 要的是 `[S, nh, Dh]` 而不是 `[B*H, S, Dh]` —— 会静默按
   seq=BH 解释、算出垃圾但"不报错"。改成 cast 之后再 permute。
8. **fp16 端到端 autograd 在 ORV 里本来就不通**：逐元素算子全按 fp32 语义，
   `(o*o).sum()` 这类链在 fp16 缓冲上是错的。所以回退反向只能用
   `out.backward(ones)` 这种显式上游梯度来验证。

## 6. 用法

```python
import os
os.environ["TG_ATTN_TC"] = "1"          # 必须显式开（默认关）

import numpy as np
import tgai_go.train as T
from tgai_go.train import functional as F

T.set_default_device("cuda")
rt = T._cuda()
# 真 fp16 缓冲：h2d 必须显式传 dtype，否则会被转成 fp32 上传
q = T.Tensor._from_gpu(rt.h2d(q_np.astype(np.float16), dtype=np.float16))
k = T.Tensor._from_gpu(rt.h2d(k_np.astype(np.float16), dtype=np.float16))
v = T.Tensor._from_gpu(rt.h2d(v_np.astype(np.float16), dtype=np.float16))
o = F.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=True)  # fp16 输出

# 只量 fp16 前向：不要传 requires_grad=True（反向未实现，会回退 fp32）
```

其它入口：
* `T.sdpa_lowmem_ready()` / `rt.ensure_attn_nop()` —— 旧 fp32 内核预热；
* `rt.ensure_attn_tc()` —— 张量核内核预热（计时前调用，避免把 NVRTC 算进 s/step）；
* `rt.attn_tc_supported()` —— 本机能不能用（缓存，只探一次）；
* `rt.attn_tc_fwd(q, k, v, scale)` —— 底层接口，q/k/v 是 fp16 GpuBuf
  `[B*H, seq, 64]`，返回 `(o fp16, m fp32, lse fp32)`；
* `TGAI_ATTN_TC_CU=/path/tgai_attn_tc.cu` —— 开发时覆盖内核源码路径。

## 7. 遗留问题与下一步

| # | 问题 | 影响 | 建议 |
| --- | --- | --- | --- |
| 1 | **反向未实现** | 训练用不上（开了开关也会回退 fp32） | FlashAttention-2 式反向：只存 lse，反向重算 P；QK 继续 wmma，dP/dQ/dK/dV 用 m8n8k4。参考实现里 `attn_bwd_flash`（fp32）已经是这个结构，只是矩阵乘换成 mma |
| 2 | S=2048 是 29% 峰值，只有 torch 的 71% | 端到端收益还不满 | 主要剩余瓶颈是 smem 搬运：wmma 的 A/B 片段在 Volta 上有 **2× 复制**（8 个 `.f16x2` = 16 half/线程 vs 矩阵 8 half），K/V 各读 16 KB/warp/k-tile。可选：① 把 QK 也换成 m8n8k4（手写、可自己定冲突的布局）；② 把 S 的 smem 往返换成已知 Quad 布局的寄存器 softmax（依赖未文档化布局，风险高）；③ 双缓冲 K/V 预取（现在单缓冲，每 k-tile 有一次全局加载停顿） |
| 3 | S=512/1024 只有 16~27 TFLOPS | 短序列不划算 | 按序列长度选路：短序列继续走 fp32 路径（S=512 时 fp16 TC 仍是 fp32 的 2.4 倍，但绝对值低） |
| 4 | 需要 96 KB opt-in smem | 老卡/受限环境不可用 | 已有编译/opt-in 失败即回退逻辑；若要支持 48 KB 卡需降 BQ/BC |
| 5 | `_sdpa_lowmem` 对 fp16 输入仍然是错的 | 万一有人绕过开关直接给 fp16 给 lowmem 路径 | 建议后续在 `_sdpa_lowmem_ok` 里加 fp16 拒绝（回退物化或显式报错），本次未动（避免碰 fp32 路径） |
| 6 | 逐元素算子仍按 fp32 语义 | fp16 端到端训练（不只注意力）不通 | 属于更大的工程（fp16 存储 + fp32 计算的通用算子层），不在 P0 范围 |

## 7.5 复现命令

```powershell
# 同步三个文件到 V100 的 pip 包（本机无 CUDA，只能这么测）
python lab\rent\v100_ssh.py put python\tgai_go\train\kernels\tgai_attn_tc.cu "C:\Python311\Lib\site-packages\tgai_go\train\kernels\tgai_attn_tc.cu"
python lab\rent\v100_ssh.py put python\tgai_go\train\cuda.py "C:\Python311\Lib\site-packages\tgai_go\train\cuda.py"
python lab\rent\v100_ssh.py put python\tgai_go\train\__init__.py "C:\Python311\Lib\site-packages\tgai_go\train\__init__.py"

# 正确性 + 性能
python lab\rent\v100_ssh.py put lab\rent\probe_attn_tc_fwd.py "D:\tgai\probe_attn_tc_fwd.py"
python lab\rent\v100_ssh.py run "C:\Python311\python.exe D:\tgai\probe_attn_tc_fwd.py"
python lab\rent\v100_ssh.py put lab\rent\probe_attn_tc_final.py "D:\tgai\probe_attn_tc_final.py"
python lab\rent\v100_ssh.py run "C:\Python311\python.exe D:\tgai\probe_attn_tc_final.py"
```

探针脚本一览（都在 `lab/rent/`）：

| 脚本 | 用途 |
| --- | --- |
| `probe_tc_isa.py` / `probe_tc_isa3.py` | 确认 sm_70 可用指令、逐元素验证 fragment 布局 |
| `probe_attn_tc_micro.py` | wmma / m8n8k4 纯指令吞吐微基准 |
| `probe_attn_tc_diag.py` | 寄存器 / local memory / occupancy 诊断 |
| `probe_attn_tc_fwd.py` | 正确性对拍（主机 fp64 + GPU fp32 两条参考） |
| `probe_attn_tc_final.py` | 最终性能口径（多次重复取中位数/最小值） |
