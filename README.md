# TGAI ORV — Origin Requiem Vehicle（原生之初）

**用纯 Python 跑深度学习训练，不需要 PyTorch。**

自动微分、张量算子、优化器、训练循环都是自带的；CUDA 后端直接调 **驱动 + cuBLAS + NVRTC**，
所以 `pip install` 之后就能用，**不用装 CUDA Toolkit、不用编译任何 C++ 扩展**。

同一份代码也支持 CPU 与 Vulkan 后端（本次实测以 CUDA / V100 为准）。

---

## 安装

```bash
pip install tgai-orv
```

- Windows / Linux，Python 3.9+
- 有 NVIDIA 卡时：驱动需支持 CUDA 12（运行时由驱动提供，**不需要 Toolkit**）
- 纯 CPU 也能跑，只是慢

装完自检：

```python
import tgai_go.train as T
print(T.__version__)
print(T.amp_state())      # 看 CUDA 是否可用
```

---

## 30 秒上手

完整可运行脚本见 `examples/beginner_pipeline.py`（**不依赖 GPU**）：

```python
import tgai_go.train as T

# ① 导入语料（纯文本或 .jsonl）
text = T.load_corpus("my_corpus.txt")

# ② 在自有语料上训练分词表（不需要外部 tokenizer）
tok = T.train_tokenizer(text, vocab_size=4000)
ids = tok.encode(text)

# ③ 切块并建模型
data  = T.tokens(ids, seq=128)
model = T.preset.transformer(vocab=tok.vocab_size, d_model=256, n_layers=6, n_heads=8)

# ④ 预训练
hist = T.fit(model, data, steps=2000, lr=3e-4)
print(hist["loss"][0], "->", hist["loss"][-1])
```

---

## 执行档位：显存与速度的取舍

ORV 把「用多少显存换多少速度」做成**四组预设**，算子在运行时读策略决定行为，
所以**同一份模型代码**可以在四档之间切换，不需要改模型。

```python
import tgai_go.train as T

T.set_policy("speed")        # 切档：standard / speed / extreme / memory
print(T.get_policy())        # 看当前这一档的全部参数

T.set_policy_value("attn_tile", 1024)   # 单改某一项
print(T.policy("attn_tile"))            # 读某一项（带默认值：T.policy(k, 默认)）
```

### 四档是什么

| 档位 | 注意力实现 | 梯度检查点 | 定位 |
| --- | --- | --- | --- |
| `standard` | 物化（一次性算完 S×S） | 开 | 目前与 `extreme` 同参 |
| `speed` | 物化 | **关** | 用显存换速度 |
| `extreme` | 物化 | 开 | 目前与 `standard` 相同 |
| `memory` | 分块（flash 风格，不落 S²） | 开 | 用速度换显存 |

### 该选哪一档

| 你的处境 | 选 | 这一档实测是什么样 |
| --- | --- | --- |
| 显存够、想最快 | `speed` | 比 PyTorch 快约 2.1×，代价是约 1.8× 显存 |
| 想两头都不输 PyTorch | `extreme` | 显存约 0.80×、速度约 0.93×，两轴同时赢 |
| 显存吃紧 / 长上下文 | `memory` | 显存约 0.68×，且几乎不随上下文增长 |
| 不确定、先跑通再说 | `standard` | **目前与 `extreme` 是同一组参数**（物化注意力 + 检查点）， 也就是"两头都不输"那一档 |

> `standard` 与 `extreme` 现在参数完全相同。保留两个名字，是为了将来把 `standard` 调成更保守的默认
> （例如自动按可用显存选档），而不动已有用户写的 `extreme`。

> **注意**：`set_policy` 要在**建图之前**调用；中途切换只影响之后新建的算子。
> 显存与速度的完整测量见 [`docs/THREE_TARGETS.md`](docs/THREE_TARGETS.md)。

---

## 实测性能（V100 32GB）

模型：`seq=2048 / 8 层 / d=512 / 8 头 / top_k=2 / 共享专家`，**真实训练步**（前向+反向+AdamW）。
对照：同构 PyTorch = **4,633 MiB / 0.507 s**。

| 档位 | 峰值显存 | 步时 | 相对 PyTorch |
| --- | --- | --- | --- |
| `speed` | 8,329 MiB | **0.24 s** | 速度 **快 2.1×**，显存 1.8× |
| `extreme` | **3,692 MiB** | **0.47 s** | 显存 **0.80×**、速度 **0.93×** |
| `memory` | **3,169 MiB** | 0.81 s | 显存 **0.68×**、速度 1.6× |

### 上下文变长时

| seq | ORV `speed` | PyTorch | ORV `memory` |
| --- | --- | --- | --- |
| 512 | **0.21 s** | 0.286 s | — |
| 2048 | **0.40 s** → 0.24 s* | 0.360 s | 0.93 s → 0.81 s* |
| 8192 | 跑不了（需 ~80 GB） | 1.267 s | **3.30 s / 4,349 MiB** |

\* 两列是修复一处主机往返前后的对比，见 CHANGELOG。

- 短上下文（≤2048）：ORV 速度占优
- 长上下文（≥4096）：PyTorch 的融合注意力更快，但 ORV 的 `memory` 档显存几乎不涨

---

### 长上下文：现在能跑到 64K（0.9.97+）

CSA 的全局分支改成「query 分块 + 反向重算」后，显存从 O(seq²) 降到近似线性：

| seq（d=128 / 2 层小模型） | 改前 | 改后 |
| --- | --- | --- |
| 16384 | 11.97 GiB | **2.99 GiB** |
| 32768 | 跑不了（OOM） | **7.64 GiB** |
| 65536 | 跑不了（OOM） | **20.59 GiB**（真实训练步，loss 9.00 → 8.97）|

默认 `cq=2048`：**seq ≤ 2048 自动走原路径、逐位不变**；要完全回退设 `TG_CSA_ATTN_CQ=0`。
数学：前向与物化路径 **max_abs = 0.000e+00（逐位一致）**；4 层/seq512 的 loss 与基线精确相同。

## 基准测试：ORV vs PyTorch（RTX 3090 24GB 单卡）

> **⚠️ 本节及下文所有"ORV 快 N×"的旧数字（含 `speed`/`extreme` 档位描述里的 2.1×、0.93×）
> 全部处于"待用新协议复测"状态，请勿引用。** 原因：那些数字测于
> ① PyTorch 侧开着梯度检查点、② 只跑 3 步取"稳态"、③ 部分 ORV 数字跑在 vocab=4096
> 的数据上而 PyTorch 是 vocab=96000 —— 三条都会系统性高估 ORV。
> 目前唯一按统一协议（预热后 3 次中位数、同词表、都关检查点、都开 AMP）复测过的
> 是下方 V100 的 MoE 12 层表；复测显示 ORV 在两个 seq 上都落后 1.58× / 1.83×。
> 3090 机器暂不可用，故本节的 3090 数字暂无法复测。

同一形状、同一 seq、**真实训练步**（前向+反向+优化器更新，4 步取稳态）。
ORV 用 `--opt int8gpu`（m8v16，默认档），PyTorch 用 fp32 AdamW。

### MoE（V3 形状：E=4+1 共享专家 / top_k=2 / d_ff=3328）seq512

| 层数 | 参数量 | ORV | PyTorch | ORV 快 | 显存（ORV / PyT）|
| --- | --- | --- | --- | --- | --- |
| 8 | 0.63 B | **0.190 s**（2,692 tok/s） | 0.480 s（1,067 tok/s） | **2.53×** | **8.2** / 11.8 GiB |
| 12 | 0.85 B | **0.265 s**（1,930 tok/s） | 0.623 s（822 tok/s） | **2.35×** | **11.2** / 16.1 GiB |
| 16 | 1.07 B | **0.357 s**（1,435 tok/s） | 0.819 s（625 tok/s） | **2.29×** | **14.2** / 20.4 GiB |
| **24（V3-1.5B）** | 1.50 B | **0.538 s**（951 tok/s） | **OOM** | — | **20.2** / — |

### 公平口径说明（必读，否则容易高估我们）

上表 PyTorch 一栏用的是它脚本的默认设置：**开梯度检查点**（`enable_gradient_checkpointing(True)`）。
检查点是"用重算换显存"，实测代价 **+24~26% 步时**。ORV 没有这个开关（它靠分块注意力
和更紧的激活管理省显存），所以两个口径都要给：

**而且必须跑够步数**：只跑 3 步会把 PyTorch 测慢（它前几步在内核自动调优），
实测 seq512 从 3 步的 0.455 s 变成 20 步的 **0.314 s（−31%）**。
下面是 **20 步稳态、两侧都关检查点、都开 AMP** 的口径：

| **RTX 4090（24 GB）/ 交替 3 轮中位** | ORV | PyTorch | 谁快 | ORV 峰值显存 |
| --- | --- | --- | --- | --- |
| seq512 | **0.2103 s**（0.2083~0.2151）| 0.3100 s（0.3075~0.3124）| **ORV 1.47×** | 11.22 GiB |
| seq1024 | **0.2429 s**（0.2419~0.2454）| 0.3396 s（0.3367~0.3426）| **ORV 1.40×** | — |
| seq1024（两侧同开检查点）| **0.3531 s**（0.3516~0.3570）| 0.4702 s（0.4675~0.4750）| **ORV 1.33×** | — |
| seq2048（两侧同开检查点）| **0.5109 s**（0.5077~0.5149）| 0.5532 s（0.5497~0.5664）| **ORV 1.08×** | 10.75 GiB |

换算成吞吐（tok/s，ORV / PyT）：seq512 **2,435 / 1,652** · seq1024 **4,216 / 3,015** · seq2048 **4,009 / 3,702**。

规模标度（4090 / seq512 / 20 步 / 2 轮中位）—— 倍数不是只在某一个配置上调出来的：

| 层数 | 参数 | ORV | PyTorch | 谁快 |
| --- | --- | --- | --- | --- |
| 3 | 358.6M | **0.0681 s** | 0.1038 s | **1.52×** |
| 6 | 522.0M | **0.1155 s** | 0.1635 s | **1.42×** |
| 12 | 848.6M | **0.2060 s** | 0.3014 s | **1.46×** |

### 优化器六档实测（4090 / seq512 / 848M / 整步 20 步中位 / 两侧 `--amp off`）

| `--opt` | 整步 | 状态显存带来的峰值 | 备注 |
| --- | --- | --- | --- |
| **`int8gpu`（默认）** | **0.2156 s** | 11.22 GiB | int8 分块动量 + 每块 fp32 scale |
| `adafactor` | 0.2175 s | **8.79 GiB** | 因子化 O(n+m)，**最省显存** |
| `fp16mom` | 0.2194 s | 11.96 GiB | fp16 动量 |
| `adamw`（fp32） | 0.2312 s | **15.15 GiB** | 对照基线：比默认慢 7%、多 3.9 GiB |
| `muon` | **1.0198 s** | **10.38 GiB** | 5 轮 Newton-Schulz 是算法成本。**2026-11-21 修掉单线程归约后：21.70 → 1.02 s（21.3×）**；仍比 AdamW 慢 4.43×，但**少占 4.8 GiB**。另实测 NS 的 GEMM 若开 `TG_MUON_NS_GEMM=tf32` 可再快 **1.48×**（1.02→0.69 s），代价是 8 步后 gnorm 漂 **1.76%** ⇒ **默认不开**，只作 opt-in |
| `int8` | 跑挂 | — | 该档在 v3 路径不可用（未深究）|

### 两条必须一起说的结论

* **fp16-AMP 在 4090 上对两个框架都是负优化**：ORV 0.2103→0.2989（+42%）、PyTorch 0.3030→0.3385（+12%）。
  原因是 AMP 每步多 **995 次 `cast_f32_to_f16`** 发射（+25%），而 4090 上 fp16 只比 TF32 快 2×（V100 上是 8×）——
  省下的算力抵不过多付的发射。**这条同时说明：我们不是照抄"开 AMP 就更快"的常识。**
* **1.7B 的"能跑"结论只在 32 GB 上成立**：24 GB 的 4090 上，1.72B（20 层/E=8/d_ff=2816）seq512
  用 `int8gpu` 与 `fp32 adamw` **都 OOM**；V100-32GB 上 ORV+int8gpu 可跑（0.578 s/步）而 PyTorch OOM。
  不要在 24 GB 卡上宣称能训 1.7B。

### 显存：两个口径都要看（4090 / 20 步 / 2 轮取大者）

| 配置 | ORV 分配器 | PyTorch 分配器 | ORV 整进程（smi）| PyTorch 整进程（smi）|
| --- | --- | --- | --- | --- |
| seq512 | **11.22 GiB** | 14.79 GiB | **15.02 GiB** | 16.67 GiB |
| seq1024 | **14.15 GiB** | 17.74 GiB | 19.67 GiB | **19.05 GiB** |
| seq2048（同开检查点）| **10.75 GiB** | 15.57 GiB | 20.11 GiB | **18.90 GiB** |

* **分配器口径 ORV 一直省 3.6~4.8 GiB** —— 来源是 `int8gpu` 的分块动量为 **3.06 B/参数**，
  而 fp32 AdamW 的 `exp_avg + exp_avg_sq` 是 **8 B/参数**。
* ⚠️ **整进程口径在 1024/2048 上 ORV 反而多 0.6~1.2 GiB**：ORV 的池会**缓存**空闲块供下一步复用
  （这是有意的性能取舍，见 `docs/DISPATCH_BOUND.md` 关于池 churn 的实测）。
  看"这个配置能不能跑"要用**分配器口径**；看"卡上还剩多少给别人"要用**整进程口径**。
* 两个口径的差异也提醒：**别用 nvidia-smi 的数字去评判分配器的好坏**。

口径：同数据（vocab 96000）、同 ckpt（该关的关、该开的两侧同开）、**两侧各取自身最优设置**、
batch=1、20 步、**交替 3 轮取中位数并给极差**（区间互不重叠）。
脚本：`lab/rent/bench_4090_final.py`、`bench_4090_matrix.py`、`run_opts_big.sh`。

> **⚠️ 前面的旧表（1.30× / 1.19× / 2.53× 等）已撤回**：那些数字有两个口径错误 ——
> ① PyTorch 侧开着梯度检查点；② seq512 的 ORV 数字跑在 **vocab=4096** 的数据上，
> 而 PyTorch 是 **vocab=96000**（实测仅换词表就差 **2.29×**：0.215 vs 0.493 s）。
> 详见 CHANGELOG 的两次更正。旧表待用新协议复测后再恢复。

结论一句话：**4090 上从 512 到 2048 的四个格子 ORV 全部领先（1.08×~1.47×，区间互不重叠）；V100 的历史口径（都开 AMP）512 快 1.42×、2048 快 1.08×，与 4090 同向。**
* **显存 ORV 仍占优**：seq512 低 24%，seq2048 略高 3%
* seq512 的瓶颈是**词表相关成本**：词表 4096 → 96000 每步多 0.278 s（占该步 56%），
  但已定案：这 0.278 s 来自 1.0.3 里一个 1-block/张量的 clip 内核回归（26 GB/s vs 706 GB/s），已回退
* 已知缺陷：`--opt fp16mom` 早期默认值会发散（1.0.3 已改为默认 bf16）；
  默认的 `int8gpu`(m8v16) 经 20 步对拍与 fp32 参照最大相对偏差 **4.2e-04**，无暗伤
* 2048 时把 `n_q` 从 16 降到 8，可省 **11% 时间 + 11% 显存**（ORV 单侧趋势，
  PyTorch 无对应配置故不作横向对比）

### 稠密（无专家：E=1 / top_k=1 / d_ff=13312，同参数量级）seq512

| 层数 | ORV | PyTorch | ORV 快 | 显存（ORV / PyT）|
| --- | --- | --- | --- | --- |
| 8 | **0.176 s**（2,911 tok/s） | 0.540 s（949 tok/s） | **3.07×** | **9.5** / 16.0 GiB |
| 12 | **0.258 s**（1,981 tok/s） | 0.706 s（725 tok/s） | **2.74×** | **13.1** / 22.3 GiB |
| 16 | **0.315 s**（1,627 tok/s） | **OOM** | — | **16.7** / — |

### 上下文变长时（MoE 12 层）

| seq | ORV | PyTorch | ORV 快 | 显存（ORV / PyT）|
| --- | --- | --- | --- | --- |
| 256 | **0.246 s** | 0.650 s | 2.64× | **10.0** / 16.0 GiB |
| 512 | **0.272 s** | 0.618 s | 2.27× | **11.2** / 16.1 GiB |
| 1024 | **0.334 s** | 0.729 s | 2.18× | **14.2** / 16.0 GiB |
| 2048 | **0.943 s** | 0.969 s | **1.03×** | 22.0 / **18.7 GiB** |

### 结论（如实写）

* **在两边都能跑的配置上，ORV 全部更快**：2.27×–3.07×，显存低 20%–41%
* **ORV 能在同一张 24GB 卡上训练 PyTorch 装不下的模型**：V3-1.5B（24 层 MoE）与 16 层稠密
* **seq 越长优势越小**：3090 上 512→2048 是 2.27×→1.03×；V100 上是 2.29×→1.30×。
  **实测排除了"分块没触发"这个解释**（V100 / 12 层 / seq2048，只改 `TG_CSA_ATTN_CQ`）：

  | cq | 步时 | 显存 |
  | --- | --- | --- |
  | 2048（默认，不触发分块） | **0.868 s** | 21.99 GiB |
  | 512 | 0.959 s | 19.72 GiB |
  | 256 | 0.981 s | 19.72 GiB |

  分块是**拿速度换显存**（反向重算的代价），强制打开反而慢 10%~13%。
  真实原因尚未定位到单一环节：ORV 的 CSA 全局分支计算量为 `T × T/c`（seq 增 4 倍、计算增 16 倍），
  PyTorch 侧走 `F.scaled_dot_product_attention`（CUDA 上派发 FlashAttention / mem-efficient，
  不物化 `T×T`）。要把差距归因到"注意力内核效率"还是"激活读写"，需要按组件拆开计时，**目前还没做**。

## 分布式训练（数据并行）

模型**单卡装不下**、或想要更高聚合吞吐时用它。
ORV 的 DP 是「每卡一份完整模型 + 梯度平均」，进程间走**共享内存**，
**不依赖 NCCL/MPI** —— Linux 和 Windows 都能直接用，装完即跑。

```bash
# 两张卡、两个进程；CUDA_VISIBLE_DEVICES 指定各自的卡（ORV 没有选卡参数）
CUDA_VISIBLE_DEVICES=0 TGAI_DP_WORLD=2 TGAI_DP_RANK=0 TGAI_DP_JOB=run1 \
  python -m tgai_go.train.v3_pretrain --data <语料> ...
CUDA_VISIBLE_DEVICES=1 TGAI_DP_WORLD=2 TGAI_DP_RANK=1 TGAI_DP_JOB=run1 \
  python -m tgai_go.train.v3_pretrain --data <语料> ...
```

* `TGAI_DP_WORLD` 总进程数（≥2 才启用）／`TGAI_DP_RANK` 本进程编号／`TGAI_DP_JOB` 必须一致
* 归约**默认取平均**（与 PyTorch DDP 语义一致）；要让等效 batch 变大，按 world 放大 lr

### 实测：与 PyTorch DDP 同条件对比（RTX 3090 24GB ×2，V3 形状，seq512）

| 层数 | ORV 单卡 | **ORV DP（2 卡）** | PyTorch DDP（2 卡） | ORV 优势 | 显存（ORV / PyT）|
| --- | --- | --- | --- | --- | --- |
| 12 | 0.252 s | **0.885 s** | 0.905 s | **1.02×** | 17.7 / 17.7 GiB |
| 16 | 0.328 s | **1.105 s** | 1.201 s | **1.09×** | **18.2** / 22.8 GiB |

归约耗时（3.16 GiB 梯度，12 层）：**0.57–0.61 s**，有效带宽约 5.2 GiB/s。
对照 PyTorch DDP 的通信开销约 0.33 s —— 这条我们仍**落后**（见下"为什么"）。

**正确性**（比速度更重要）：
* 两个 rank 的 **gnorm 逐位相同**（实测 7.638 / 7.575 / 7.572，两进程一致）
* 差分测试：rank0 梯度全 1.0、rank1 全 2.0 时，归约后**两边都精确等于 1.5**（偏差 `0.000e+00`）

### 为什么归约还不够快（如实说明）

3090 是消费卡，实测 `cuDeviceGetP2PAttribute(可访问) = 0` —— **没有 P2P**，
梯度只能过主机内存。PCIe 实测（页锁定缓冲）：d2h 12.05 GiB/s、h2d 10.54 GiB/s，
所以 3.16 GiB 往返的**物理下界约 0.56 s**。PyTorch 用 NCCL（GPU 侧打包 + 与反向重叠）
才做到 0.33 s。要进一步追上，需要把归约与反向过程**重叠**，这属于后续工作。

### 归约优化过程（同配置 A/B，一步一测）

| 版本 | 归约 | 有效带宽 |
| --- | --- | --- |
| 起点（逐张量 + 可分页内存） | 8.4 s | 0.35 GiB/s |
| 去掉主机冗余拷贝 | 5.2 s | 0.6 |
| 页锁定缓冲 + 一次整块传输 | 2.7 s | 1.2 |
| `np.sum(axis=0)` → 逐段 `np.add` | 2.0 s | 1.6 |
| 共享内存也注册成页锁定 | 1.4 s | 2.3 |
| **求和搬到 GPU 侧**（主机只搬运） | **0.57–0.61 s** | **5.2** |

## 插件系统

算子、后端、损失函数、优化器、分词器、调度器 —— **六条扩展轴共用一个命名空间**，
一个文件就能同时适配 CPU / CUDA / Vulkan：

```python
from tgai_go.train import ops_plugin

@ops_plugin.register("my_op", backends=("cpu", "cuda", "vulkan"))
def my_op(x):
    ...
```

能力不足时**必须报错**而不是静默降级。细节见 `docs/`。

### 插件包：一个文件装下整个环境

`.tgop` 就是一个 zip（改后缀），根目录一份 `manifest.json`，里面可以放 **任意多个插件**。
用户只要**一行**：

```python
import tgai_go.train as T

T.plugin.install("svs-ops-0.2.0.tgop")   # ← 一行，装完整的包
T.svs.stft_mag(wav, 1024, hop_length=256)
```

开发期也可以直接指向目录：`T.plugin.install("examples/bundles/svs-ops")`。

### 看信息、卸载

```python
T.plugin.list()          # 装了什么：名字 / 版本 / 后端 / 算子数 / 命名空间
T.plugin.info("svs-ops") # 单个包的详细信息（元信息 + 它装了哪些算子）
T.plugin.uninstall("svs-ops")   # 卸载：注销算子 + 移除 T.svs
```

元信息是**强制**的：`name`、语义化 `version`、`api_version`、`backends` 缺一不可，
非法值在注册时就报错并给出修法 —— 不是写在文档里的口头约定。

### 插件的反向：三档，总能挑一档

```python
# ① 组合式：用已有算子拼，梯度由 autograd 自动给出（不用写反向）
ops_plugin.register("my_softplus", lambda x: (x.exp() + 1.0).log())

# ② 半自动：前向走原生快路径，反向自动导出 —— 只需要一份"数学等价的可微实现"
ops_plugin.register_auto("my_kernel", forward=native_call, fallback=composed_version)

# ③ 全手写：追求极致性能时给前向 + 解析反向
ops_plugin.register_numpy("my_gelu", gelu_fwd, gelu_bwd)
```

② 是**不想写反向**的人用的：训练时前向仍是原生的，反向把 `fallback` 重算一遍再反传
（所以梯度精确，不是差分近似）；推理时完全不建图。默认还带一致性检查 ——
两份实现数学不一致会直接报错，而不是悄悄算错。

---

### 一条命令装完整个环境

```python
T.plugin.install_all("bundles/")     # 目录下所有插件包一次装完
```

原生算子也**不用写反向**：manifest 里给 `form: "native"` 加一个 `impl.fallback`
（一份用已有算子写的等价实现），ORV 就自动把反向导出 —— 前向仍是你编译好的原生 kernel。
详见 [`docs/PLUGIN_SPEC.md`](docs/PLUGIN_SPEC.md)。

## 从 PyTorch 迁移

四步：换 import → 搬权重 → 改自定义 `forward` → 接训练循环。

```python
# import torch as T
import tgai_go.train as T
```

- 权重可用 `T.from_numpy(...)` / safetensors 直接搬，**不要重新训练**
- `Tensor` 语义与 PyTorch 接近，但**只支持 fp32**
- 不支持的能力会显式报错，不会静默算错

迁移检查清单见 [`docs/`](docs/)。

---

## 项目结构

**主产物是 `python/tgai_go/train/` 这个 Python 包**（`pip install` 装的就是它）；
其余目录是后端源码、构建脚本、文档与开发工具。

```
python/tgai_go/train/     ← 核心：自动微分 + 算子 + 优化器 + 训练循环
  __init__.py              Tensor / autograd / nn / optim / 执行档位（约 24 万字符）
  cuda.py                  CUDA 运行时 + 全部内核（NVRTC 源码字符串，约 35 万字符）
  v3_ops.py / v3_moe.py    V3 模型的算子与 MoE 分派
  v3_flash_op.py           分块注意力（flash 风格）
  nn_ops.py / checkpoints.py / plugin.py / ops_plugin.py
  dist.py / doctor.py      分布式 / 环境自检

src/                       原生后端 C 源码（launcher、matmul、tgai_backend）
cuda/                      .cu 后端（与 NVRTC 运行时并存的另一条路径）
cpu/                       CPU 后端构建
shaders/                   Vulkan compute shader（.comp 与编译产物 .spv）
embedded/ tcc/ third_party/ 嵌入式运行时、TCC 工具链、Vulkan 依赖

tests/                     测试（含 fixtures 与算子对拍）
bench/                     逐模块微基准（一个模块一个文件，CUDA event 计时）
tools/                     仓库维护脚本（发版、探针、远程运维）
examples/                  可直接运行的示例
scripts/                   一键上手脚本
docs/                      设计文档、计划、实测报告
.github/                   CI（单元测试 + 打包自检）
```

各平台的构建脚本在根目录：`build_cuda*.bat`、`build_vulkan*.bat`、`rebuild_cpu.bat`；
发版用 `build_dist.bat` 与 `publish_pypi.bat`。

> 只想用的话不需要碰 `src/` `cuda/` `shaders/`：`pip install` 装到的 Python 包已自带
> NVRTC 与全部内核源码，运行时现编译。

---

## 文档

| 文件 | 内容 |
| --- | --- |
| [`docs/THREE_TARGETS.md`](docs/THREE_TARGETS.md) | 三档目标的完整实测、上下文曲线、每次优化的前后对比 |
| [`docs/ORV_KERNEL_INVENTORY.md`](docs/ORV_KERNEL_INVENTORY.md) | 引擎里已有哪些内核/入口（避免重复造轮子） |
| `docs/OP_COVERAGE.md` | 算子支持矩阵 |
| [`CHANGELOG.md`](CHANGELOG.md) | 逐版本变更 |
| `bench/` | 逐模块微基准（CUDA event 计时），可单独运行 |

---

## 许可

MIT，见 [`LICENSE`](LICENSE)。

## 社区与示例插件

ORV 的插件系统允许第三方**不改核心**就扩展算子、层与优化器（规范见 [PLUGINS.md](PLUGINS.md)）。
目前社区的示例插件：

### Polaris（社区示例插件 · experimental）

自适应正交融合优化器（Muon 系：2D 权重走 Polar-Express 正交化 + 逐行 RMS + Cautious 门，
1D 参数走 AdamW），由社区用户 **yyfdfs** 开发，并作为本项目的**官方示例插件**收录。

- 作者主页：<https://github.com/yyfdfsyyf-source>
- 形态：单文件插件 `polaris.py` + `tgai_go.ops` entry point，`T.plugin.autoload()` 自动发现
- 安装：`pip install tgai-polaris`（独立包，尚未随 `tgai-orv` 一起分发）
- 用法：`import tgai_go.train as T; import polaris; opt = T.optim.Polaris(params, lr=1e-3)`
- 它的真正价值是**插件 API 的参考实现**：注册幂等、`status()` 诊断、entry point 发现、
  CPU/CUDA 双路径，都在这一个文件里演示完整

**数值现状（实测，如实公开）**

| 口径 | AdamW | Polaris |
| --- | --- | --- |
| 同 lr（1e-2）末段 loss | **6.06** | 112.77 |
| 各自最优 lr 末段 loss | **0.296**（lr 1e-2）| 8.376（lr 1e-3）|

其 CUDA 路径与 CPU 路径**并非完全等价**：文档声明可以把「逐行 RMS」近似为「整体 RMS」
（已定位、待修）。因此它**不是**默认优化器的替代品 —— 官方默认仍是 `--opt int8gpu`。
把它放进主线，是为了给社区一个可运行的范例，而不是推荐它替换 AdamW。

想要自己的插件进官方索引，请看 [PLUGINS.md](PLUGINS.md)。