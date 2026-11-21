# ZeRO / 专家并行 / 对外部实现的对照：方案与落地顺序

> 状态标记只用三种：**【已验证】/【已实现未验证】/【未实现】**。
> 数字分两类：**实测**（标明机器与命令）与**估算**（标 `≈`，给出算式）。
> 本文的实测机器 = 2×RTX 5090 31.4 GiB（`p2p=False`、拓扑 `SYS` = 跨 NUMA），
> 归约带宽实测 **24.1 GB/s**（NCCL，3.16 GiB @ 140.78 ms；host 路径 148.03 ms）。

---

## 0. 先看结论：3B 与 7B 到底能不能装下

每参数显存（ORV，fp32 参数 + fp32 梯度 + `--opt int8gpu` 状态 = 3.0625 B/参数）：

| 方案 | 每卡字节/参数（**峰值时刻**） | 2×31.4 GiB 上的实际配置 |
| --- | --- | --- |
| 数据并行（现状） | `4 + 4 + 3.0625 = 11.0625` | 2.55B（28L）实测 OOM（活张量 28.04 GiB）；上限 ≈2.5B |
| ZeRO-1（状态 1/w） | `4 + 4 + 3.0625/2 = 9.53` | 2.55B **算术上可跑**（≈27.9 GiB/卡，**未上卡验证**）；3.0B 仍会 OOM（≈31.9 GiB） |
| ZeRO-2（状态+梯度 1/w；峰值仍在反向末） | `4 + 4 + 3.0625/2 = 9.53` | 同上；反向结束后降到 `4+2+1.53 = 7.53`（续航收益，不改峰值） |
| ZeRO-3（参数+梯度+状态全 1/w + 分层 gather） | `(4+4+3.0625)/2 = 5.53` + 单层全参数 | 上限 **≈5B**（激活按 4~5 GiB 估） |
| **7B** | 需要 `7e9×11.0625 = 77.4 GB` | **2 卡无解**：ZeRO-3 也要 36.1 GiB/卡 + 单层 0.75 GiB + 激活 **≈41.8 GiB/卡** |

> 例：2.55B 在 ZeRO-1/2 下 `2.55e9×9.53 = 24.3 GB = 22.6 GiB`，加 5.3 GiB 激活
> ≈ **27.9 GiB/卡**（算术值；`zero.estimate_memory` 与 `plan.estimate_memory` 逐档对拍，
> 但**未上卡验证**，因为实例在验收前被关机）。3.0B 是 `25.6 GiB + 激活` ≈31.9 GiB，
> **仍然装不下** —— 所以"ZeRO-1/2 能救 3B"要看清 3B 是哪个口径：
> 本仓库 28 层 V3-MoE 是 **2.55B**（预计能救）；字面 3.0B 参数在 2×32 GiB 上还差约 1 GiB。

⇒ **7B 在 2×32 GiB 上装不下，ZeRO-3 也装不下**。要 7B 必须满足其一：
① 3×32 GiB（ZeRO-3，≈29 GiB/卡）；② 2 卡 + CPU offload 优化器状态（ORV 未实现）；
③ 2 卡 + 专家并行（只对 MoE 有效，见 §2，仍差约 8 GiB）。
下面 §1 仍按要求算 7B 的 all_gather 时间 —— 那是**假设装得下**时的通信下界，
用来判断"就算有 3 张卡，值不值得"。

---

## 1. ZeRO-3（参数分片）：方案、与 autograd 的冲突、现实上限

### 1.1 方案（DeepSpeed ZeRO-3 的最小可用版）

参数按「层」为单位切给各 rank（**不要按张量内切**，理由见 `zero.py` 模块文档：
`int8gpu` 的量化块按张量分，切开就无法逐位一致）：

```
每层 forward 前：
    full = all_gather(本层所有参数的 1/w 分片)     # 设备→设备，w×layer_bytes
    把 full 切成各参数的视图，临时替换每个 leaf 的 _gpu 指针
    跑本层 forward（用 full）
每层 forward 后：
    保留 full（backward 要用）——或释放并在 backward 重 gather（省显存、多一次通信）
每层 backward 后：
    释放 full；本层梯度就地 reduce_scatter 到 owner，释放非本 rank 梯度
优化器 step：只更新 1/w 参数；下一层/下一步 forward 前重新 gather
```

显存账（w=2，7B 稠密，`≈`）：分片参数 13.04 GiB + 分片梯度 13.04 GiB +
int8 状态 9.98 GiB + 单层全参数 **0.75 GiB**（32 层 / 202M 参数/层）+
激活（开梯度检查点 ≈5 GiB）≈ **41.8 GiB/卡**。**32 GiB 装不下。**

### 1.2 与 autograd 叶子语义怎么共存（这是 ZeRO-3 唯一的"硬骨头"）

ORV 的 autograd 把 `Parameter` 当叶子：反向时把梯度**累加进 `p.grad`**，
前向图的节点持有的是 `p._gpu` 这块缓冲的引用。ZeRO-3 要求同一块参数在
「分片形态」与「全量形态」之间来回换，有三条**必须同时成立**的性质：

1. **前向与反向必须用同一份 full**。否则 dW 与 W 不对应（数值错，但不报错）。
   做法：`full` 的生命周期覆盖 forward→backward，用 `keepalive` 钉住缓冲
   （ORV 的池子按 `keepalive` 决定能否复用，与 `v3_pretrain` 里 `out.keepalive` 同一机制）。
2. **叶子替换不能破坏 `.grad` 的落点**。反向是按节点里记录的 `p` 对象写 `.grad`，
   只换 `p._gpu` 指针是安全的（对象身份不变）；**不能**换 `p` 本身（换对象后
   反向会把梯度写到一个没人读的临时对象上 —— 静默丢梯度）。
3. **梯度检查点（`TG_GRAD_CKPT`）会让"每层一次 gather"的次数翻倍**：
   重算的 forward 又要一份 full。当前实现里 `_checkpoint_layer` 会重新建图，
   所以 ZeRO-3 + 检查点必须允许"同一层 gather 两次"，并在反向段结束时释放。

【未实现】。落地时**先做不重叠、不检查点的版本**（正确性优先），
它一定慢（通信 100% 暴露），但能先把"逐位一致"钉住。

### 1.3 7B 的 all_gather 时间（假设装得下；按实测 24.1 GB/s 单向）

参数量 7e9，fp32 参数 `N = 28 GB`；w=2 时每个 gather 每 rank：

| 项 | 字节 | 耗时 @24.1 GB/s |
| --- | --- | --- |
| forward gather（每层一次，合计） | `N*(w-1)/w = 14 GB` | **0.58 s** |
| backward re-gather（不保留 full 时） | `14 GB` | **0.58 s** |
| 梯度 reduce_scatter | `14 GB` | **0.58 s** |
| **合计 / 步** | 42 GB | **≈1.74 s/步** |

保留 full 到反向结束可以省掉 backward re-gather ⇒ **≈1.16 s/步**，代价是每层多占
一层全参数（+0.75 GiB/卡）。对照：7B 稠密在这台机器上单卡算一步 ≈0.9 s
（`≈ 6×7e9×512×32/ 有效算力`，仅量级参考）⇒ **通信会把步时放大 2 倍以上**。

**现实上限（p2p=False / 跨 NUMA）**：

* 这台机器两张 5090 之间 **`p2p=False`**，任何 GPU 间传输都过主机内存 + PCIe；
  实测单向 24.1 GB/s 就是**物理上限**（不是 NCCL 没调好：host 路径 22.9 GB/s，
  两者只差 5%）。
* 跨 NUMA 节点（拓扑 `SYS`）意味着帧要过 UPI/QPI，**延迟**也高：小消息
  （<1 MB）的固定开销约 0.1~0.2 ms/次（DP 实测 321 个张量分组归约没有明显劣化 ⇒
  分组有效）。ZeRO-3 每层一次 gather ⇒ 32 层 ×2（fwd/bwd）= 64 次集合通信/步，
  固定开销 ≈ 6~13 ms/步 —— 相对 1.16 s 的带宽时间可忽略。
* ⇒ **限制是带宽不是延迟**。只要 7B 装不下，讨论"能不能更快"没有意义；
  装得下的规模（≤5.2B）用 ZeRO-3，通信/算力比约 **1.2~1.6 s 通信 : 0.6 s 计算**，
  即 **每步 2~3 倍于单卡计算时间**。这是 p2p=False 消费卡集群的**结构性代价**，
  与实现无关。

---

## 2. 专家并行（MoE EP）：token all-to-all 的字节数与划算条件

### 2.1 方案

V3-MoE 的 65.7% 参数在专家里（3B 实测）。EP 把**专家整组**分给各 rank：
每个 rank 常驻 `E/w` 个专家，token 按路由结果 all-to-all 到目标 rank 的专家，
算完再 all-to-all 回来。**参数/梯度/状态都不需要每步重建**（与 ZeRO-3 不同：
EP 分完就是永久分片，没有 gather）。

每层每个 rank 的 all-to-all 字节数（seq=512、batch=1、top_k=2、d=1024、fp32）：

```
token 数/rank       = batch * seq = 512
路由指派/rank       = 512 * top_k = 1024           （每 token 去 2 个专家）
发往其他 rank 的指派 = 1024 * (w-1)/w = 512        （w=2）
单 token 隐状态       = d * 4B = 4 KiB
forward 发 + 回      = 2 × 512 × 4 KiB = 4 MiB/层
backward（梯度，同量）  = 4 MiB/层
合计                  = 8 MiB/层  ⇒ 28 层 = 224 MiB/步/rank
@24.1 GB/s + 每层 2 次集合通信固定开销 ⇒ ≈9 ms + 28×2×0.1ms ≈ 15 ms/步
```

⇒ **seq512 下 EP 的 all-to-all 只有 ~15 ms/步，比 ZeRO 的通信便宜一个量级**。
而且 EP 的通信量 ∝ token 数（`seq×batch`），不是 ∝ 参数量。

### 2.2 什么配置下才划算

| 条件 | 说明 |
| --- | --- |
| **MoE 且专家参数占比高** | V3-MoE 65.7% 在专家；稠密模型 EP 没有对象 |
| **`seq×batch×top_k` 不太大** | 通信 ∝ token 数；seq2048 时 ≈60 ms/步，仍然便宜，但收益不变（显存是常数受益） |
| **`w <= E/top_k`** | 每个 rank 至少 `top_k` 个专家，否则一次路由要跨多跳/复制；V3 E=8、top_k=2 ⇒ **w≤4** |
| **专家负载均衡** | 路由不均时某个 rank 的专家成瓶颈（all-to-all 等最慢那个）；V3 的 aux loss 已做均衡，但 batch=1/seq512 时实测单层计数抖动可见 |
| **相比 ZeRO-1** | EP 省的是"专家部分"的全部字节（4+4+3.0625 B/参数），ZeRO-1 只省状态；3B 上 EP=2 可把每卡从 33.2 GB 降到 ≈22.6 GB（专家 65.7% 部分减半）——**这才是 3B 在 2 卡上最省的路线**，但实现量远大于 ZeRO-1/2：要做 scatter/gather 语义的路由、专家内 GEMM 的变长批处理、反向的 dtoken 回传 |

**EP 的诚实结论**：通信上完全可行（15 ms/步），**成本全在实现**（路由的变长批处理
+ 反向）。本轮不做；它的优先级排在 ZeRO-3 之前（因为省得多且通信便宜）。

---

## 3. 与 PyTorch FSDP / TP 的对照：能做 / 做不到 / 差距的可测原因

### 3.1 能力对照

| 能力 | PyT FSDP | PyT TP | ORV 本轮 |
| --- | --- | --- | --- |
| 优化器状态分片 | ✅ | ✅（按张量切） | **✅【已验证】** `--zero 1` |
| 梯度分片（reduce_scatter） | ✅ | ✅ | **✅【已验证】** `--zero 2`（**但峰值显存未降**，见 §4 步骤 3） |
| 参数分片（按层 gather） | ✅ | ✅（列/行并行常驻） | **【未实现】** |
| 通信与计算重叠 | ✅（prefetch 下一层 / 反向 hook） | ✅（与 GEMM 流水） | **【未实现】**（DP 重叠由另一任务在做：`dp_overlap.py`） |
| 跨卡 GEMM（TP 的 row/col parallel） | — | ✅ | **【未实现】**（cuBLAS 单卡；没有分块 GEMM） |
| tied embedding 去重分片 | ✅ | ✅（需 vocab 维切） | **✅【已验证】**（`ShardPlan` 按 `id()` 去重 + 单测） |
| 逐位可复现（确定性） | ❌（默认不保证） | ❌ | **✅【已验证】**（`--zero 0/1/2` 的 loss/gnorm 逐位相同） |

### 3.2 差距的**可测原因**（不是"实现没优化好"）

1. **没有反向 hook ⇒ 通信 100% 暴露**。实测（DP，seq512，848M）：
   PyT DDP 的 140 ms 归约只暴露 **7~17 ms**（它边反向边归约），ORV 旧路径暴露
   **145 ms**。同一条物理链路，差的是**重叠**不是带宽。FSDP 同理：它的
   `all_gather` 是 prefetch 进 forward 的，ORV 里没有"下一个算子要什么"的调度器。
   落点：`dp_overlap.py`（另一任务）+ ZeRO-3 需要再把 all_gather 做成流水。
2. **参数是 fp32，激活也是 fp32（AMP 只 cast GEMM 输入）**。PyT FSDP/bf16 训练
   权重/激活减半 ⇒ 同样的 32 GiB 能多放 ~40% 参数。这是**数据精度口径**差异，
   不是分片算法差异。落点：参数/梯度 bf16 存储（未做）。
3. **小消息延迟**：TP 每层 2 次 all_reduce（fp32，seq512 时 512×1024×4≈2 MiB），
   64 次/步 × (0.05~0.2 ms) ≈ 3~13 ms；PyT 用 NCCL + 独立流可以压到 ~1 ms 级。
   差距来自**流/事件管理**（ORV 现在用 legacy NULL 流，NCCL 调用是同步暴露的）。
4. **p2p=False 是硬件事实**：任何方案都过主机。

---

## 4. 落地顺序 + 每步验收判据

严格「一次只改一处，每步都有可复现判据」。**A 不通过不许写"已支持"**。

| 步骤 | 内容 | 验收判据（硬） | 状态 |
| --- | --- | --- | --- |
| **0** | `zero.py` 分片规划 + 打包/写回 + CPU 2 进程对拍 | `pytest tests/test_zero_shard.py tests/test_zero_transport.py` 全过；分布式 gnorm 与不分片参考**逐位**相同 | **✅ 已完成（CPU）** |
| **1** | ZeRO-1 接线（状态 1/w + all_gather 参数） | 1.82B/20L/seq512、2 rank、20 步：`--zero 0/1` 的 loss 与 gnorm **逐位相同**；2.55B 峰值 ≤30 GiB | **【已实现未验证】**（GPU 验收被实例关机打断） |
| **2** | ZeRO-2（reduce_scatter + 分布式 gnorm） | 同上三种模式逐位相同；2.55B/28L/seq512 在 2 rank 跑通且峰值 ≤30 GiB | **【已实现未验证】**（同上） |
| **3** | 反向分桶归约（真正的 ZeRO-2 显存收益） | 峰值比步骤 2 **再降 ≥(w-1)/w×梯度**（2.55B 上 ≈4.5 GiB）；**逐位判据必须重新验**（分桶改变归约顺序 ⇒ 可能不再逐位，届时要么改判据为 ≤1 ulp，要么放弃） | 【未实现】 |
| **4** | NCCL 版 collectives（`ncclAllGather`/`ncclReduceScatter`，ctypes） | 同 payload 下比 host 路径**快 ≤5%**（这台机器实测 NCCL 只快 5%）；逐位与 host 路径相同 | 【未实现】（预期收益 ≤5%，优先级低） |
| **5** | 通信/计算重叠（复用 `dp_overlap` 的 hook 机制） | seq512 下 DP/ZeRO 的 `comm_s` 占步时 100% → ≤25%；逐位不变 | 【未实现】（另一任务在做 DP 版） |
| **6** | ZeRO-3（参数分片） | ① 与 ZeRO-2 在"两边都装得下"的配置上 20 步逐位相同；② 5B 级配置 2 rank 跑通且峰值 ≤30 GiB；③ all_gather 实测带宽 ≥20 GB/s（单层 gather ≥64 MiB 才测得出） | 【未实现】 |
| **7** | 专家并行（EP） | ① 与不分片 MoE 前向/反向对拍 ≤1e-5；② all-to-all 实测 ≤20 ms/步（seq512）；③ `w≤E/top_k` 校验生效 | 【未实现】 |

**顺序的依据**：0→2 是"省状态/梯度"（改动小、立刻把 2.55B 装进 2 卡）；
3 是 ZeRO-2 的"真正兑现"但会破坏逐位（风险最高，放后面单独验）；
4/5 是通信优化（收益 ≤5% / 需要另一个任务的基础设施）；
6 是唯一能再上一个数量级的显存改动，但需要 autograd 层配合；
7 对 MoE 收益最大，但实现量最大。

---

## 5. 本轮 ZeRO-1/2 状态：**代码 + CPU 单测完成；GPU 验收未做**

> ⚠️ 诚实记录：2×RTX 5090 实例在验收**之前**被 AutoDL 关机回收（主机报
> `实例GPU需求数量：2 卡 / 主机GPU空闲数量：0 卡`），克隆/重开机时间不确定。
> 所以 **"3B 在 2×32 GiB 上是否 ≤30 GiB/卡" 与 "loss/gnorm 逐位一致"
> 都还没有实测数字**，本文不把它们写成已验证。代码、接线、验收脚本与判据都已就绪，
> 机器回来按下面命令跑一次即可出结论。

### 5.1 已完成并验证（CPU，无需 GPU）

* `pytest tests/test_zero_shard.py tests/test_zero_transport.py`：**54 passed**。
  其中最关键的两条：
  - `tests/test_zero_transport.py` 真起 **2 个进程**（真 `DataParallel` host 后端 +
    numpy 假运行时）跑完整链路：`reduce_scatter` → 分布式 gnorm → `all_gather`；
    分布式 gnorm 与"不分片参考路径"**逐位相同**，两 rank 的 gnorm 也逐位相同。
    共享内存「注册成页锁定」与「注册失败退回 pinned+copyto」两条路径都覆盖。
  - 分片集合互补且不重叠（含 tied embedding 去重）、`all_gather` 用 numpy 模拟后
    参数与分片前**逐位相同**、`world=1`/`--zero 0` 零行为变化、每卡字节数与
    `plan.estimate_memory` 对拍。
* 干净 HEAD(1.0.15) + 本次改动的全量 pytest：**963 passed / 64 skipped**（基线 914/64）。

### 5.2 算出来的（**不是**实测）：2.55B 在 2×5090 上"装得下但更慢"

用 `plan.py` 的实测标定链路（`MEASURED_LINK_GBPS = 48.2`，收发合计口径）与
`zero.estimate_memory` / `zero.bus_comm_bytes` 算术：

| 项 | ZeRO-1 | ZeRO-2 |
| --- | --- | --- |
| 每卡"参数+梯度+状态"（2.55B，int8gpu） | 22.6 GiB | 22.6 GiB（峰值）/ 17.9 GiB（反向结束后） |
| + 激活（20L 实测反推 ≈3.8 GiB → 28L ≈5.3 GiB） | **≈27.9 GiB/卡** | ≈27.9 GiB/卡 |
| 通信（`plan.estimate_comm` 口径，**收发合计**） | `2N(梯度)+2N(参数)` = **48 GB/步** | 同式 48 GB/步 |
| 通信耗时 @48.2 GB/s | **≈996 ms** | ≈996 ms |
| 单卡计算（实测 1.82B 0.2714 s；28L 按层数线性 ≈0.38 s，`plan` 用 0.310 s 标定） | 0.310 s | 0.310 s |
| **2 卡吞吐比**（`plan.judge`：`w·单卡步时 ÷ (单卡步时 + 暴露通信)`） | **0.62/1.306 = 0.47×（净亏）** | 同量级（净亏） |

⇒ **ZeRO 在本机的价值是"装得下"，不是"更快"**：2.55B 单卡 DP 会 OOM，ZeRO-1/2 是
唯一能把它放进 2×32 GiB 的已实现路线（算术 ≈27.9 GiB/卡，余量约 3.5 GiB），
代价是每步多 ≈1 s 通信、聚合吞吐 **低于单卡**（因为每卡算的还是一份 batch）。
把它写成"加速"是错的，**不要这么写**。

> 口径说明（别混）：
> * `zero.comm_model` = **单向有效数据**：ZeRO-1 = `3N(w-1)/w` = DP 的 1.50×，
>   ZeRO-2 = `2N(w-1)/w` = DP 的 1.00×（这是"算法通信量"，用于回答"比 DP 多多少"）。
> * `zero.bus_comm_bytes` = **host 路径真正过 PCIe 的 d2h+h2d**：ZeRO-1 = 3N，
>   ZeRO-2 = 2.5N，DP = 2N（这是"物理代价"，plan.py 的 48 GB 是它的近似）。
> 两种口径都**不会**得出"分片省通信"。

### 5.3 GPU 回来后怎么跑（脚本已就绪）

远端已经放过（实例重开后需重新上传，见 `lab/rent/`）：

```
# 本地：生成验收 runner（HEAD 版 v3_pretrain + snippet，每处替换断言命中 1 次）
python lab/rent/zero_make_verify.py            # -> lab/rent/_zero_verify.py

# 远端（干净 HEAD overlay + zero.py + runner）：
bash /root/x_zero.sh smoke      # ~40 s：2 rank 2 层 2 步，验证接线/传输
bash /root/x_zero.sh bitwise    # ~3 分钟：1.82B/20L/seq512/int8gpu，zero 0/1/2 各 20 步，
                                #   zero_cmp.py 断言 loss 与 gnorm 逐位相同
bash /root/x_zero.sh big        # ~2 分钟：2.55B/28L 2 rank，zero=1/2 各 4 步，读 peak
```

判据（硬）：① 1.82B 三档的 loss/gnorm **逐位相同**（`zero_cmp.py` 用浮点精确相等）；
② 2.55B 每卡峰值（`RESULT.pool_peak_live_gib` 与 `nvidia_peak_mib`）**≤30 GiB**。
任何一条不满足都按"未通过"记录，不写"已支持"。

---

## 6. 接线 snippet（给 `v3_pretrain.py`；本模块不改那个文件）

`zero.py` 自包含，接线只有 6 处。生成器 `lab/rent/zero_make_verify.py` 就是把这
6 处打进 **HEAD 版** `v3_pretrain.py` 生成验收 runner 的（每处都断言命中 1 次）。

```python
# ① 新开关（加在 --dp 旁边）
g.add_argument("--zero", type=int, default=0, choices=(0, 1, 2), metavar="0|1|2",
               help="0=不分片；1=优化器状态分片；2=状态+梯度分片。需要 world>=2；"
                    "只支持 --opt adamw/fp16mom/int8gpu")

# ② 建优化器之前：建会话，用分片后的参数列表建状态
_zt = None
_zparams = params
if int(getattr(a, "zero", 0) or 0) > 0:
    from tgai_go.train import zero as _Z
    _zt = _Z.ZeroTrainer(params, zero=int(a.zero), opt_name=a.opt)
    _zparams = _zt.shard_params
    print("[zero] " + _zt.describe(), flush=True)

# ③ 六个优化器构造的 params → _zparams（AdamW/Muon/Int8AdamWGpu/Fp16MomAdamW/
#    Adafactor/Int8AdamW）

# ④ 存档与分片状态暂不兼容：明确拒绝，不许静默
if _zt is not None and (int(a.ckpt_every) > 0 or a.resume):
    raise SystemExit("--zero 与 --ckpt-every/--resume 暂不兼容（状态只存 1/w）"
                     "。修法：验证阶段去掉存档；要存分片 checkpoint 需另行实现")

# ⑤ 调度器建好之后包外壳（zero_grad 清全部参数 + lr 透传）
if _zt is not None:
    opt = _zt.wrap(opt)

# ⑥ DataParallel 建好、broadcast 之后 attach
if _zt is not None:
    _zt.attach(_dp)          # world/rank 与 TGAI_DP_* 不一致会当场报
```

步内三处（顺序不能换：**归约 → 裁剪 → step → all_gather**）：

```python
loss.backward()
if _zt is not None:
    _comm_s = _zt.reduce_grads()          # 必须在裁剪之前（各 rank gnorm 才一致）
...
if float(a.grad_clip) > 0:
    if _zt is not None:
        gnorm, gscale = _zt.grad_norm_scale(float(a.grad_clip), defer=_defer_clip)
        gnorm = float(gnorm)
        gscale = float(gscale) if _defer_clip else 1.0
    elif _defer_clip: ...（原样）
...
opt.step(grad_scale=gscale) if _defer_clip else opt.step()
if _zt is not None:
    _zt.after_step()                      # all_gather 参数（在 AMP 失效**之前**）
# 之后才是 T.amp_invalidate() / rt.f16_cache_clear()（原样不动）
```

收尾：`_zt.close()` 放在 `_dp.close()` 之前（先关 ZeRO 的共享内存槽，再关 DP）。
**world=1 / `--zero 0` 时全部是空操作**（`shard_params is params`、`wrap` 返回原对象）。

