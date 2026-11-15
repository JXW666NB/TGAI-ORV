# P100：attention 反向内核是唯一的性能短板（实测定位）

> 2026-11-15。本文记录 **实测定量**，不是推断。所有数字来自朋友那台 P100（16 GB）
> 与真机 `bench_p100_full.py`（两边同一份手写模型）。

## 一、结论

**② 速度差 22%（4.27 s vs torch 3.33 s，每步差 0.93 s），其中 0.84 s 来自
`attn_bwd_flash_w` 这一个内核。**

## 二、证据链（逐层剥到最后只剩它）

| # | 测量 | ORV | torch | 结论 |
| --- | --- | --- | --- | --- |
| 1 | 整步（dense B=4，fp32，ckpt 开） | 4.27 s | 3.33 s | 差 0.93 s |
| 2 | 一步内分前向 / 反向 / 优化器 | fwd 756 ms / **bwd 3486 ms** / opt 23 ms | — | 反向占 82%，优化器可忽略 |
| 3 | 纯前向 GEMM `[8192,1152]@[1152,1152]ᵀ` | **3.09 ms** | 3.33 ms | 我们更快 |
| 4 | 反向 GEMM（dX + dW，用 `backward(ones)` 隔离） | **9.05 ms** | 9.00 ms | 打平 |
| 5 | attention 前向 | **11.7 ms** | 36.0 ms | **我们快 3 倍**（融合 vs 物化）|
| 6 | **attention 反向** | **82.2 ms** | 42.1 ms | **我们慢 2 倍** |

第 6 行 × 21 层 = **0.84 s**，占总差距 0.93 s 的 **90%**。

## 三、为什么慢：滑窗是为省 smem 付出的代价

P100 动态 smem 上限 48 KB（且不支持 opt-in）。常驻版 `attn_bwd_flash` 要 68 KB
⇒ 只能退到沿 head_dim 滑窗的 `attn_bwd_flash_w`（40.4 KB，DCHUNK=8）。

滑窗版把 head_dim 拆成 8 个 8 宽的窗口迭代，**每个窗口都要重新载入 K/V 分块并做一遍
QKᵀ/dP 的累加**，重复访存与重复循环是它慢于物化反向的直接原因。

参考：V100 上常驻版比滑窗版**快 34%**（`docs/EXTERNAL_P100_2026-11-15.md` 第 447 行）。

## 四、修法：降 BC，而不是滑窗

常驻版 smem 公式（已用 45,056 / 40,448 两个点反推验证过）：

```
smem = 4 × 64 × BC × 4 B     (smK / smV / smP / smDS 各 64×BC float)
     + 2 × 64 × QSTR × 4 B   (smQ / smDO，QSTR = DH + 8 = 72)
```

| BC | smem | 能否用于 48 KB 卡 |
| ---: | ---: | --- |
| 32（现状）| 69,632 B | 否 |
| 16 | 53,248 B | 否 |
| **11** | **48,128 B** | **是** |
| 8 | 40,448 B | 是（更省但 K 迭代更多）|

⇒ **BC=11 的常驻版刚好塞进 48 KB，且完全不需要滑窗。**

## 五、实现步骤（尚未实施）

1. `cuda.py`：加 `_ATTN_FLASH_NARROW_BC = 11`、`_ATTN_FLASH_NARROW_SMEM = 48128`
2. 新增 `ensure_attn_nop_narrow()`：编译第二份模块，源码用
   `src.replace("#define BC 32", "#define BC 11")`（`_compile` 没有 options 参数，
   用它自带的 `source` 形参即可，不需要给 NVRTC 加 `-D`）
3. `_attn_flash_launch` 里把选择顺序改成：
   `常驻版(69,632) → 窄常驻版(48,128) → 滑窗版(40,448) → 报错`
4. 数值对拍：与滑窗版/物化版比相对误差，要求 ≤ 1e-6；V100 上强制 48 KB 上限跑一遍
   `tests/test_attn_smem_guard.py`（该测试已能强制走小 smem 分支）
5. 全套测试 0 failed 后发版，真机复测第 6 行的数字（目标 ≤ 42 ms）

## 六、风险

- BC=11 不是 2 的幂，`load_KT` 的分块访存可能未对齐产生性能回退（需实测，不能假设）
- 若 BC=11 实测不快，退而求其次试 BC=16 配 96 KB 卡（V100）验证公式与正确性，再决定是否保留

## 七、不要重复的弯路（已实测否掉）

前向/反向 GEMM、优化器 step、全量 logits、梯度残留、池子统计口径、计算图引用环、
checkpoint 分段粒度、cuBLAS math mode、梯度累加就地化 —— **全部不是原因**。
唯一没有量到的是「元素级 kernel 的绝对耗时」，但它最多占剩下 0.09 s，不值得先查。


---

## 八、后续进展与最终数字（2026-11-15 晚）

### 已落地并实测

| 版本 | 改动 | 整步（dense B=4）| 峰值 |
| --- | --- | --- | --- |
| 0.9.69 | lowmem 默认由 False 改为自动（bench 里硬编码关掉了它，一直在测物化路径）| 4.47 -> 4.26 s | 11.33 -> 7.14 GB |
| 0.9.70 | 梯度累加就地化（
t.add_）| 4.26 -> 4.27 s（无实质变化）| 7.14 -> 7.11 GB |
| 0.9.73 | 滑窗宽度 8 -> 16（45,056 B 仍装 48 KB）| 4.26 -> **4.17 s** | 7.11 GB |
| torch fp32 同期 | — | 3.33 s | 7.37 GB |

**③ 显存：达成**（7.11 GB，是 torch 的 0.96 倍，绝对值和比值都更好）。
**② 速度：4.17 s / 1,965 tok/s，为 torch 的 80%**（torch 3.33 s / 2,458 tok/s）。

### 为什么 ② 停在 80%

P100 动态 smem 上限 48 KB（不支持 opt-in），而常驻反向内核按 BC=32 要 69,632 B。
逐条试过：

| 方案 | smem | 结果 |
| --- | ---: | --- |
| 常驻版 BC=32 | 69,632 | 装不下 |
| 窄常驻 BC=11 | 48,128 | **rc=700 越界** —— load_KT 的 32-lane 载入要求 BC>=32 |
| 滑窗 DCHUNK=8 | 40,448 | 82.8 ms |
| 滑窗 DCHUNK=16（现默认）| 45,056 | **77.6 ms** |
| 滑窗 DCHUNK=32 | 54,272 | 装不下（16 是能装下的最大滑窗宽度）|
| torch 物化 | 吃 4.2 GB 显存 | 42.1 ms |

**否定掉的假设（别再走）**：滑窗迭代次数是瓶颈（减半只快 6%）、窄 BC 可行、数学模式、
GEMM（前向我们更快、反向打平）、优化器 step（23 ms）、梯度累加分配。

### 剩下的唯一路径

改 ttn_bwd_flash_w 的访存模式 —— 让 BC 可以小于 32（重写 load_KT 的 32-lane 协作载入），
或者减少每窗口的 K/V 重载。这是**内核代码工作量**，不是常量或参数能解决的；
收益上限约 0.6 s（把 77.6 ms 压到 torch 的 42 ms），风险与工作量都显著高于前面所有改动。

### 结论

lowmem 这条路径在 P100 上已经到顶。torch 在 P100 上用 4.2 GB 显存换 35 ms/层，
我们选择省 4 GB 显存、多付 35 ms/层 —— **这是取舍，不是缺陷**。
16 GB 卡上我们只吃 7.11 GB，显存有余量的用户可以把 lowmem=False 打开换速度
（实测 11.33 GB / 4.47 s，反而更慢 —— 因为我们的物化路径有 3 份 [B,H,S,S]，也不划算）。


---

## 九、内核阶段隔离：79% 的时间在 dK/dV 的原子累加（2026-11-15 深夜）

方法：cuda.py:2558 的 ttn_nop_source_path() 支持 TGAI_ATTN_NOP_CU 覆盖内核源码。
据此生成「在第 N 阶段后 return」的变体，让引擎编译并运行该变体，逐档计时。
实验台：lab/attn_bwd_win/stage_probe.py（未改动仓库内核，也不需要发版）。

| 阶段 | 累计 ms | 增量 ms |
| --- | ---: | ---: |
| 3 载入 K/V + Q/dO | 12.76 | — |
| 4 + QKᵀ 与 dP | 13.30 | **0.54** |
| 5 + dS 写回 smem | 13.37 | 0.07 |
| 6 + dQ 原子累加 | 14.72 | 1.35 |
| 7 完整（含 dK/dV 原子）| **77.6** | **62.9** |

**结论：计算只占 0.5 ms、载入 12.8 ms，剩下 62.9 ms（81%）在 dK/dV 的 atomicAdd。**

### 为什么这么慢

cuda.py 里有 _ATTN_FLASH_GROUPS = 8，注释说「反向 atomic 按 query-tile 分组写到
独立 plane，再加回 dK/dV，减轻 128 路冲突」—— 但**该常量全文只出现一次，从未被使用**。
也就是说分组机制**没有实现**，所有 query tile 的 dK/dV 原子都打在同一个缓冲上，
128 路（本测试是 72 个 tile × warp 内并发）冲突全在。

### 修法（下一步实施）

1. 分配 G 份 dK/dV plane（G = _ATTN_FLASH_GROUPS），query tile 按 	ile_id % G 选 plane
2. 内核末尾用一次归约把 G 份 plane 加起来（G=8 时：2 × 37.7 MB × 7 ≈ 528 MB 读写，
   约 5 ms，远小于 63 ms 的原子开销）
3. 用 TG_ATTN_GROUPS 暴露成可测参数，实测 G=1/4/8/16 选最优

预期：反向从 77.6 ms 降到 20–30 ms，整步再省 0.9–1.1 s —— 足以让 ② 追平甚至反超 torch。
风险：归约本身要占显存（G=8 时约 300 MB），需与 pool_stats 一起核算。


### 实施规格（内核签名已确认，可直接照做）

当前签名（kernels/tgai_attn_nop.cu:733）：

`c
extern C __global__ void __launch_bounds__(256, 1)
attn_bwd_flash_w(const float* dO, const float* Q, const float* K, const float* V,
                 const float* O, const float* m_in, const float* l_in,
                 float* dQ, float* dK, float* dV,
                 int seq, int nh, int dh, float scale);
`

改造步骤：

1. **内核**：加 #define GROUPS 1（可被 TG_ATTN_GROUPS 替换成 2/4/8/16）。
   entry 里算 const int gid = (int)(blockIdx.y);            // query tile 下标
   const int pl = (GROUPS > 1) ? (gid % GROUPS) : 0;
   const long psz = (long)nh * seq * dh;
   loat* dKp = dK + (long)pl * psz;、loat* dVp = dV + (long)pl * psz;
   把原来的 tomicAdd(dK + ..., v) / tomicAdd(dV + ..., v) 改成写 dKp / dVp。
   **注意**：GROUPS=1 时指针偏移为 0，行为与现在完全一致（可作为回归保证）。

2. **启动端**（cuda.py 的 _attn_flash_launch）：GROUPS>1 时
   - 多分配 GROUPS * 2 * psz * 4 字节的临时缓冲（G=8、nh=72、seq=2048、dh=64 时约 302 MB）
   - 把临时缓冲指针当 dK/dV 传进去
   - 启动一个归约内核（或复用现有 dd_）把 G 份 plane 累加到真正的 dK/dV
   - 释放临时缓冲

3. **验收**：先测 TG_ATTN_GROUPS=1 必须与当前 77.6 ms 一致且数值逐位相同；再扫 2/4/8/16。

预算提示：G=8 的归约是 2 × 72×2048×64 × 4 B × 7 ≈ 528 MB 读写，约 5 ms；
相对 62.9 ms 的原子开销，净收益上限约 58 ms/层。

**当前状态：规格已定，实现未做**（需要完整上下文一次改完内核+启动端+归约，并做数值对拍）。


---

## 十、发版流程（0.9.74 事故后新增，必须遵守）

### 事故回顾

0.9.74 给内核加 plane 偏移时，插入锚点匹配到了**常驻版** ttn_bwd_flash（文件里有多个
相似函数），导致 identifier "dK" is undefined ⇒ 整个 	gai_attn_nop.cu 编译失败
⇒ sdpa_lowmem_ready() 变 False ⇒ **所有卡静默回退物化路径**（P100 前向 11.7 -> 30.3 ms）。
本机 pytest 全程 788 passed，**完全没拦住** —— 因为本机无 GPU，pytest 不编译 CUDA。

### 本机为什么没法预检

- 
vcc 存在但缺 cl.exe（MSVC 宿主编译器）⇒ 无法编译
- 
vrtc64_*.dll 未安装（本机只有 cudart）⇒ 无法用 NVRTC 编译

⇒ **编译检查必须在有 GPU 的机器上做。**

### 发版清单（缺一不可）

1. 全套 pytest 0 failed
2. python tools/verify_release.py（版本一致性 + 全套；有 GPU 时还会查内核）
3. **在真机上装好后立刻跑内核探针**：

   `python
   import tgai_go.train as T
   T.set_default_device("cuda")
   print(T.functional.sdpa_lowmem_ready())      # 必须是 True/句柄，不能是 False
   print(T.functional.scaled_dot_product_attention)  # 再跑一次真实形状，看耗时是否回到正常
   `

   判据：P100 上 attention 前向应约 11 ms；若变成 30 ms 左右，就是**回退到物化**了。
4. 只有上面都过，才对外宣布该版本可用

### 改内核时的额外规则

- 文件里有多个结构相似的内核（ttn_bwd_flash / ttn_bwd_flash_w / ttn_bwd_flash_g2），
  **禁止用裸函数体片段当锚点**；必须先用函数名定位，再在该函数范围内找锚点
- 每次只改一处，改完立刻在真机验证 sdpa_lowmem_ready()，不要攒着一起发


---

## 十一、真正的第二个瓶颈：sum_all 只启动一个 block（已定位，未修）

逐算子对拍过程中发现的独立问题（**与 attention 无关**）：

| 环节（[8192,2560]，P100）| ORV | torch | 倍数 |
| --- | ---: | ---: | ---: |
| .sum() **纯前向** | **9.881 ms** | **1.285 ms** | **7.7×** |
| SiLU 前向 | 0.651 ms | 0.358 ms | 1.8×（正常）|
| 残差 add 前向+反向 | 4.36 ms | 0.50 ms | 8.7× |

根因（cuda.py:6483）：

`python
self.launch(reduce_sum, (1, 1, 1), (256, 1, 1), [...])
#                       ↑ grid 只有 1 个 block
`

docstring 写的是「一个 256 线程的 block，M2 的张量都很小」—— 在 21M 元素上这个假设失效：
P100 有 56 个 SM，全归约只用一个 ⇒ 实测 8.5 GB/s，torch 约 65 GB/s，与并行度差距吻合。

### 修法（两遍式，标准做法）

1. 新内核 
educe_sum_partial(src, part, n, nblk)：G 个 block 各归约一段，写入 part[G]
2. 第二遍用现有的 
educe_sum 把 part[G] 归约成标量
3. G 取 min(56 * 4, ceil(n / 4096)) 之类；
 很小时（现在的 M2 场景）G=1，行为不变

**注意**：多 block 会改变求和顺序 ⇒ 与单 block 版本不是逐位相同（归约本身就有序依赖）。
验收要用 llclose（rtol 1e-6），不能用逐位比较；并确认现有测试里没有对 sum 结果做精确相等断言。

### 三次失败假设的记录（避免重复）

1. **SiLU 缺 float4 向量化** —— 加了（0.9.76，逐位一致），耗时 10.32 → 10.19，**无改善**
2. **sum 反向的 D2H 同步** —— 改成设备端填充（0.9.77），10.03 ms，**无改善**，已回滚（0.9.79）
3. **sum 反向才是瓶颈** —— 归因错误：.sum() **纯前向**就已是 9.881 ms

教训：微基准的「合计耗时」不能直接归因给某一侧，**必须分别测前向与反向**再下结论。
