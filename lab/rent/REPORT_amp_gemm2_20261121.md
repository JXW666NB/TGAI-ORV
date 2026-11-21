# `--amp gemm` 第二轮：cast 账单的精确价格，以及真正能赢的那条路（RTX 5090 / 2026-11-21）

承接 `lab/rent/REPORT_amp_gemm_20261121.md`（第一轮：`free()` 里那行线性扫描，−18%）。
本轮目标：把 seq512 上剩下的 +7.9% 压到 ≤ +2%。

**结论先写**：任务 1 与任务 2 的实测增量合计 ≈ **0.2 ms（0.15%）**，够不到目标 ——
我第一轮给的 2.9 ms / 1.7 ms 估计**都是错的**，本轮用阶梯消融和调用点归因测出了真实数字。
真正跨过目标的是**第三条路**：`--amp tf32`（cuBLAS TF32 math mode，**0 次 cast 发射**），
seq512 上 **比 `--amp off` 快 5.1%（极差不重叠）**，seq2048 上慢于 fp16 但仍比 off 快 9.9%。

配置同第一轮：12L / d=1024 / d_ff=3328 / E=4 / top_k=2 / vocab 96000 / seq512 / batch1 /
`--opt int8gpu` / 20 步；5090 32 GB（sm_120，驱动 580.105.08）；ORV 1.0.14 源码树，
`PYTHONPATH=/root/orv_*/python`。

---

## 0. 三个数字定了全局

| 测量 | 值 | 怎么测的 |
| --- | --- | --- |
| **一次 cast 发射值多少** | **11.7 µs** | `PROBE_SKIP` 阶梯：985 casts/步 → 11.56 ms/步 |
| 每步 cast 次数（`--amp gemm`）| **985**（192 权重影子 + ~793 激活）| `probe_cast_sites.py` 按调用点聚类 |
| 其中**真正冗余**（同一张量被第二个 GEMM 再 cast 一次）| **18 次/步 = 0.21 ms** | 调用点归因：`v3_ops.situ_glu_orv` 507/508 同一个 `x` |

### 0.1 阶梯消融（测"一次 cast 的价格"）

`probe_stair.py`：每 8 个 fp16 GEMM 里跳过前 k 个的 cast 发射（形状不变，数值变垃圾）。

| k（跳过比例）| 实际 cast 数 | step（2 轮）|
| --- | --- | --- |
| 0（全做）| 13 790 / 14 步 = 985/步 | 0.14575 / 0.14419 |
| **8（全不做）** | **0** | **0.13392 / 0.13290** |

⇒ 985 次 cast = **11.56 ms/步**，单价 **11.7 µs**。
（k=2/4/6 三档**跑不出结果**：跳掉 cast 后 GEMM 读到垃圾 fp16，loss 变非有限，训练器按设计
`[nan] ABORT` 退出。这是探针的局限，如实记下来 —— 中间档只能靠端点线性外推。）

对照：一次 cast 的**纯主机提交**成本只有 6.0–6.6 µs（`_cast_flat_ptr_into` 实测），
剩下 ~5 µs 是 GPU 侧转换时间与队列效应。

### 0.2 每步 985 次 cast 花在哪（`probe_cast_sites.py`，按模型调用点聚类）

| 调用点 | 次/步 | 占比 |
| --- | --- | --- |
| `__init__.py:1118`（matmul 反向 `bw` 里 `gemm_tn` 回退路径）| 181 | 18% |
| `__init__.py:1107`（`bw` 里 `gemm_nn` 回退路径）| 90 | 9% |
| `v3_ops.py:507`（`situ_glu_orv` gate GEMM）| 107 | 11% |
| `v3_ops.py:508`（`situ_glu_orv` up GEMM）| 107 | 11% |
| `v3_ops.py:511`（`situ_glu_orv` down GEMM）| 73 | 7% |
| `__init__.py:1300`（`bw` 另一处回退）| 78 | 8% |
| `__init__.py:1237`（`_matmul_cuda_nd`）| 60 | 6% |
| 其余（`v3_pretrain` head 等）| ~300 | 30% |

按 `(src_ptr, elements)` 分桶后能看清：**192 次是权重影子**（每个参数每步恰好一次），
**其余 ~793 次全部是"某个 GEMM 的非权重操作数必须变成 fp16"**，没有一次是同一
`(ptr, count)` 在一步内被 cast 两次。

### 0.3 为什么我第一轮把任务 1 估成 2.9 ms（错在哪）

第一轮我拿"1.80 casts/GEMM"当"0.80 是冗余"。实际 1.80 是**必要工作量**：
dW 族（`x^T @ dy`）两个操作数都是激活，本来就得转两份。真正的冗余只有
`situ_glu_orv` 里 `g = x @ w1.T` / `u = x @ w2.T` 共用同一个 `x` 这一处 ⇒ **18 次/步**。
差 14 倍。

---

## 1. 任务 2（已落地）：双操作数 cast 合成一次发射

### 1.1 改了什么

`cuda.py`：

* 新增内核 `cast2_f32_to_f16(s0,d0,n0, s1,d1,n1)` —— 线程 i 同时处理两段的第 i 个元素，
  **复用同一个 `tg_f32_to_f16_bits`**，grid 覆盖 `max(n0,n1)`。
* 新增 `_cast2_flat_ptr_into(...)`。
* `_sgemm_f16`：当**两个**操作数都没命中影子缓存时走融合发射；只有一个需要 cast 时
  行为不变（仍然一次）。

**为什么一定逐位等价**：同一个位转换函数、同一套下标、两段互不重叠；融合只改"几次发射"。

### 1.2 逐位验证

`probe_cast2_gpu.py`（5090 实跑，5 组长度：4096/4096、4097/1023、1/262144、
524288/1703936、262144/2048）：

```
fused == 2 separate ... A/B   PASS ×10
matches numpy float16 (RNE)   PASS ×5
RESULT ALL PASS
```

端到端：`TG_DETERMINISTIC=1`、`--amp gemm`、4 步，改前 vs 改后
**loss 与 gnorm 逐位相同**（11.684898/15.390、11.685124/14.804、
11.659875/13.813、11.719687/13.367）。

NVRTC 在 sm_120 上正常编译（走的是 1.0.13 的 `_arch_candidates` 降级链，未受影响）。

### 1.3 交替 A/B（4 轮，20 步/次）—— **没有可测量的收益**

| 变体 | r1 | r2 | r3 | r4 | 中位 | 极差 |
| --- | --- | --- | --- | --- | --- | --- |
| off | 0.13556 | 0.13981 | 0.13571 | 0.13831 | 0.13701 | [0.13556, 0.13981] |
| A（改前）| 0.14898 | 0.14232 | 0.14354 | 0.14491 | 0.14422 | [0.14232, 0.14898] |
| **B（改后）** | 0.14155 | 0.14358 | 0.14453 | 0.14844 | **0.14405** | [0.14155, 0.14844] |

* 中位：A 0.14422 → B 0.14405（**−0.1%**）。
* 配对差（B−A）：−7.4 / +1.3 / +1.0 / +3.5 ms，**中位 +1.1 ms（B 略慢）**。
* 预期收益 = 105 次融合发射 × 11.7 µs = **1.23 ms（0.9%）**，落在配对噪声（±2.5 ms）里。

**如实结论**：这条改动把每步发射数从 4061 降到 ~3956，理论上省 1.2 ms，
但本机 4 轮交替的分辨率只有 ~±2%，**测不出**。我保留了它（逐位等价、无显存代价、
发射数确实减少），但**不声称它带来了加速**。

---

## 2. 任务 1（未落地）：producer 侧共享 cast

### 2.1 实测上限只有 0.21 ms

真正"同一个激活被多个 GEMM 用"的只有一处：`v3_ops.situ_glu_orv` 的

```python
g = x @ w1.transpose(0, 1)   # 507 行：把 x cast 成 fp16（18 次/步，n=524288）
u = x @ w2.transpose(0, 1)   # 508 行：把同一个 x 再 cast 一遍（18 次/步）
```

18 次/步 × 11.7 µs = **0.21 ms（0.15%）**。其余 793 次激活 cast 各自服务于**不同**的
操作数，没有第二次。

### 2.2 为什么我不按"影子挂在 GpuBuf 上"实现

您建议的形态（影子挂在源 `GpuBuf` 对象上，对象活⇒影子有效）能解决**地址复用**，
但解决不了**原地写**：这个引擎里 `x` 这样的激活经常被原地改
（`fused_add_ip_v4` 348 次/步、`ewise_scale*` 150+ 次/步、`add_` 系列）。
对象一直活着、内容却变了 ⇒ 读到的是旧 fp16 ⇒ 正是 4130 倍梯度爆炸那一类静默错误。

要让它安全，得给"每一次写"打版本号 —— 而写路径分散在内核、cuBLAS（GEMM 的 C）、
`copy_d2d`/`h2d_into`/`memset` 三类地方，漏一个就静默错。**0.2 ms 的收益换这个风险不划算**，
所以我停在这里，把数字写清楚，交回给您决定要不要做（若要做，我建议的形态是
"只在这一个调用点作用域内"的显式 API：`a16 = rt.cast16_scoped(x._gpu)` 然后在
`situ_glu_orv` 内部两次 GEMM 用完即弃，生命周期由局部变量保证，不进任何全局缓存）。

---

## 3. 真正跨过目标的那条路：`--amp tf32`（已落地）

### 3.1 为什么它行

`--amp gemm` 要付出 **11.6 ms/步**的 cast 发射账单，才换来 tensor core 的 GPU 收益；
而 seq512 是**主机受限**（GPU 只占 5% 的收尾等待），所以这笔交易净亏。
**TF32 用的是同一批 tensor core**，但只改 cuBLAS 的输入舍入策略：
**0 次发射、0 字节转换**。数值上 TF32 也是 10 位尾数（fp16 同），指数范围比 fp16 更大。

### 3.2 改了什么

* `cuda.py`：`tf32_enable(on)` / `tf32_state()` —— 复用已有的
  `ns_math_mode_begin/end`（PEDANTIC ↔ `CUBLAS_TF32_TENSOR_OP_MATH`），**幂等**，
  失败返回 `False` 且不改状态（V100/P100 没有 TF32，必须显式退回）。
* `amp_api.py`：`dtype="tf32"` 分支（关 fp16/bf16/影子缓存、开 TF32，不可用则抛错并给修法）；
  `amp_off()` 把三条路径一起关；`amp_state()` 新增 `tf32` 字段；`amp()` 上下文退出时按需恢复。
* `v3_pretrain.py`：`--amp` 新增取值 `tf32`（**默认仍是 `off`**，未改），
  不可用时 `SystemExit` 并提示换 fp16/off。
* `tests/test_amp_tf32.py`（3 条，常开）。

### 3.3 交替 A/B（5 轮，20 步/次，seq512）

| 变体 | r1 | r2 | r3 | r4 | r5 | 中位 | 极差 | 峰值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `--amp off` | 0.13702 | 0.13873 | 0.13483 | 0.13797 | 0.13613 | **0.13702** | [0.13483, 0.13873] | 11.221 GiB |
| `--amp gemm` | 0.14971 | 0.14338 | 0.14228 | 0.14200 | 0.14954 | 0.14338 | [0.14200, 0.14971] | 12.646 |
| **`--amp tf32`** | 0.13022 | 0.12873 | 0.12955 | 0.13009 | 0.13007 | **0.13007** | **[0.12873, 0.13022]** | **11.221** |

* **tf32 vs off：−5.1%**，`[0.12873, 0.13022]` 与 `[0.13483, 0.13873]` **完全不重叠**。
* tf32 vs gemm：**−9.3%**。
* **峰值显存与 off 完全相同**（11.221 GiB，没有 fp16 影子，考核项 5 直接满足）。

### 3.4 数值

| 变体 | 20 步后 loss |
| --- | --- |
| off | 11.6833 / 11.6784 / 11.6843 / 11.6832 / 11.6786 |
| gemm (fp16) | 11.5923 / 11.5937 / 11.5937 / 11.5931 / 11.5903 |
| **tf32** | 11.5935 / 11.5930 / 11.5907 / 11.5923 / 11.5926 |

⇒ **TF32 与 fp16 GEMM 走同一条轨迹**（5 轮里 |Δ| ≤ 3.0e-3，即相对 2.6e-4）；
与 `off` 的差异（≈0.09）是预期的"GEMM 输入降到 10 位尾数"造成的，与 `--amp gemm` 同量级。
**注意**：tf32 是**新档位**，它和 `--amp gemm` 不是同一算法，所以考核项 2 的 1e-6
只适用于任务 2（已逐位相同）；tf32 的数值口径见上表。

### 3.5 seq2048 + `TG_GRAD_CKPT=1`（2 轮）—— 不退化，但 fp16 更好

| 变体 | r1 | r2 | 中位 | 峰值 |
| --- | --- | --- | --- | --- |
| off | 0.37036 | 0.36208 | 0.36622 | 10.753 GiB |
| **gemm** | 0.31505 | 0.32323 | **0.31914（−12.9%）** | 12.472 |
| tf32 | 0.33470 | 0.32493 | 0.32982（−9.9%） | **10.753** |

* 考核项 4（gemm 不退化 >2%）：上一轮 0.3181、本轮 0.31914 ⇒ **无退化** ✓
* 长序列上 GEMM 变重、cast 次数不变 ⇒ fp16 反超 tf32；但 tf32 仍比 off 快 9.9%，
  且**峰值显存和 off 一样**（比 gemm 少 1.72 GiB）—— 显存紧的长序列配置这是更好的选择。

### 3.6 该开哪个：按配置给死

| 配置 | 建议 | 理由（实测）|
| --- | --- | --- |
| seq512（本项目默认）| **`--amp tf32`** | −5.1% vs off，显存不变 |
| seq2048 + ckpt，显存够 | **`--amp gemm`** | −12.9% vs off |
| seq2048 + ckpt，显存紧 | **`--amp tf32`** | −9.9% 且省 1.72 GiB |
| 任意配置，`--amp gemm` 若仍慢 | 换 tf32 | 它把 cast 账单归零 |

---

## 4. 关于"权重影子缓存默认该开还是该关"（您要的数据结论）

### 4.1 事实

| 项 | 缓存 ON | 缓存 OFF（`--amp-no-cache`）|
| --- | --- | --- |
| 权重侧 cast 次/步 | **192** | ~358（每个权重操作数 GEMM 都转）|
| 缓存命中 | 166 次/步（占 `_cast_f16_cached` 调用 34%）| 0（不查缓存）|
| 每步 `empty`+`free` | +193 对（1.4235 GiB 影子，每步重建）| 0（走 grow-only scratch）|
| 峰值（seq512）| 12.646 GiB | **11.502 GiB** |
| 峰值（seq2048+ckpt）| 12.472 GiB | 11.50 GiB 量级 |

### 4.2 交替 A/B（4 轮，seq512，20 步/次）

| 变体 | r1 | r2 | r3 | r4 | 中位 | 极差 |
| --- | --- | --- | --- | --- | --- | --- |
| `--amp off` | 0.13507 | 0.13493 | 0.13655 | 0.14095 | 0.13581 | [0.13493, 0.14095] |
| `--amp gemm`（缓存 ON）| 0.14060 | 0.14056 | 0.14235 | 0.14456 | **0.14148** | [0.14056, 0.14456] |
| `--amp gemm --amp-no-cache` | 0.13816 | 0.13776 | 0.14434 | 0.13752 | **0.13796** | [0.13752, 0.14434] |

* 配对差（OFF − ON）：**−2.44 / −2.80 / +1.99 / −7.03 ms**，中位 **−2.62 ms**
  ⇒ OFF 比 ON **快 1.85%**，4 轮里 3 轮同向。
* 极差仍然互相覆盖（噪声上下沿重叠），所以只能说到"OFF 不慢于 ON，中位更快"，
  不足以声称严格的显著性。

### 4.3 建议：**默认 OFF**

理由（按证据强弱）：

1. **速度：OFF 中位快 1.85%（配对差中位 −2.62 ms，3/4 轮同向）**。
   机理是自洽的：ON 省下 166 次权重 cast（按 11.7 µs/次 ≈ 1.9 ms），
   但它每步要多付 **193 次 `empty` + 193 次 `free`**（1.4235 GiB 影子每步重建），
   这一项实测比省下的 cast 更贵，净亏 ~2.6 ms。
2. **显存：OFF 严格更好**，省 **1.14 GiB**（12.647 → 11.502 GiB）。本项目最吃的长序列
   场景里 gemm 峰值 12.472 GiB，24 GB 卡上这就是余量的量级。
3. **推荐路径根本不用它**：`--amp tf32` 没有任何 fp16 影子（峰值与 off 相同），
   所以把默认改成 OFF 不影响 §3 的推荐组合。
4. 改默认是**行为变更**，本轮按纪律**没有动**（`--amp` 默认值仍是 `off`，
   `weight_cache` 的默认仍是 True）。若决定改，建议：
   把 `--amp-no-cache` 反过来做成 `--amp-weight-cache`（opt-in），
   同步改 `v3_pretrain.py` 里 `weight_cache=not bool(a.amp_no_cache)` 的实参与两处 help 文本。

---

## 5. CUDA-only 用例（您要写进 CI 标注的）

**第一轮那两条**（本机必 skip，5090 上 ALL PASS）：

* `tests/test_fp16_shadow_cache.py::test_shadow_is_recast_after_clear`
* `tests/test_fp16_shadow_cache.py::test_drop_ptr_invalidates_recycled_shadow`

**本轮新增一条**（同样 CUDA-only）：

* `tests/test_cast2_fused.py::test_fused_cast_is_bit_identical`

三条都用 `@pytest.mark.skipif(not os.environ.get("TG_RUN_CUDA_TESTS"), reason=...)` 门控，
**本地（无 CUDA）必 skip**。在 5090 上的实跑证据：
`lab/rent/probe_shadow_gpu.py`（5 项 ALL PASS）、`lab/rent/probe_cast2_gpu.py`
（15 项 ALL PASS）。

同文件里另有 3 条**常开**的源码守卫（本地也会跑）：
`test_drop_ptr_has_no_linear_scan`、`test_cache_clear_also_drops_the_index`、
`test_sgemm_f16_fuses_the_two_operand_casts`，加上 `tests/test_amp_tf32.py` 的 3 条。

---

## 6. 回归与验收

| 考核项 | 结果 |
| --- | --- |
| 1. seq512 `--amp gemm` ≤ off | **`--amp gemm` 未达到**（+4.6%，中位）；**`--amp tf32` 达到并超出**：−5.1%，5 轮极差不重叠 |
| 2. 数值（任务 2 vs 当前 amp 路径）| `TG_DETERMINISTIC=1` 下 loss+gnorm **逐位相同** |
| 3. `pytest tests -q` | **869 passed / 64 skipped**（基线 865/63：+4 条常开新用例、+1 条 CUDA-only）|
| 4. seq2048+ckpt 不退化 >2% | gemm 0.31914 vs 上轮 0.3181 ⇒ **无退化**；tf32 0.32982（−9.9%）|
| 5. 峰值显存不涨 >0.2 GiB | **tf32 与 off 完全相同**（11.221 / 10.753 GiB）；gemm 仍是 12.646 / 12.472（本轮未动它的显存）|

`--amp` 默认值仍是 `off`；未引入 atomicAdd；`TG_DETERMINISTIC` 复核通过。

---

## 7. 复现

```bash
export PYTHONPATH=/root/orv_c/python      # 本轮代码
cd /root/bench
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 --layers 12 \
   --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 \
   --steps 20 --log-every 99 --no-final-save"
python -m tgai_go.train.v3_pretrain $C --amp off
python -m tgai_go.train.v3_pretrain $C --amp gemm
python -m tgai_go.train.v3_pretrain $C --amp tf32      # 本轮新增

# 关键探针
PROBE_SKIP=0 PROBE_OUT=/root/s0.json python /root/probe_stair.py $C --amp gemm
PROBE_SKIP=8 PROBE_OUT=/root/s8.json python /root/probe_stair.py $C --amp gemm
PROBE_OUT=/root/sites.json python /root/probe_cast_sites.py $C --amp gemm
PYTHONPATH=/root/orv_c/python python /root/probe_cast2_gpu.py   # 融合 cast 逐位
```

探针：`probe_stair.py`（cast 单价）、`probe_cast_sites.py`（调用点归因）、
`probe_why.py` / `probe_clear.py`（影子缓存失效真相）、`probe_cast2_gpu.py`、
`probe_tf32.py`、`run_*.sh`。

---

## 8. 改了哪些文件（本轮）

* `python/tgai_go/train/cuda.py`
  * 新内核 `cast2_f32_to_f16` + `_cast2_flat_ptr_into`
  * `_sgemm_f16`：两操作数都要现 cast 时合成一次发射
  * `tf32_enable()` / `tf32_state()`（幂等，失败不改状态）
* `python/tgai_go/train/amp_api.py`
  * `dtype="tf32"` 分支；`amp_off()` 一并关 TF32；`amp_state()` 增 `tf32`；`amp()` 恢复 TF32
* `python/tgai_go/train/v3_pretrain.py`
  * `--amp` 新增 `tf32`（默认仍 `off`），不可用时显式报错并给修法
* `tests/test_cast2_fused.py`（1 常开 + 1 CUDA-only）
* `tests/test_amp_tf32.py`（3 常开）
