## [0.9.6] - 2026-11-09

阶段 C：卷积算子（CPU）。

### 新增

- `T.conv2d(x, weight, bias, stride, padding, dilation, groups)` / `T.conv1d(...)`，含反向
  （对输入、权重、偏置都正确）。
- `T.Conv2d` / `T.Conv1d` 模块（参数名与 torch 一致：weight/bias）。
- 实现方式：**主机端 im2col + matmul**；`Conv1d` 用可微 reshape 视图委托给 2D 路径。

### 实测（对 torch.nn.functional）

| 配置 | 前向 | dx | dw | db |
| --- | --- | --- | --- | --- |
| stride=1 / 2 | ≤1e-5 | ≤1e-5 | ≤1e-5 | ≤1e-5 |
| padding=1（配合 stride 1/2）| ≤1e-5 | ≤1e-5 | ≤1e-5 | ≤1e-5 |
| dilation=2 | ≤1e-5 | ≤1e-5 | ≤1e-5 | ≤1e-5 |
| groups=2 | ≤1e-5 | ≤1e-5 | ≤1e-5 | ≤1e-5 |
| bias=False | ≤1e-5 | ≤1e-5 | ≤1e-5 | — |
| Conv1d | ≤1e-5 | ≤1e-5 | ≤1e-5 | ≤1e-5 |

### 未完成（如实标注）

- **CUDA 卷积未实现**：im2col 在主机侧构造，GPU 路径需要新的 im2col/col2im 内核。
  当前传 CUDA 张量会抛 `NotImplementedError`（含"修法"），**不会静默算错**；
  测试覆盖了这条报错路径。因此目标 C2 的"GPU 也测"一条**未达成**。
- 性能：主机 im2col 比 cuDNN 慢，大 batch/大 kernel 时差距更明显。
- `groups` 走逐组循环实现（正确但非融合）。
## [0.9.5] - 2026-11-09

阶段三：证明"能用 ORV 训**用户自己的模型**"（不依赖 `.tg`，TGAI 只是样例）。

### 新增

- `examples/custom_model_quickstart.py`：纯 CPU、秒级可跑的自定义 Transformer 训练示例，
  演示 `AdamW + CosineLR(warmup) + Accumulator + clip_grad_norm_` 的组合用法。
- `docs/QUICKSTART_CUSTOM_MODEL.md`：安装、建模型、训练循环、一致性与能力边界。

### 实测（同模型、同初始化、同超参）

```
ORV vs PyTorch 各训练 8 步：逐 step loss 最差相对误差 = 1.44e-07   （判据 1e-4）
```

测试：`tests/test_custom_model_parity.py`。

### 能力边界（写进 QUICKSTART，未隐藏）

- 已支持：autograd、7 种优化器、5 种调度器、梯度累加/裁剪/激活重算、预训练/SFT/DPO/GRPO、
  单卡 32 GB 训 3B 全参 AdamW（torch 同卡 OOM）。
- 未支持：多卡；conv 与 gather 等通用算子；内置 DataLoader；与 `torch.nn.Module` 互操作。
## [0.9.0] - 2026-11-09

阶段二：对齐训练形态（DPO / GRPO）。torch 核心不含这些损失，参考实现按论文公式手写并逐项对照。

### 新增（`tgai_go.train.align`，导出为 `T.align.*` 与顶层函数）

- `logsigmoid(x)`：数值稳定（`−softplus(−x)` 形式）
- `seq_logprob(logits, labels, ignore_index)`：回答段 logπ **之和**，保持可微
- `dpo_pair_loss(pw, pl, rw, rl, beta)` / `dpo_batch_loss(...)`：DPO
- `group_advantages(rewards, eps)`：GRPO 组内归一化优势（总体标准差）
- `token_logprobs(logits, labels, ignore_index)`：**逐 token** logπ（可微）
- `grpo_loss(policy_logps, old_logps, ref_logps, advantages, masks, clip_eps, beta, kl)`：
  token 级 PPO 裁剪 + KL（k1 / k3），返回 (loss, stats)

### 实测对照（手写 torch 参考，独立实现同一公式）

| 项 | 判据 | 实测 |
| --- | --- | --- |
| `logsigmoid`（含 ±40）| 1e-6 | ≤1e-6 |
| `seq_logprob`（含 ignore_index）| 1e-5 | ≤1e-5 |
| DPO 单对 loss / 对两侧 logits 的梯度 | 1e-5 | ≤1e-5 |
| DPO batch 平均（beta ∈ {0.01,0.1,0.5}）| 1e-5 | ≤1e-5 |
| `token_logprobs` | 1e-5 | ≤1e-5 |
| GRPO loss / 梯度（beta=0 / 0.1·k3 / 0.1·k1）| 1e-5 | ≤1e-5 |
| GRPO 裁剪触发统计 | — | ratio 远离 1 时 clip_frac>0 |

### 已知限制（写进 docstring，未隐藏）

后端没有 gather 算子，因此：
- `seq_logprob` 逐序列调用（batch 内多几次 kernel）；
- `token_logprobs` 需要一份 [T, V] 的 one-hot，词表大时显存开销明显。

两处都留了"等 gather 就绪再向量化"的说明，接口不需要变。
## [0.8.0] - 2026-11-09

阶段一：训练基础设施对齐 PyTorch（每项都有与 torch 的数值对照测试）。

### 新增

- **学习率调度器** `T.optim.lr_scheduler`：`ConstantLR` / `LinearLR` / `CosineLR`（含 warmup）/
  `StepLR` / `LambdaLR` / `make_scheduler`。lr 序列与 `torch.optim.lr_scheduler` 逐 step 一致
  （float32 存储精度内，判据 1e-6）。
- **优化器** `T.optim.SGD`（momentum / dampening / nesterov / weight_decay）、`T.optim.Lion`、
  `T.optim.Adafactor`。
- **梯度裁剪**导出到 `T.nn.utils.clip_grad_norm_`（torch 同路径）与 `T.clip_grad_norm_`。
- **梯度累加** `T.optim.Accumulator`（tick/finalize/reset）与 `T.optim.scale_grads`。

### 与 torch 的实测对照

| 项 | 判据 | 实测 |
| --- | --- | --- |
| 调度器 lr 序列 | 1e-6 | 逐 step 一致（float32 存储精度）|
| SGD（5 组配置，4 步）| 1e-6 | ≤1e-6 |
| Lion（含 wd）| 1e-6 | ≤1e-6（对独立参考实现）|
| Adafactor（2D/1D/wd/scale_parameter/clip）| 1e-6 | ≤1e-6（对 `transformers.optimization.Adafactor`）|
| 梯度裁剪（3 组 max_norm）| 1e-6 | 范数与梯度均 ≤1e-6 |
| 梯度累加 4×(3 行) vs 1×12 行 | 1e-6 | **1.30e-06**（fp32 求和顺序，见下）|
| 激活重算 开/关 梯度 | 1e-5 | ≤1e-5 |

梯度累加未达 1e-6（实测 1.30e-06）：数学等价，差异来自 fp32 下 4 次小 GEMM 与 1 次大 GEMM 的
求和顺序。测试判据写明 1e-5 并记录实测值，未把"接近"当作达标。

### 修正的 torch 语义（对照测试抓出来的）

- **SGD 首步不做 dampening**：torch 用原始梯度初始化 momentum buffer，之后才乘 `(1-dampening)`。
- **`CosineAnnealingLR` 在 T_max 之后周期回升**（warm restart），不是单调降到 eta_min。
- **模块参数的 `p.data` 是 memoryview**：`p.data[...] = x` 会抛
  `memoryview: invalid slice key`，新优化器统一改用 `np.copyto(np.asarray(...))`。

### 显存（V2.7 3B，28 层，实测）

- AdamW（fp16 动量）：公式 **11.064 GB**，GPU 实测 **+11.072 GB**
- Adafactor（row/col）：公式 **0.021 GB**，GPU 增量小于 1 MiB（走了已预留的池）
- 约 **527×** 状态显存差距

## [0.7.0] - 2026-11-09

### 修复（与 PyTorch 数值对齐，实测 15 步 loss 相对偏差 1.00e-05）

- **注意力缺因果掩码**：`v27_train.py::fwd` 原为 `softmax((q@k.T)*scale)`，**没有 mask**，
  等价于双向注意力（能看到未来 token）。已加上三角 −1e9 掩码。这是语义级 bug：
  修复前单层 loss 与 PyTorch 差 2.56e-02，修复后 1.35e-07。
- **CUDA MoE 的 softmax 作用域错误**：`_combine_cuda` 在**全部专家**上做 softmax 再取 top-k，
  而语义（CPU 路径 / 引擎 `tgai_cuda.cu:301-307` / PyTorch）是**只在选中的 k 个上** softmax。
  现改用 `moe_topk` 返回的选中值 `val_buf` 直接 softmax。实测 scale=8 时 4.55e-02 → 1.42e-06。
- **训练路径丢失输入 embedding 梯度**（tied head）：训练喂 `x0` 用的是 numpy 常量，
  tied embedding 只拿到"输出投影"一侧梯度（实测差 34%）。新增
  `V27Trainer.add_embed_grad()`，`backward_and_step` 记录 `last_input_grad`，
  用 `rt.emb_bwd` 把输入侧梯度散射回 `head.weight.grad`。
- **`emb_bwd` 的 id dtype**：传 int64 会被内核按 int32 解读，梯度落到序列里不存在的行
  （实测 row 0 拿到 0.62，而该 token 根本不在输入里）。改为 int32 后总梯度误差 9.29e-01 → 5.85e-06。

### 新增

- `tgai_go.train.v27_train.V27Trainer.add_embed_grad(ids, g_x0=None)`：补齐 tied embedding 的输入侧梯度。
- `V27Trainer.last_input_grad`：逐层反向结束后保留 `d(loss)/d(x0)`。

### 对齐证据（可复现）

前向 1/2/4 层 loss 相对误差 1.35e-07 / 6.65e-08 / 0.00e+00；分组梯度 ≤7e-6；
tied-embed 总梯度 5.85e-06；AdamW 单步参数 ≤5.09e-08；逐层重算反向 vs 图反向 ≤3.71e-07；
ORV 自比噪声底（双进程）3.20e-07；真实语料 15 步曲线最差 1.00e-05。
全 28 层前向 loss：ORV 1.3444903 vs PyTorch 1.3444886（相对 1.2e-06）。

> 注意：`attention` 由双向改为因果会**改变已有模型的训练/推理行为**，
> 之前用旧版训练的 checkpoint 语义上等价于另一个模型。故按 minor 版本升级。

## [0.5.0] - 2026-11-03

这一版几乎全是**性能与可复现性**，每条都有实测数字。判据与证据见
`docs/GAP_VS_PYTORCH.md` §10~§22 与 `docs/PRODUCTION_READINESS.md`。

### 新增 · 真 AMP（激活存 fp16）
- `train/amp_mlp.py`：MLP 块内部激活存 fp16，**主权重 fp32 + fp32 累加**（标准 AMP 语义）。
- 数值（块级 vs fp32）：前向 **6.36e-04**、dh **7.56e-04**、dW1/2/3 **8.4e-04 ~ 1.17e-03**；
  端到端 3 步 loss 相对差 **1.54e-05**。
- 步时（V100，0.5B 形状，24 层，6 步）：fp32 527.2 → 仅 fp16 GEMM 354.0 → **AMP 263.5 ms**
  ⇒ 对原现状 **1.34×**、对 fp32 **2.00×**。
- 开关 `cfg["MLP_FP16"]`，**默认关**（fp16 有精度地板：1000 步任务 fp16 停在 1.2e-02、fp32 到 2e-06）。
- 过程中修掉的三个自有 bug 记录在 `docs/PRODUCTION_READINESS.md`。
### 新增 · 存储精度 float32 / float16 / bfloat16（新模块 	rain/dtypes.py）
- save_weights(path, weights, dtype="bfloat16") / load_weights(path)：
  检查点可按 fp16 或 bf16 存储（体积减半），文件内**逐张量记 dtype**，允许混合精度。
- bf16 用 uint16 存高位截断（round-to-nearest-even），**指数范围与 fp32 相同**：
  1e30 / 1e-30 都不溢出（fp16 会）。实测 bf16 相对误差 < 2^-8，且只有 8 位尾数。
- cast_weights() / 	o_storage() / rom_storage() 供训练侧按需 upcast。
- 注意：这是**检查点存储**精度；Tensor 的运行时存储仍是 float32
  （fp16/bf16 只用于 GEMM 内部计算）。真 AMP（激活存 fp16/bf16）仍未做。
- 新增 	ests/test_dtypes.py（7 项，纯 CPU）。

### 新增 · 可复现模式（`TG_DETERMINISTIC=1` 或 `rt.deterministic = True`）
- 三处 `atomicAdd` 全部换成确定性归约：rmsnorm 的 dw（按列升序）、
  词嵌入反向（按 id 独占一行）、`repeat_interleave` 反向（按 j 升序，新内核 `repeat_reduce`）。
- **实测：同一配置两次训练，参数最大绝对差 0.000e+00（逐位一致）**；默认模式 7.63e-06。
- 代价约 1%（717.3 → 724.3 ms/step，V100 / Qwen2.5-0.5B 形状 / B=4·S=512）。
- 与 fp16 加速**可同时开启**：492.9 ms/step 且参数逐位一致。

### 新增 · fp16 Tensor Core GEMM（`rt.fp16_gemm = True`，默认关）
- 零转换入口 `gemm_f16_nt_raw` / `gemm_f16_tn_raw` / `gemm_f16_nn_raw`
  （fp16 输入 + **fp32 输出** + **fp32 累加**），配权重影子与激活 scratch。
- 前向 GEMM 156.3 → **28.4 ms（5.5×）**；整步 754 → **489 ms（1.54×）**。
- 已知代价：**精度地板** —— 1000 步记忆任务 fp32 能到 2e-6，fp16 停在 1.2e-02。
  因此默认关闭，只适合常规训练。

### 新增 · fused AdamW（默认开）
- `adamw_update_multi`：290 个 kernel → **1 个**（设备指针表 + 前缀和 + 二分查找）。
- 隔离对拍 **0.000e+00**（与逐参数版逐位一致）；整步 **1.04×**。

### 新增 · CUDA Graph 捕获 arena
- `reserve_capture_arena()`：捕获期 bump 分配，绕开"捕获中禁止 cuMemAlloc"。
- 捕获成功（2441 节点），重放 **748.5 → 605.5 ms（1.23×）**。

### 修复 · 三处"看起来没问题但很致命"的实现
- `ce_row_stats` 是**非合并访问**（每线程连续分块，步长 V/256，仅 37 GB/s）：
  改跨步后 CE 前向 **67.5 → 3.6 ms（18.7×）**。
- `log_softmax_lastdim` 前向/反向是**线程-per-row 串行扫 k**：改 warp-per-row。
- 分配器两处**硬编码 12.2 GiB** 判据（工作集本就 13 GiB）导致每次未命中都整设备同步：
  改为看真实空闲显存 + 节流，步时 **1.125 → 0.898 s（-20%）**。

### 新增 · 算子
- `rope_half`（half-split RoPE 融合内核，前向/反向各 1 个 kernel）：步时 **-14%**。
  注意：现有 `fused_rope` 是**交错式**，与 Qwen/LLaMA 的 rotate_half 不等价（实测差 2.478）。
- `checkpoint_sequential(..., segments=)`（对齐 PyTorch 同名签名）。
- causal softmax 接线（含 n-D 支持）—— 但**实测更慢**（内核每行单线程串行），
  已改为 opt-in（`TG_SOFTMAX_CAUSAL=1`）。

### 工具
- `tools/profile_kernel.py`：CUDA event 级内核剖析（曾用它发现"只 instrument launch
  会漏掉 cuBLAS"的测量错误）。
- `tools/bench_gemm_fp16.py`：fp16 vs fp32 逐形状对照。
- `tools/tg_check.py`：`.TG` 表结构自检（哈希唯一、区间不重叠、越界、空洞）。

### 测试
- 新增 `tests/test_deterministic.py`（GPU-gated，断言"确定性模式下反向两次差 == 0"）、
  `tests/test_rope_half.py`、`tests/test_ckpt_segments.py` 等。
- 合计 **292 passed / 6 skipped**，`tools/check_package.py` PASS。

### 诚实结论（**2026-11-03 已纠正口径**，见 docs/BENCHMARK_RIGOR.md）
- ⚠ 更正：此前写的「对 PyTorch fp32 快 1.25×」**是错的** —— 我们的基准每步都读了一次
  loss（D2H + 同步），虚增了 9~12% 步时，而 torch 那侧没有。
- 同口径（循环内不读设备标量，只在末尾 sync）实测：
  我们 fp32 **683.9 ms** vs torch fp32 **621.0 ms** ⇒ 我们**慢 1.10×**；
  我们 fp16 **389.3 ms** vs torch AMP **219.5 ms** ⇒ 我们**慢 1.77×**。
- 我们 fp16 对 torch fp32：**快 1.60×**（389.3 vs 621.0）。
- torch AMP 的优势主要来自把**激活也存 fp16**；我们只把 GEMM 输入换成 fp16，
  真 AMP（激活 fp16 存储）尚未做。
- CUDA Graph 曾量到 1.23×/1.71×，但那是优化前；同口径重测**只有 1.01×**
  （见 docs/CUDA_GRAPH_FINDINGS.md），已不作为优化方向。
- L3 三项仍未全过（24h 长稳 21.8h、2 卡缺硬件、真 AMP 未做），
  因此对外标签保持 **「可用（自用/小团队）」**。

## [0.6.1] - 2026-11-05

### 修复 / 优化
- **`sum_axis` 占用率 bug**：通用 `sum_axis8` 是「一个输出一个线程」，bias 梯度只有 896 线程
  （不足 1 个 block）。新增 `sum_axis0_rows`（一列一 block、256 线程并行归约），
  **省约 14.5 ms/step**，数值核对 1.4e-07。
- **AMP 补上 `GradScaler`**（`amp_mlp.GradScaler`）：动态缩放 + 每步梯度有限性检查 + 溢出跳步。
  没有它时静态缩放两头不讨好（S=1 梯度下溢学不动；S≥32 先快后溢出尖峰）；
  有它之后 0.5B/1200 步只跳步 **2 次（0.17%）**、scale 自愈 256→8192、无 NaN。
  开销 **+1.2%**（221.4 vs 218.8 ms）。
- `TG_MLP_FP16` / `TG_SDPA_LOWMEM` 环境变量开关，便于基准与剖析工具使用。

### 实测（V100，同口径 V=32768 / B=4·S=512 / 24 层 / 6 步）
- 我们：fp32 **488.5 ms**、AMP **209~221 ms**；PyTorch：fp32 **500.0 ms**、AMP **182.9 ms**
  ⇒ fp32 **反超 1.02×**；AMP 仍慢 1.15~1.21×。

### 文档
- `README.md` 新增 **AMP 章节**：何时开（loss 1.5~4）、何时别开（要压到 <1e-2）、
  为什么必须配 `GradScaler`（附三档实测对照）。
- `docs/PRODUCTION_READINESS.md` 置顶「收官摘要」：判据三项 + 性能对照 + 三个关键发现 + 已证伪方向。
- `docs/ENGINEERING_STANDARD.md` §14「rc=700 先怀疑机器」、§13 文档一致性门禁、
  以及 Qwen3 对拍因主机内存不足失败的环境限制记录。

### 限制（如实）
- AMP 有**精度天花板**：同任务 1200 步 AMP 停在 0.79、fp32 到 5.3e-4。
- ③-2（2 卡）仍缺硬件；③-3（正面对决 ≥ PyTorch）仍未达标。
## [0.6.0] - 2026-11-05

### 新增 · 分块交叉熵（`CHUNK_CE`）—— 把 32 GB 卡跑不动的形状变成能跑
- `chunked_ce()`：lm_head 投影与 CE 按 token 分块，用 `T.checkpoint` 保证反向时重算，
  图里只保留块大小的 logits（不落地 `[N,V]`）。
- `cfg["CHUNK_CE"] = True`（可选 `CHUNK_TOKENS`，默认 8192）即在 `sft()` 生效。
- ⚠ **测量作废（2026-11-01 晚复核）**：先前记的"整块 29.92 GiB OOM / 分块 17.72 GiB"
  不可用 —— 当时 `T.checkpoint` 在 CUDA 上是**空操作**，且 `CHUNK_TOKENS` 默认 8192
  大于 N=2048（实际只有 1 块），因此两条路径的差异**不能归因于分块**。
  待用修好的 checkpoint 重测后再填数字。
- 数值等价：loss 差 4.8e-07、梯度差 <1e-5（`tests/test_chunked_ce.py` 4 项）。


### 新增 · `api.train` 支持 `pretrain`（从零预训练）
- 纯文本语料切定长块（`build_lm_sequences`，支持 `.txt` / 每行 `{"text"}`），
  **全 token 参与 loss**（不做对话模板、不遮 prompt）。
- `cfg` 给结构即可随机初始化（`make_random_weights`），CPU 上也可跑。
- V100 实测（V=151936, D=256, L=4, 400 块 × seq 256，120 步）：
  probe **12.0112 → 7.4026（−38.4%）** PASS。
- `dpo`/`grpo` 仍明确抛 `NotImplementedError`（不假装支持）。

### 修复 · 梯度检查点**真正接进训练路径**
- 之前 `T.checkpoint` 只有实现与单测（梯度逐位一致、中间张量 −96.1%），
  但 `Qwen25.forward` 里**没有调用** —— 即"有实现 ≠ 训练省显存"。
- 现在逐层前向抽成 `_layer()`，用 `cfg["GRAD_CKPT"]=True`（或 `model.use_ckpt`）包一层
  `T.checkpoint`；新增 3 项测试断言**前向与梯度逐位一致**（max|diff| = 0.0）。

## [0.4.0] - 2026-11-01

### 新增 · 十行代码训练 API（`tgai_go.api.train`）
- **进程内**训练：模型与 SFT 循环搬进 `tgai_go/train/qwen.py`（与示例共用同一份代码），
  `pip install` 之后不再需要仓库目录。
- `api.train(weights=…, data=…, method="sft", lr=…, batch=…, steps=…, out=…)`
  返回 `TrainingResult`（`probe_base/probe_final/drop_pct/verdict/ckpt/log`）。
- `method` 只支持 `sft`；`pretrain/dpo/grpo` **明确抛错并说明计划**，不假装支持。
- `tiny=True` 走 CPU 迷你权重，几秒跑通（CI 与首次上手用）。
- 实测（V100，Qwen2.5-0.5B × alpaca-zh，120 步）：`probe 2.5825 → 2.3771（−8.0%）`，
  0.5 s/step；同一配置走 CLI 为 −10.6%（**留出集划分不同，数字不可直接相减**，见文档 §11.7）。

### 新增 · `.TG` 加载体检与护栏
- `tools/tg_audit.py`：同一模型用 `.TG` 与 safetensors 各加载一遍比 CE，
  差 > 0.5 判 FAIL 并给出三个最可能原因。
- `load_tg` 加两道体检：线性层布局（`[in,out]` 自动转置或报错）、
  归一化权重是否像 gamma（**单点离谱即报错**并给修法，留 `allow_suspicious_norms`）。
  实测某份 `.TG`：2-D 权重正常，但 6/49 个归一化张量读错（`final_norm` 均值 7.14、
  `ln1` 0.033）⇒ CE 8.30 vs safetensors 3.66。

### ⚠ 更正（2026-11-01 晚）：先前"`.TG` 读错"的判断是**错的**

原判断：某份 `.TG` 的 6/49 个归一化张量"读错"（`final_norm` 均值 7.14、`ln1` 0.033），
并据此加了"**单点离谱即抛错**"的护栏。

对拍官方 safetensors 后推翻：

| 检查 | 结果 |
| --- | --- |
| 容器自洽性 | 290 个 hash **全唯一**、**零重叠**、**零空洞**、不越界 ⇒ 读到的就是文件里存的 |
| 官方 Qwen2.5-0.5B 的 `blocks.0.ln1` 均值 | **0.032**（`.TG` 是 0.033）—— 官方本来就这样 |
| 官方 `final_norm` 均值 | 约 **7.1** —— 也一致 |
| 逐张量相关系数 | v_proj 0.939 / ln1 0.886~0.910（**同模型、被扰动**，不是读错） |

⇒ 真因：**我是拿"归一化权重应≈1"的常识当判据**，而实测模型不满足它。
护栏已改为**只记录不抛错**；客观判据换成两个工具：
- `tools/tg_check.py`：容器自洽性（hash 唯一/无重叠/不越界）—— 纯结构，不猜
- `tools/tg_audit.py`：同模型两路加载**比 CE** —— 对比，不猜阈值

仍未收口：`.TG` 与 safetensors 的相关系数只有 0.89~0.94（不是 1.0），
说明它是**同一模型的扰动副本**（疑似量化/合并/微调来源），这解释了 CE 8.30 vs 3.66，
但源头在哪个转换器仍待查。

### 新增 · 数据层（`tgai_go.train.data`）
- `read_jsonl` 流式读（坏行报行号+修法）、`normalize`（原生/`text`/ShareGPT/Alpaca 四形态）、
  `from_hf`（鸭子类型接 datasets，不 import datasets）、`train_eval_split`（确定性划分）。

### 新增 · 对齐 PyTorch 的能力
- 梯度检查点：梯度逐位一致，保留中间张量 **1041 → 41（−96.1%）**。
- `LAMB` 优化器（仅 CPU；CUDA 上明确报错）。
- `Embedding(weight=)`：直接挂已有参数，省掉一张白分配的随机词表。

### 修复 · 会静默出错的三个问题
1. `api` 解析日志时取「训练 loss（不可比）」而非「探针 loss（可比）」⇒ 报出错误数字。
2. `load_tg` 对布局/归一化异常**静默放行** ⇒ 能训但训坏（现已响亮报错）。
3. `sweep_report` 的 gnorm 阈值按实测重新校准（旧规则会误杀最优配置），并修正排序。

### 度量 · V100 显存构成（Qwen2.5-0.5B / b2 / seq256 / fp32）
权重 2.56 + 优化器 3.46 + 一步激活 **6.85** = 峰值 **16.5 GiB**；
⇒ 激活改 bf16 存储理论省 ≈3.4 GiB。**V100 = sm_70，硬件无 bf16**，加速比必须等 sm_80+。


### 新增 · 梯度检查点 + LAMB（对齐 PyTorch 的两项能力）
- `T.checkpoint(fn, *inputs)` / `T.checkpoint_sequential(fns, x)`：
  前向不建图、反向重算。实测梯度与不检查点**逐位一致**，保留的中间张量
  **1041 → 41（−96.1%）**。限制写清：单输出、fn 必须确定性。
- `T.optim.LAMB`：逐层 trust ratio（`‖p‖/‖u‖`），大 batch 训练用。
  与测试内独立参考实现 5 步一致；满足 `‖Δp‖ = lr·‖p‖` 不变量。**目前仅 CPU**
  （CUDA 上明确报错而不是静默退化）。

### 新增 · RoPE 缩放（Llama-3 / Mistral-v0.3 类模型）
- 读 `config.json` 的 `rope_scaling`，支持 `linear` 与 `llama3`（分段缩放）；
  **yarn / longrope / dynamic / mrope 明确报错并给修法**，不再静默忽略。
- 与官方 `transformers.LlamaForCausalLM` 完整 logits 对拍：llama3 / linear 均 **rel < 1e-5**。
- 为什么重要：忽略 scaling 时短上下文"看着差不多"、长上下文**静默算错**且不报错。

### 工程 · 错误信息质量（算子 / 形状 / 修法）
- CPU 路径此前直接抛 numpy 裸异常，用户看到的是
  `operands could not be broadcast together with shapes (3,4) (7,)`、
  `index 9 is out of bounds for axis 0 with size 5` —— 看不出是哪个算子、怎么改。
  现在 `RMSNorm` / `Embedding` / `functional.rms_norm` 都在最前面做形状校验，
  消息统一为「算子: 形状不匹配 —— 输入最后一维 X，期望 Y（x.shape=…）。修法：…」。
- `cross_entropy` 现在**接受 Tensor 标签**（以前会报 `int() argument must be … not 'Tensor'`）。
- 新增 `tests/test_error_messages.py`：7 个故障场景断言消息里必须同时有
  **算子名**、**具体数字**、**修法**，且不得是 numpy 裸异常。9 项用例。

### 训练 · 真正可续训（存档带优化器状态）
- `save_params(model, path, opt=None, step=0)`：除参数外还存 `__step` / `__opt_t` /
  `__opt_m_*` / `__opt_v_*`。此前只存参数，AdamW 的动量一丢，**续训等于重启动量**，
  轨迹不连续（表现为续训后 loss 先跳一下）。
- `load_params(model, path, opt=None)`：恢复参数 + 动量 + 步号；**老存档（只有参数）仍可加载**。
- `sft(..., start_step=0, opt=None)`：取样改为**按 seed 预生成计划**（第 s 步固定用
  `plan[s*batch:(s+1)*batch]`），warmup 用真实步号 ⇒ 续训与连续跑对齐。
- `examples/qwen25_sft.py --resume <npz>`：一条命令接着训。
- 测试 `tests/test_resume.py`：**"3 步 + 存档 + 续 3 步" 的 loss 与 "一口气 6 步" 的末 3 步一致**。

### 工程 · 行尾与 BOM 规则自动化
- 全仓文本文件统一为 **UTF-8 无 BOM + LF**：`python/ tools/ tests/ docs/ .github/` 共 259 个文件
  （其中包树里 16 个 `.py`/`.c` 之前带着 CRLF 被打进了 wheel）。
- `tools/check_package.py` 新增**wheel 内** BOM/CRLF 检查 —— 检查的是"用户实际拿到的东西"。
- 新增 `tests/test_hygiene.py`：包树与 docs/tools/tests 出现 BOM/CRLF 会让 CI 变红。
- 已知欠账（如实记录）：仓库根目录的历史脚本（`tgai*.py` 等）仍是 CRLF，
  一次性全改会把这 259 个文件的 diff 混进来，留作单独提交。

## [0.3.0] - 2026-10-31

### 修复 · CUDA 地基（重要，影响所有用户）
- **ptx 加载失败 `rc=222` 的根因**：pip 的 NVRTC（12.8）产出的 PTX ISA **8.7** 高于驱动
  （CUDA 12.4）支持的 **8.4**，任何**新模块**都会被驱动 JIT 拒绝。
  此前基线"能跑"只是因为驱动 JIT 缓存里有旧 SASS —— **清缓存后引擎完全不可用**。
- **修复**：改为**优先加载 cubin**（`--gpu-architecture=sm_86` + `nvrtcGetCUBIN`，SASS 不过 JIT），
  PTX 作为兜底；两条都失败时报错给出修法（钉 `nvidia-cuda-nvrtc-cu12<12.5` 或升级驱动），
  并注明"降 arch 到 compute_75 修不了"（已实测）。
- `doctor` 现在打印**模块形态（cubin/ptx）+ NVRTC/驱动版本**，不匹配时主动警告。

### 新增 · 检查点（GGUF 直接拿去训练）
- `tgai_go.train.checkpoints`：`load_gguf` / `load_safetensors_dir` / `save_npz` / `load_npz`
  → 统一的 `Checkpoint(weights, config)`。
  - 未知架构 / 量化类型 / 缺必需权重一律**明确报错并给修法**（静默丢权重比崩溃更危险）；
  - 读 `config.json` 取 `head_dim` / `tie_word_embeddings` / `rope_theta` / `rms_norm_eps`；
  - `tie=true` 时**丢弃重复的 `lm_head`**（官方 Qwen3 safetensors 带它；Qwen2.5 GGUF 的
    `output.weight` 与词嵌入逐位相同），并在数据不一致时报错。

### 新增 · Qwen3 微调支持
- `Qwen25` 增加 `DH` / `QK_NORM` / `QKV_BIAS` / `TIE` 四个开关（默认值保持 Qwen2.5 行为）：
  - **`head_dim` 解耦**：Qwen3-0.6B 是 128 而 `hidden/heads`=64，硬编码必错；
  - **`q_norm`/`k_norm`**：每个 head 对 `head_dim` 做 RMSNorm，且必须在 **RoPE 之前**；
  - **无 qkv bias**；`params()` 按 `id()` 去重（tie 参数曾被优化器更新两次）。
- `tools/qwen3_gpu_parity.py`：与 transformers 现场对拍（**先关 TF32**，失败时用 fp64 归因，
  并做 `q_norm` 梯度冒烟）。

### 新增 · 性能
- **并行平方和内核**（两段归约）：1.36 亿元素 **2.665 ms**，原单线程 `sumsq_sequential`
  需 **7595 ms（2850×）**；且单线程版在 float32 下累积误差达 **63%**，并行版 5e-9。
  梯度裁剪因此恢复可用（`--clip` 默认 1.0）。

### 工程
- **CI 安装 CPU 版 torch**：此前 `ci.yml` 只装 numpy/pytest，导致 **117 项里 71 项
  （所有 torch 对拍）在 CI 里根本没被收集** —— 数值对齐等于没有门禁。
- 新增 `docs/FINETUNE_RESULTS.md`：微调实测的**唯一事实来源**（R1~R8 全部实验的命令、
  超参、probe 起止、判定、日志原文、8 个坑）。
- 测试 129 项（新增 checkpoints 18 项 / Qwen3 6 项 / SFT 冒烟 3 项）。

## [0.2.0] - 2026-10-30

这一版的主题是 **「能替别人的模型训练」**：算子面按 PyTorch 对齐、可插拔扩展、真机结构验证。

### 新增 · 算子
- `nn_ops` 33 个算子（与 `torch` 同名同义，全部有数值对照测试）：
  - 归一化/激活：`rms_norm` `layer_norm` `gelu(tanh)` `silu` `sigmoid` `tanh` `rsqrt` `pow`
  - 线性/形状：`linear` `embedding`（**带反向**）`stack` `split` `chunk` `triu` `tril` `arange`
  - 归约：`argmax` `amax` `amin` `norm` `sum/mean`
  - 损失：`cross_entropy` `nll_loss` `mse_loss` `log_softmax`
  - 优化：`clip_grad_norm_` `sgd_step`
  - 采样：`topk_indices` `multinomial` `randn` `manual_seed`
  - 张量收缩：`einsum`（1~2 操作数常见形式）
- `contrib` 官方插件集 20 个算子（`erf` `gelu_exact` `elu` `selu` `leaky_relu` `mish`
  `hardswish` `hardsigmoid` `log1p` `lerp` `sign` `floor` `ceil` `trunc`
  `flip` `roll` `flatten` `unsqueeze` `squeeze`）

### 新增 · 架构
- **算子插件系统**（`ops_plugin`）：组合式与自带反向两种写法、三种发现方式
  （手工 / `TG_OPS_PATH` / setuptools 入口点）、重名与覆盖核心均有保护。
- **后端三层接口**：core 54 / fallback 8 / optional 17，`backend.report()` 给分层体检。
- **NumPy 参考后端**（`backends/numpy_ref.py`）：接口活文档 + CI 无 GPU 可测 + Vulkan/ROCm 移植模板。
- `tools/backend_port.py`：一条命令看新后端还差哪些核心方法。

### 验证（可复现）
- **真实开源权重对齐**：Qwen2.5-0.5B（GQA 14:2 + qkv bias + RoPE θ=1e6）前向 vs HF
  `max|Δ|=1.278e-04`，相对误差 **6.29e-06**（判据 ≤1e-5）。
- **LLaMA 结构训练对齐**：RMSNorm+RoPE+SwiGLU+GQA、20 步，与 PyTorch 孪生
  逐步 loss 相对偏差 **1.59e-05**、训练后参数相对偏差 **4.90e-05**、loss 下降 99.1%。
- **算子覆盖率**：以 PyTorch 公开面为基准，Tier-1 训练必需 **63 项中 62 项具备（98%）**。

### 工程
- CI（GitHub Actions，CPU-only）：`pytest` + `build` + `check_package`，**已转绿**。
- `tests/conftest.py` 统一 sys.path，测试不再依赖"本地是否装过"。
- `tools/check_package.py`：发布前自检（文件齐 / 语法净 / 干净 venv 可 import / `.cu` 随包）。
- `docs/` 新增：`ENGINEERING_STANDARD.md` `PLUGINS.md` `BACKEND_PORTING.md`
  `OP_COVERAGE.md` `GGUF_TO_TG.md`。

### 修复
- `gguf_to_tg`：token 的 UTF-8 被 `[:255]` 硬截断切坏多字节字符 → 读 `.TG` 时崩溃。
- `gguf_to_tg`：`output.weight` 与 embedding 权重绑定，说明文档化（有意跳过）。
- `ops_plugin`：`kwargs` 未传给反向 → `elu(alpha=)`/`softplus(beta=)`/`lerp` 梯度静默算错。
- 形状算子：额外位置参数（`dims`）未进反向 → `flip`/`roll`/`flatten` 反向 `TypeError`。
- `squeeze`：对齐 torch 语义（非 1 维原样返回，而不是 numpy 那样报错）。
- 全仓库清理 16 个带 UTF-8 BOM 的源文件。

## [0.1.5] - 2026-10-30
- 新增 `tools/check_package.py`（发布前自检）；清理 BOM。

## [0.1.4] - 2026-10-30
- 补齐 `tgai_go` 顶层模块（`pack` `presets` `read_tg`）—— 修复 `--save-resume` / 导出时报
  `ModuleNotFoundError: tgai_go.pack`。

## [0.1.3] - 2026-10-30
- **开箱即用**：自动发现本机任何 Python 环境里的 `nvidia/*/lib`；`TGAI_CUDA_BIN` 支持多路径。
  实测：无 torch 的全新 venv 里 `pip install tgai-orv` → `doctor` 12/12 PASS。

## [0.1.1] - 2026-10-30
- 新增 `[cuda]` 可选依赖（`nvidia-cublas/nvrtc/runtime-cu12`）；`libcublas` 缺失的报错补上修法。

## [0.1.0] - 2026-10-30
- 首个公开版本：推理引擎 + 训练引擎（`tgai_go.train`）、CUDA 算子源码、示例、验收工具、
  `scripts/quickstart.*`、`docs/PRODUCTION_READINESS.md`。
