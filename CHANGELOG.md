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

## [未发布]

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
