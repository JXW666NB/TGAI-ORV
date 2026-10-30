# 变更记录 CHANGELOG

版本遵循 [semver](https://semver.org/lang/zh-CN/)：`0.x` 期间不保证接口稳定，
但**破坏性改动一定写在这里**。

## [未发布]

### 新增
- `examples/qwen25_sft.py`：真实 Qwen2.5 前向 + SFT（GGUF / safetensors 两种权重、权重绑定、
  bf16 权重兜底、`--tiny` 迷你模式、`--holdout` 留出集、固定探针评估）。
- `tools/qwen25_gpu_parity.py`：现场用 transformers 算参考 logits，与我们的引擎逐位比对。
- `tests/test_qwen25_forward.py` / `tests/test_qwen_sft_smoke.py`：Qwen2.5 前向与 SFT 链路的 CPU CI 用例。

### 修复
- 转换器：`tie_word_embeddings` 缺失时改为**比对权重数据**判断是否绑定（元数据里常常没有这个键）。
- `nn_ops.embedding` 补反向（numpy scatter-add），训练时词嵌入才拿得到梯度。

### 实测
- 官方 Qwen2.5-0.5B 权重前向对齐：CUDA 上相对误差 **4.816e-06**。
- 真实 SFT：A10 单卡 30 步，**留出集 loss 0.965 → 0.452（−53.2%）**，0.3 s/step。

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
