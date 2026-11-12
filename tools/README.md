# tools/ 里哪些脚本能直接跑

这个目录混了两类东西。2026-11-12 外部报告指出「工具脚本腐烂」（硬编码作者本机路径、跑就崩），
下面把它们分清，避免再有人照着跑。

## 一、通用工具（任何克隆本仓库的人都能跑）

| 脚本 | 用途 | 依赖 |
| --- | --- | --- |
| `accept.py` | 一条命令跑验收（doctor + 形状矩阵 + 精度 + 速度） | 仓库内 + CUDA |
| `h2h_clean.py` | ORV vs PyTorch 头对头（`--impl orv\|torch`） | 另需 torch |
| `mem_breakdown.py` | 显存归因：权重/梯度/优化器/激活四类（`--ckpt` 可开检查点） | 仓库内 + CUDA + torch |
| `profile_block.py` | 逐阶段剖析一层（`--fp16` 对比），找瓶颈用 | 仓库内 + CUDA |
| `attn_ref.py` | 融合注意力的**逐位回归**基准（`--save` / `--check`） | 仓库内 + CUDA |
| `p100_fit.py` | 按指定形状测峰值显存，判断某张卡装不装得下 | 仓库内 + CUDA |
| `ckpt_one.py` / `ckpt_scale.py` / `ckpt_trace.py` | 检查点的显存/峰值诊断（**每个配置必须独立进程**，原因是 `peak_live_bytes` 从不重置） | 仓库内 + CUDA |
| `smoke_min.py` / `smoke_skill.py` | 最小训练冒烟（skill 里那段代码的逐字副本） | 仓库内 + CUDA |
| `inventory.py` | 生成 `docs/CAPABILITY_AUDIT.md`（模块/接口/内核/测试计数） | 仓库内 |
| `check_all_refs.py` | 检查文档里引用的路径是否存在 | 仓库内 |
| `check_package.py` | 发布前自检 wheel（文件齐、无 BOM/CRLF、干净 venv 可 import） | 需先 `python -m build` |
| `release_09NN.py` | 各版本的发布脚本（版本号、CHANGELOG、构建、产物校验） | 仓库内 |
| `publish_pypi.py` | 上传到 PyPI（需要 `~/.pypi_token`） | 另需 twine |

## 二、作者本机的一次性调试脚本（**跑不了，别照着跑**）

`align_*.py`、`qwen*_parity.py`、`kernel_count.py`、`accum_floor.py`、`pytorch_align_debug.py` 等。

它们依赖仓库外的东西，例如：

* `align_*.py` —— 指向作者本机的 `D:\tgai_orv\TGAI-ORV-main\python` 与 `v27_train_run.py`；
* `kernel_count.py` —— 需要 `V27_RUN`（作者本机的 V2.7 训练入口）与 `V27_TG`（V2.7 权重），
  两者**都不在仓库里**；缺了会明确告诉你需要什么，并指向 `profile_block.py`；
* `qwen*_parity.py` —— 需要本机下载的 HF 权重。

**为什么还留着**：它们是当初得出 `docs/` 里那些数值结论的原始脚本，留着可追溯。
想要通用版本请用上表里的工具；`profile_block.py` 与 `mem_breakdown.py` 自带合成权重，不需要任何外部文件。

## 三、约定

* 新增通用工具请**只依赖仓库内路径**（用 `os.path.dirname(os.path.dirname(__file__))` 推仓库根），
  外部输入一律走环境变量或命令行参数，并在缺失时给出「修法」。
* 新增一次性脚本请放进本目录并在上表第二类里登记，避免下次又有人照着跑。
