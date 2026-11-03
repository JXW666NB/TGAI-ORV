# 工程标准 ENGINEERING STANDARD

> 这份文件管「怎么改」，`docs/PRODUCTION_READINESS.md` 管「改到什么程度算能用」。
> 两条线都过，才算一次合格的交付。

## 0. 一条命令（本地，改完必跑）

```bash
bash scripts/quickstart.sh        # doctor 12 项
python tools/accept.py --quick    # 6 项端到端（含存/续训）
python tools/check_package.py --build   # 发布前才需要：打包+装进干净 venv 验 import
python -m pytest tests -q               # 纯 CPU 单元测试（无需 GPU）
```

**任何一项红了都不许提交**，也不许"下次再修"。

---

## 1. 改动纪律

| 场景 | 要求 |
| --- | --- |
| 改代码前 | 备份 `.bak_<用途>_<时间戳>`；重大改动先打 git tag |
| 改内核/算子 | 先跑 `doctor`（12 项）证明数值没坏，**再**谈速度 |
| 谈速度 | 微基准必须 `sync()`；同一台机器、同一份数据、多跑几次取中位数；注明波动范围 |
| 改动训练循环 | 必须跑 `accept.py` 的「存/续训」项（断点不得跳变） |
| 结论 | 区分「实测」与「推断」；没测过的写"未验证"，不许写成"应该没问题" |

---

## 2. 文件与编码

- **UTF-8 无 BOM**。BOM 会让 `tomllib`/部分工具直接报错，历史上已踩过一次（`tgai_go/train/models/__init__.py`）。
- **源码文件用 LF**；`*.sh` 尤其不能用 CRLF（会让重定向文件名带上 `\r`，历史上已踩过一次）。
- 不进仓库：`__pycache__`、`*.pyc`、`build/`、`dist/`、`_z_*.py` 之类临时脚本。
- 新增文件请在提交里说明用途；不确定要不要公开的，先放本地。

---

## 3. 错误信息标准

每条会抛给用户的错误，必须包含三要素：

```
哪个算子 / 什么形状 / 怎么修
```

例（真实实现）：

```
找不到 libcublas.so.12。已查目录: [...]
修法：pip install "tgai-orv[cuda]"（裸机器），或设 TGAI_CUDA_BIN 指向含 libcublas.so.12 的目录
```

禁止裸 `CudaError: rc=1` 这种没有修法提示的报错。

---

## 4. 版本与发布

- **semver**：`0.1.x` 修 bug / 小改；`0.2.0` 加能力；`1.0.0` 才敢叫稳定。
- 每次发布必须：`check_package.py` 通过 → `python -m build` → `twine upload` → **打 git tag** → 写进 `CHANGELOG.md`。
- PyPI 包名 `tgai-orv`，import 名始终 `tgai_go`（换包名不影响用户代码）。
- 发布后自查：`pip download tgai-orv==<版本>` 能取到；`pip install` 后 `doctor` 能跑。

---

## 5. 测试

| 层 | 位置 | 需要 GPU | 跑法 |
| --- | --- | --- | --- |
| 单元（纯 CPU） | `tests/` | 否 | `python -m pytest tests -q` |
| 算子自检 | `tgai_go.train.doctor` | 是 | `python -m tgai_go.train.doctor` |
| 端到端验收 | `tools/accept.py` | 是 | `python tools/accept.py --quick` |
| 长稳 | `tools/soak_test.py` | 是 | `python tools/soak_test.py --minutes N` |

CI（GitHub Actions）只跑**不需要 GPU** 的那部分 —— 它能拦住"能装不能跑"这一类，
（历史上两次真实事故：`.cu` 没随包、`tgai_go.pack` 缺模块，都属这一类）。

---

## 6. 提交信息

```
<type>(<scope>): <一句话说清做了什么>

<为什么这么做 / 关键取舍 / 实测数字>
```

`type` 只能是：`feat` / `fix` / `perf` / `test` / `docs` / `chore` / `build`。
**提交里带上实测数字**（ms、loss、relerr），方便以后追溯为什么改。

---

## 7. 产线边界

- V100（`D:\TGAI`）是**推理产线**：`run_tgaigo_server.py` + `libtgai_engine_gpu.dll`，端口 11450，**不走 Python 训练包**。
  训练侧改动不需要、也不应该部署到它上面。
- 动产线前必须有明确授权；改 DLL 必须备份；重启走 `schtasks`，**禁止 `taskkill /f`**。
- 只读排查可以自主做（`dir`/`wmic`/`nvidia-smi`），但不要写文件、不要重启服务。

---

## 8. 每个新能力都要有「反例」

只写"能做什么"不够，必须同时给出：

- 它**不**支持什么（例如：跨机不支持、激活不省显存）
- 什么时候会**退化**（例如：bf16 在 GEMM 占比低时反而更慢）
- 达不到的判据要**留在表里显示未通过**，不许悄悄删掉

`docs/PRODUCTION_READINESS.md` 的"它不是什么"一节就是这条的示例。

---

## 9. 基准口径（2026-11-03 起强制）

**因为踩过坑**：我们的基准曾经每一步都读一次 loss（`float(loss.to_numpy())`），
那是 D2H + 整设备同步，把异步流水线打断，**虚增 9~12% 步时**；
而 PyTorch 那侧没这么写。于是得出"我们比 torch fp32 快 1.25×"的**错误结论**，
后来同口径重测才发现是**慢 1.10×**。详见 `docs/BENCHMARK_RIGOR.md`。

强制规则：

1. **计时循环内不许读任何设备标量**（loss / 梯度范数 / 任何 `.to_numpy()`）。
   需要监控 loss 时，另跑一份「带同步」的版本，并在结论里**标注它不代表吞吐**。
2. **对手用同一段测量代码**：步数 N、是否 sync、是否含数据加载，两边必须一致。
3. 结论里必须写清：`N 步 / 是否含同步 / 是否含数据加载 / 平均值还是中位数`。
4. 单次测量一律加 `sync()` 再取时间；多次取平均，必要时给中位数与最大偏差。

---

## 10. 触碰驱动的测试必须子进程化

**因为踩过坑**：AMD 驱动在同一进程里加载/卸载 `vulkan-1.dll`、或反复建销
`VkInstance`，会让 **pytest 进程在退出时段错误**（表现为跑完一堆点后直接崩、
没有失败报告）。NVIDIA 侧创建 `VkDevice` 也建议同样处理。

规则：任何真实触碰 GPU / Vulkan / 驱动的用例，**主体逻辑放子进程**，
父进程只解析子进程 stdout 并断言；子进程崩溃按失败处理，不吞掉。

示例见 `tests/test_ops_vk.py`、`tests/test_ops_vk_ops.py`（`_child` / `_vk_ready`）。

---

## 11. 插件包（`.tgop`）的硬性契约

| 契约 | 谁在管 | 违反会怎样 |
| --- | --- | --- |
| 必须有 `python` 兜底档 | `ops_bundle._load_manifest` + `tools/tgop_pack.py` | 打包直接失败；加载报错并给修法 |
| 引用的文件必须真在包里 | `tools/tgop_pack.py`（打包时）+ `ops_bundle._pick`（加载时） | 打包失败 / 加载时**顺延**到下一个后端 |
| 后端按 `cuda → hip → musa → ascend → bang → vulkan → cpu → python` 挑 | `ops_bundle._ORDER` | — |
| 声明了后端但加载器未实现 | `ops_bundle.load` | 明确报错并列出本机可用后端，**绝不静默跑错** |

**能力矩阵（当前真实状态）**：

| 后端 | 加载器 | 实测 |
| --- | --- | --- |
| cuda | ✅ | GELU 前向 relerr **7.78e-08**（V100） |
| cpu（C JIT） | ✅ | 前向/反向正确（自动发现 zig/gcc/clang/tcc，`pip install ziglang` 即可零配置） |
| vulkan | ✅ | relu 前/反向 **0.000e+00**（AMD 核显）；但**比 numpy 慢 20~150×**，见 `docs/VULKAN_BACKEND.md` |
| python | ✅ | 兜底，永远可用 |
| hip / musa / ascend / bang | ❌ | 只有探测，加载器未实现；缺硬件**没法测**，不写假的 |

---

## 一条命令跑完全部门禁

```bash
python tools/verify_all.py          # 四道门禁，约 90 秒（纯 CPU）
python tools/verify_all.py --quick  # 跳过打包，约 40 秒
```

| 门禁 | 覆盖什么 | 实测耗时 |
| --- | --- | --- |
| 单元测试 | 形状 / 错误信息 / 存档往返 / Qwen2.5+Qwen3 前向对拍 / SFT 冒烟 | 15.3 s |
| 打包自检 | 文件齐 / 语法净 / 干净 venv 可 import / `.cu` 随包 | 46.5 s |
| LLaMA 结构对拍 | 与 PyTorch 孪生逐步对齐（60M 级可跑，这里用 tiny 档） | 7.5 s |
| Qwen3 真权重对拍 | `.TG` 切片 vs 官方 `Qwen3ForCausalLM` | 21.0 s |

**为什么要有它**：判据写进文档只是一半，**能一条命令重跑**才是可执行的标准。
发布前、改完引擎后、怀疑回归时，先跑它，再谈别的。

> 说明：Qwen3 真权重对拍需要本机的 `Qwen3-fix.TG`（1.1 GB，不入库）；
> 找不到时该项会**跳过并标注**，不会假装通过。

---

## 已经「自动化」的规则（不靠自觉）

| 规则 | 谁在管 | 违反会怎样 |
| --- | --- | --- |
| 包树/文档/工具目录无 BOM、无 CRLF | `tests/test_hygiene.py` | CI 红 |
| **wheel 内**无 BOM、无 CRLF | `tools/check_package.py` | 打包自检 FAIL |
| wheel 内 `.py` 全部可编译 | `tools/check_package.py` | 打包自检 FAIL |
| 干净 venv 能 import | `tools/check_package.py` | 打包自检 FAIL |
| 版本号唯一来源 + CHANGELOG 有对应段落 | `tests/test_version.py` | CI 红 |
| 数值对齐（LLaMA / Qwen2.5 / Qwen3） | `tests/test_*_forward.py`、`tools/verify_all.py` | CI 红 / 门禁 FAIL |
| 工具链版本风险（NVRTC vs 驱动） | `tgai_go.train.doctor` | 运行时主动警告 |
| 补位 token 不计入 loss | `tests/test_qwen3_forward.py::test_pad_targets_are_masked` | CI 红 |
| 错误信息含「算子名 + 形状 + 修法」 | `tests/test_error_messages.py` | CI 红 |
| 数据截断率 / 答案 token 一致性 | `tools/dataset_report.py` 的硬性自检 | 非零退出 |

> 判断一条规则有没有"自动化"，标准很简单：**违反它的时候，有没有东西会红**。
> 只写在文档里的规则，等于没有规则。

---

## 12. 错误信息「必须有修法」已被门禁接管

	ests/test_error_messages.py 用 AST 扫源码：**新增/改写模块**里每条 
aise
的消息必须含「修法」，否则 CI 红。历史模块先只统计（LEGACY 列表），逐个纳入。

消息是变量（
aise ValueError(msg)）时静态判不了，扫描器会**跳过并计数**，
不假装通过、也不误杀。扫描器本身也有自测用例。
