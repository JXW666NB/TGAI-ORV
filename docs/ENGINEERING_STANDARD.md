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
