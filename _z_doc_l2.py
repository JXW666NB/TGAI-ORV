import io

# 1) README 版本号
s = io.open("README.md", encoding="utf-8").read()
s = s.replace("version-0.1.0-blue", "version-0.1.4-blue")
s = s.replace("当前 `0.1.3`", "当前 `0.1.4`")
io.open("README.md", "w", encoding="utf-8", newline="\n").write(s)

# 2) 标准文档追加 L2 实证
SEC = """

---

## ★ L2「别人能拿去用」—— 两条路径实测通过（2026-10-30 晚，A10）

这一节是**判据 L2 的真正落地证据**：不是"在我们机器上能跑"，而是"从一个干净环境开始能跑"。

### 路径 A：pip 安装（无需仓库、无需 torch）

```bash
python3 -m venv /tmp/orv && /tmp/orv/bin/pip install tgai-orv
/tmp/orv/bin/python -m tgai_go.train.doctor
```

| 环境 | 结果 |
| --- | --- |
| 全新 venv，**没有 torch**，**不设任何环境变量** | **12/12 PASS** |
| 同一 venv，若设 `TGAI_CUDA_BIN`（多路径） | 12/12 PASS（备用路径也验证） |

引擎会自动去本机任何 Python 环境里找 `nvidia/*/lib`（例如系统装了 torch、而我们在干净 venv 里）。

### 路径 B：git clone + 一条命令

```bash
git clone https://github.com/JXW666NB/TGAI-ORV.git && cd TGAI-ORV
bash scripts/quickstart.sh
```

| 项 | 结果 |
| --- | --- |
| 环境检查 / GPU 信息 / PYTHONPATH 设置 | 正常 |
| 12 项自检 | **12/12 PASS** |
| 下一步指引 | 正确打印出训练与验收命令 |

### 路径 C：六项端到端验收

```bash
PYTHONPATH=$PWD/python python3 tools/accept.py --quick
```

| 项 | 结果 |
| --- | --- |
| doctor / 形状 4 组 / fp32 120 步 / bf16 120 步 / 显存 | 5 项 PASS |
| **存·续训** | PASS（断点 loss 2.156563 → 2.003522，上升 −7.1% 属正常下降） |
| **验收结论** | **PASS（0 项 FAIL）** |

### 过程中抓到的两个「能装不能跑」级缺陷（都已修）

| # | 现象 | 根因 | 修法 |
| --- | --- | --- | --- |
| 1 | 干净 venv 里 `找不到 libcublas.so.12` | 只查系统目录；DSW 的 cuBLAS 来自**系统 torch**，venv 里没有 | 自动扫描本机所有 Python 环境的 `nvidia/*/lib`；`TGAI_CUDA_BIN` 支持多路径；报错直接写修法 |
| 2 | `--save-resume` / 导出 `.TG` 报 `No module named 'tgai_go.pack'` | ORV 树是「TGAI-GO 包 + 覆盖的 train 子包」，**缺了训练侧依赖的兄弟模块** | 以完整副本为准同步 `tgai_go` 全部顶层 `.py`（pack / presets / read_tg 等 14 个） |

**这两个都是"装了但一用就崩"，只有真正在干净环境里跑才会暴露** ——
这也是为什么必须做清环境验证，而不是在开发目录里反复自检。

### PyPI 版本轨迹

`0.1.0`（首发，含算子源码）→ `0.1.1`（加 `[cuda]` extra + 报错修法）→
`0.1.3`（自动发现 CUDA 库，**开箱即用**）→ `0.1.4`（补齐 `tgai_go` 顶层模块）
"""
io.open("docs/PRODUCTION_READINESS.md", "a", encoding="utf-8", newline="\n").write(SEC)
print("文档与 README 已更新")
