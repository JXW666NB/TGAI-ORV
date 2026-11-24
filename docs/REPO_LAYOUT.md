# 仓库结构（1.0.27 起）

> 本仓库**只包含构建与运行 ORV 所必需的文件**（外加 wiki、宣传用性能原始日志、算子源码）。
> 测试、实验脚本、评测数据、第三方复现包、历史产物一律不入库；
> 本地开发需要它们时放在 `lab/`（已 gitignore）。

## 目录

| 路径 | 作用 |
| --- | --- |
| `python/tgai_go/` | **主产物**：pip 包。`train/` 是自动微分 + 算子 + 优化器 + 训练循环；`_native/` 是推理引擎原生库；`wiki/` 是随包分发的离线 wiki；`train/kernels/` 是运行时用 NVRTC 现编的算子源码 |
| `src/` | 原生后端 C 源码：`tgai_engine.c`（CPU/通用引擎）、`tgai_backend.c`、`tgai_chat_meta.c`、`launcher/` |
| `src/mm/` | 多模态子模块（图像/音频/路由） |
| `cuda/` | CUDA 后端 `.cu` 源码（非 NVRTC 的另一条路径）与构建脚本 |
| `cpu/` | CPU 后端构建脚本（`build.ps1` / `build_msvc.bat`），产物在 `cpu/build/`（不入库） |
| `shaders/` | Vulkan compute shader：`.comp` 源 + `.spv` 编译产物 |
| `third_party/vulkan/` | Vulkan 头文件与 volk（构建 Vulkan 后端用） |
| `training/kernels/` | 训练相关算子源码（`.cu`） |
| `wiki/` | wiki 源文件（由 `scripts/sync_wiki_package.py` 同步进 `python/tgai_go/wiki/`） |
| `bench/` | 逐模块微基准（一个模块一个文件，CUDA event 计时） |
| `docs/` | 设计文档与实测报告；`docs/measurements/` 是**用于宣传的性能测试原始日志** |
| `examples/` | 可直接运行的示例 |
| `scripts/` | 一键上手脚本（`quickstart.bat` / `quickstart.sh`）与 wiki 同步/校验 |
| `tools/` | 少量**会被执行或推荐使用**的脚本：`accept.py`（验收）、`gguf_to_tg.py`（`.gguf → .TG`，`_engine.py` 会自动调用）、`check_plugin.py`、`tg_audit.py`、`tg_requantize.py`、`make_tg_names.py`、`dataset_report.py`、`write_chat_meta.py`、`verify_svs_ops.py`、`bin/glslangValidator.exe`（Vulkan 构建用） |
| 根目录 `build_*.bat` / `rebuild_cpu.bat` | 各平台构建入口 |
| `tgai*.py` / `tgai.bat` | 运行时入口（聊天 / 服务端 / TUI / UI / 启动器） |

## 构建

```bat
REM CPU 引擎（src/ → cpu/build/libtgai_engine.dll）
build_engine.bat

REM CUDA 引擎（cuda/ + src/ → cpu/build/libtgai_engine_gpu.dll）
build_cuda.bat sm_70

REM Vulkan 后端 + 自测（shaders/ + third_party/vulkan + tools/bin/glslangValidator.exe）
build_vulkan.bat

REM Python 包（含 kernels / wiki / _native）
python -m build
```

## 运行

```bash
pip install tgai-orv          # 或 pip install -e .
python -c "import tgai_go; m=tgai_go.load('model.TG'); print(m.generate('你好'))"
```

## 不入库的东西（`.gitignore` 已覆盖）

`lab/`（实验）、`tests/`（测试）、`release/`、`embedded/`、`tcc/`、`skills/`、
`.github/`、`dist/`、`backups/`、`build/`、`*.TG`、`*.gguf`、`*.npy`、`*.npz`、
`cpu/build/` 产物、`__pycache__/`。

需要这些内容时从本地备份或对应上游获取；仓库本身保持最小可构建集合。
