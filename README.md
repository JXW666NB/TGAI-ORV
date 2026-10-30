# TGAI ORV — Origin Requiem Vehicle（原生之初）

**一套自研引擎，推理 + 训练，全程不依赖 PyTorch。**
同一个模型，从训练到部署用同一套代码、同一份量化、同一份词表。

> **ORV = Origin Requiem Vehicle（原生之初）**
> "Origin" 指它从零写起（内核、autograd、优化器都是自己的）；
> "Requiem" 是给"必须装 PyTorch 才能训练"那个时代写的挽歌；
> "Vehicle" 是载体——它承载模型从训练走到部署。

## 与 TGAI GO 的关系

| 仓库 | 定位 |
| --- | --- |
| **[TGAI GO](https://github.com/JXW666NB/TGAI-GO)** | **纯推理引擎**。跨平台、零依赖、嵌入式也能跑（ESP32 8MB PSRAM 到 CUDA 服务器） |
| **TGAI ORV（本仓库）** | **完全体**。推理引擎 **+ 训练引擎**，训完直接部署，不用换栈 |

本仓库包含 TGAI GO 的全部推理能力，并额外提供 `tgai_go.train` 训练栈。

## 一条命令开始

```bash
git clone https://github.com/JXW666NB/TGAI-ORV.git && cd TGAI-ORV
bash scripts/quickstart.sh        # Windows 用 scripts\quickstart.bat
```

它会自动检查环境、装 numpy、打印 GPU 信息、设好 `PYTHONPATH`，然后跑 **12 项自检**，
失败时明确告诉你**哪个算子 / 什么形状 / 怎么修**。

## 训练：改一行 import 就行

```python
# import torch as T                 # 以前
import tgai_go.train as T           # 现在（其余代码不用动）
```

`tgai_go.train` 提供 PyTorch 风格的 `Tensor` / autograd / `optim.AdamW` / `Module` / `functional`，
**已有 PyTorch 训练脚本只需换 import**。

### 实测速度（同一份训练代码、同种子、各 600 步，2026-10-30）

| 机器 | ORV | PyTorch | 比值 |
| --- | --- | --- | --- |
| RTX 4090 | 18.9 ms/step | 27.1 ms | **快 1.43×** |
| Tesla V100 | 41.4 ms/step | 52.4 ms | **快 1.27×** |
| NVIDIA A10 | 49.3 ms/step | 57.0 ms | **快 1.16×** |

### 能力

- **算子**：RMSNorm / RoPE / SwiGLU / attention（含 flash 版）/ MoE，融合核 + CUDA Graph
- **混合精度**：`TG_BF16=1` / `TG_FP16=1`（A10 上 bf16 跑 600 步 loss 偏差 0.03%）
- **数据管线**：流式读取，1 GB 语料 host 内存 **600 MiB**（旧实现要 17.5 GiB）
- **多卡**：`tools/launch_dp.py` 数据并行（单卡 2 rank 参数 SHA256 一致）
- **续训**：`--save-resume` / `--resume`，断点处 loss 差 0.06%
- **形状**：14 组配置实测通过（含 `dh=8~128`、`S=8~256`、`FF=1000` 非 128 倍数、`L=16`）

## 自检与验收

```bash
python -m tgai_go.train.doctor     # 12 项算子级自检
python tools/accept.py             # 6 项端到端验收，输出 PASS/FAIL 表
```

## 目录结构

```
TGAI-ORV/
├── python/tgai_go/train/    训练引擎（cuda.py 运行时 + autograd + 优化器）
├── training/kernels/        CUDA 算子源码（.cu，运行时由 NVRTC 编译）
├── examples/                可跑的示例（10M 对话模型 + 流式数据）
├── tools/                   验收 / 长稳 / 分层计时 / 发布同步
├── scripts/                 quickstart.sh / quickstart.bat
├── docs/
│   ├── QUICKSTART_TRAIN.md      从零到训出第一个模型（一页 A4）
│   └── PRODUCTION_READINESS.md  L0–L3 判据与全部实测证据
├── src/ cuda/ cpu/ embedded/ shaders/ tcc/   推理引擎（与 TGAI GO 同源）
└── third_party/             上游 vendored 依赖
```

## 状态（不夸大）

| 级别 | 状态 |
| --- | --- |
| L0 能跑（数值正确） | ✅ |
| L1 能训（形状/语料/速度/显存） | ✅ |
| L2 能被人用（自检/文档/报错） | ✅ |
| L3 生产级（24h 长稳、2 卡加速比） | 🟡 部分（缺硬件未验证） |

⇒ 当前标签：**「可用（自用/小团队）」**，尚未达到「生产级」。
全部判据与实测证据见 [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md)。

## 许可

见仓库内 `LICENSE`。
