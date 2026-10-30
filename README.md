# TGAI ORV — Origin Requiem Vehicle（原生之初）

<p align="center">
  <strong>Train and infer with one codebase, without PyTorch. / 训推一体，不依赖 PyTorch。</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-blue" alt="version">
  <img src="https://img.shields.io/badge/python-3.9+-green" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="license">
  <img src="https://img.shields.io/badge/self--test-12%2F12-brightgreen" alt="selftest">
  <img src="https://img.shields.io/badge/speed-1.16~1.43%C3%97%20vs%20PyTorch-red" alt="speed">
</p>

> **Origin** — kernels, autograd and optimizers written from scratch.
> **Requiem** — a farewell to "you must install PyTorch to train".
> **Vehicle** — carries one model all the way from training to deployment.
>
> **Origin** —— 内核、autograd、优化器全是从零写的。
> **Requiem** —— 写给"必须装 PyTorch 才能训练"那个年代的挽歌。
> **Vehicle** —— 载着同一个模型，从训练一路走到部署。

---

## 这玩意到底是啥 / What the Hell Is This Thing

Most "self-built engines" only do inference — the moment you want to train, you are back on PyTorch.
**TGAI ORV does both**, and it is literally one line of import away:

大多数"自研引擎"只做推理，一到训练就得回头求 PyTorch。
**TGAI ORV 两边都做了**，而且切换只差一行 import：

```python
# import torch as T                 # 以前 / before
import tgai_go.train as T           # 现在 / now — 其余代码不用动
```

Your PyTorch training script keeps working: `Tensor`, autograd, `optim.AdamW`, `Module`, `functional` —
same names, same semantics.

原来的 PyTorch 训练脚本不用改逻辑：`Tensor`、autograd、`optim.AdamW`、`Module`、`functional`，名字和语义都对齐。

---

## 吹牛区（但都是实测）/ The Bragging Section (All Measured)

Same training code, same seed, 600 steps each / 同一份训练代码、同种子，各 600 步：

| GPU | ORV | PyTorch | Ratio |
| --- | --- | --- | --- |
| RTX 4090 | 18.9 ms/step | 27.1 ms | **1.43× faster** |
| Tesla V100 | 41.4 ms/step | 52.4 ms | **1.27× faster** |
| NVIDIA A10 | 49.3 ms/step | 57.0 ms | **1.16× faster** |

- 🧠 **Real kernels, not wrappers** — RMSNorm / RoPE / SwiGLU / attention (incl. flash) / MoE as fused CUDA kernels, plus CUDA Graph.
  **真算子，不是套壳**：RMSNorm / RoPE / SwiGLU / 注意力（含 flash）/ MoE 都是融合核，还带 CUDA Graph。
- 🧪 **12-check self test** — `python -m tgai_go.train.doctor`, every failure tells you *which op / what shape / how to fix*.
  **12 项自检**：每项失败都写清"哪个算子 / 什么形状 / 怎么修"。
- 📉 **1 GB corpus, 600 MiB RAM** — streaming data pipeline (the old one needed 17.5 GiB).
  **1 GB 语料只吃 600 MiB 内存**：流式管线（旧实现要 17.5 GiB）。
- ♻️ **Resume is lossless** — 40+40 steps vs 80 continuous: 0.06% difference.
  **续训无损**：40+40 步对比连续 80 步，末 loss 差 0.06%。
- 📐 **14 shape configs verified** — including `dh=8~128`, `S=8~256`, `FF=1000` (not a multiple of 128), `L=16`.
  **14 组形状实测通过**：含 `dh=8~128`、`S=8~256`、`FF=1000`（非 128 倍数）、`L=16`。
- ⚡ **Mixed precision** — `TG_BF16=1` / `TG_FP16=1`; bf16 on A10 keeps loss within 0.03% over 600 steps.
  **混合精度**：bf16 在 A10 上 600 步 loss 偏差 0.03%。
- 🔗 **Data parallel** — `tools/launch_dp.py`; two ranks on one GPU end with **bit-identical parameters**.
  **数据并行**：单卡起 2 个 rank，最终参数 SHA256 完全一致。

---

## 一条命令跑起来 / One Command

```bash
git clone https://github.com/JXW666NB/TGAI-ORV.git && cd TGAI-ORV
bash scripts/quickstart.sh          # Windows: scripts\quickstart.bat
```

It checks Python, installs numpy if missing, prints your GPU, sets `PYTHONPATH`,
then runs the 12-check self test and tells you the next command.

它会查 Python、缺 numpy 就装、打印 GPU、设好 `PYTHONPATH`，然后跑 12 项自检，最后告诉你下一步敲什么。

Want to train something right away / 想直接训一个：

```bash
PYTHONPATH=$PWD/python python3 examples/exam2_chat10m_tgai.py --device cuda --steps 600 --out chat10m
```

---

## 它不是什么 / What It's NOT

We would rather undersell than lie / 宁可说少，不说满：

| Gate | Status |
| --- | --- |
| L0 correctness / 数值正确 | ✅ |
| L1 trainability / 能训 | ✅ |
| L2 usability / 能被人用 | ✅ |
| L3 production / 生产级 | 🟡 partial — 24h soak and 2-GPU speedup **not verified** (no hardware) |

⇒ Label: **"usable (self / small team)" / 「可用（自用/小团队）」**, not "production-ready".
⇒ 标签是**「可用（自用/小团队）」**，不是「生产级」。

Every number above is measured on real hardware; evidence lives in [`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md).
上面每个数字都在真机上测过，证据在 [`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md)。

---

## 仓库结构 / Layout

```
TGAI-ORV/
├── python/tgai_go/train/     training engine / 训练引擎（运行时 + autograd + 优化器）
├── training/kernels/         CUDA operator sources / 算子源码（.cu，NVRTC 运行时编译）
├── examples/                 runnable samples / 可跑示例（10M 对话模型 + 流式数据）
├── tools/                    accept / soak / profile / sync
├── scripts/                  quickstart.sh · quickstart.bat
├── docs/
│   ├── QUICKSTART_TRAIN.md       zero to your first model / 从零到第一个模型（一页 A4）
│   └── PRODUCTION_READINESS.md   L0–L3 gates + evidence / 判据与实测证据
├── src/ cuda/ cpu/ embedded/ shaders/ tcc/   inference engine / 推理引擎
└── third_party/              vendored upstream / 上游依赖
```

## 关联仓库 / Sibling Repo

| Repo | What |
| --- | --- |
| [**TGAI GO**](https://github.com/JXW666NB/TGAI-GO) | Inference only / 纯推理引擎，跨平台，嵌入式也能跑 |
| **TGAI ORV** (this / 本仓库) | Inference **+ training** / 完全体，训推一体 |

## License

[MIT](LICENSE) © 2026 JXW666NB — do whatever you want, just keep the notice.
