# TGAI GO 训练 · 一页 A4 入门

> 目标读者：想用自己的模型/语料在 TGAI GO 上训练的人。
> 全程只有三件事：**跑自检 → 准备语料 → 开训**。

---

## 0. 环境要求

| 项 | 要求 |
|---|---|
| 系统 | Windows 10/11 x64 或 Linux x64 |
| 显卡 | NVIDIA（CUDA 11.8+ 驱动）。无卡会自动退回 CPU（能跑但很慢） |
| Python | 3.9+（**不需要装 PyTorch**） |
| 依赖 | `numpy`（唯一必需）；`paramiko` 只有远端脚本才用 |

安装：把 `python/tgai_go` 放进 `PYTHONPATH`，或 `pip install -e python/`。

---

## 1. 先跑自检（30 秒）

```bash
python -m tgai_go.train.doctor
```

期望输出结尾：

```
自检通过：10/10 项 PASS
```

它真跑 GPU，逐项检查：GEMM、转置 GEMM、逐元素（含 float4）、RMSNorm、
SiLU-GLU、带 `ignore_index` 的交叉熵、causal attention、AdamW、一步训练。
**任何一项 FAIL 都会打印「怎么办」**，先修环境再往下走。

---

## 2. 准备语料

每行一个 JSON，字段 `text`，内部用九标记格式：

```json
{"text": "<用户>\n你是谁研发的<TGAI><回答>\n我是由JXW开发的TGAI。"}
```

规则：
- `<TGAI>` 前**不要**换行（引擎就是这样拼的）；
- 回答长度 **≤ 40 字符**（超出的会被跳过，用于字符级小模型）；
- 一行一条对话。

用环境变量指定语料（**会替换默认语料**，不是追加）：

```bash
# Windows
set TG_CORPUS=D:\data\my_corpus.jsonl
# Linux
export TG_CORPUS=/data/my_corpus.jsonl
```

⚠️ 两个常见坑（都已在代码里改成硬报错/告警）：
- 语料解析不出任何一条 → **直接报错**，不会静默用内置样例；
- 超过 `TG_MAX_CHARS`（默认 200 万字符）→ 打印「扫过多少 / 用了多少」，
  要全量设 `TG_MAX_CHARS=0`。

---

## 3. 开训

```bash
python examples/exam2_chat10m_tgai.py --device cuda --steps 600 --batch 64 --out my_model
```

关键参数：

| 参数 | 含义 | 建议 |
|---|---|---|
| `--batch` | 每步取多少条对话 | **语料大于几千条时必须给**（否则整语料当一批，显存爆炸） |
| `--steps` | 训练步数 | 小语料 600 步足够；语料大按 `epoch × ceil(条数/batch)` 算 |
| `--lr` | 学习率 | 默认 2e-3（AdamW） |
| `--out` | 输出名 | 产物：`my_model.TG`（引擎格式）+ 词表/元信息 |

产物直接能被 C 引擎加载（`.TG` 单文件：权重 + 词表 + 对话模板）。

---

## 4. 可选开关（出问题时逐个关掉定位）

| 环境变量 | 作用 |
|---|---|
| `TG_NO_FUSED=1` | 关掉融合内核（RoPE/GLU），回退组合实现 |
| `TG_NO_VEC4=1` | 关掉 float4 向量化 |
| `TG_LAUNCH_FAST=0` | 关掉 launch 参数模板缓存 |
| `TG_CUDA_GRAPH=1` | 开 CUDA Graph 重放（**只在固定 batch 时可用**，与 `--batch` 互斥） |
| `TG_FP16=1` | fp16 GEMM（实验性：精度达标时几乎没有速度收益） |

---

## 5. 换形状 / 换模型

模型结构在脚本顶部的 `CFG` 里：

```python
CFG = dict(D=256, NH=8, L=8, FF=1280, MAXLEN=64)
```

已实测可用范围（4090）：`D 128~512`、`NH 4~8`（`D/NH` 需整除，16/32/64 都测过）、
`FF 512~2048`、`L 2~8`、`S 32~128`。改完直接重训，不需要改引擎。

---

## 6. 从 PyTorch 迁过来

同一份代码，只换导入：

```python
import tgai_go.train as T      # 原来是 import torch as T
```

其余写法照旧（`T.tensor` / `T.nn.RMSNorm` / `T.optim.AdamW` / `T.functional.*`）。

---

## 7. 出问题怎么查

1. `python -m tgai_go.train.doctor --trace` —— 带完整堆栈的自检；
2. 训练脚本会打印 `语料=… 词表 … 对话 N 条 序列 S`，先确认这三个数对；
3. 数值不对时按上表逐个关融合开关，二分定位。

---

## 8. 精度选择（fp16 / bf16）

| 你的卡 | 建议 | 理由 |
| --- | --- | --- |
| **Ampere 及以后**（A10 / A100 / 30/40 系 / L40） | `set TG_BF16=1` | 8 位指数、动态范围同 fp32；实测 1.24×、loss 偏差 0.009% |
| Volta（V100） | `set TG_FP16=1` | **没有 BF16 硬件**（`doctor` 会显示"跳过"）；fp16 实测 1.17×、loss +0.55% |
| 更老的卡 | 不用（保持 fp32） | 无张量核，开了反而慢 |

> 两者都**默认关闭**，且都可以用 `*_MAX_K` / `*_MIN_OUTPUT` 调门槛（默认 K≤1280、输出≥262144）。
> 打开后如果某个 GEMM 数值/形状不合适，会自动回退 fp32 并只警告一次。
> **CUDA Graph 捕获期间会自动走 fp32**（不会让图捕获失败）。

`python -m tgai_go.train.doctor` 会顺带验证 bf16 路径（无硬件的机器会显示"跳过"）。
---

## 9. 一条命令验收（给接手的人）

```bash
python tools/accept.py            # 标准：600 步
python tools/accept.py --quick    # 快速：120 步（改完代码先自检用）
python tools/accept.py --json acc.json
```

它会真跑下面这些，最后打印一张 PASS/FAIL 表：

| 项 | 判据 |
| --- | --- |
| 自检 `doctor` | 12/12 |
| 形状矩阵（4 组 D/NH/FF/L/S） | 全部能训、loss 有限 |
| fp32 N 步 | 报 ms/step 与末 loss |
| bf16（有硬件才做） | loss 偏差 ≤2% 且不慢于 fp32 |
| 显存/内存 | 池峰值 + host 峰值 |

**验收结论 PASS** 才说明这台机器上引擎是好的。无 BF16 硬件的机器会把 bf16 标成 `SKIP` 而不是失败。

---

## 10. 微调真实开源模型（Qwen2.5 / Qwen3）

这一节解决的是「**别人的模型拿来训**」。已验证的两个目标：
**Qwen2.5-0.5B-Instruct**、**Qwen3-0.6B**（含 Qwen3 特有的 `q_norm/k_norm` 与 `head_dim=128`）。

### 10.1 三个文件就够

| 需要什么 | 说明 |
| --- | --- |
| 权重目录 | HF 官方目录（`model.safetensors` + `config.json`）。**必须带 `config.json`** —— `head_dim` / `tie_word_embeddings` 只能从它读 |
| 词表 | 该模型的 `tokenizer.json` |
| 语料 | `jsonl`，每行 `{"prompt": "...", "completion": "..."}`，或 `{"text": "..."}`（整段都算 loss） |

### 10.2 先做数据体检（别跳过）

```bash
python tools/dataset_report.py --data sft.jsonl --tokenizer tokenizer.json --seq-len 256
```

它会给四条**硬性自检**，任一条不过就非零退出。最常拦下的是**截断率**：

> 实测 `sft_big.jsonl` 在 `seq_len=128` 下 **27.6%** 的样本被从回答中间切断；
> 改 `seq_len=256` 降到 **2.1%**。截断会让模型学"说到一半就结束"，
> 而且**在 loss 曲线上完全看不出来**。

### 10.3 开训

```bash
# Qwen2.5-0.5B
python examples/qwen25_sft.py   --safetensors /path/Qwen2.5-0.5B-Instruct   --data sft.jsonl --tokenizer /path/Qwen2.5-0.5B-Instruct/tokenizer.json   --seq-len 256 --steps 600 --batch 4 --lr 2e-5 --warmup 40 --clip 1.0   --holdout 8 --device cuda --save out_qwen25.npz

# Qwen3-0.6B（同样的命令，只换目录；结构会自动识别）
python examples/qwen25_sft.py   --safetensors /path/Qwen3-0.6B   --data sft.jsonl --tokenizer /path/Qwen3-0.6B/tokenizer.json   --seq-len 256 --steps 600 --batch 4 --lr 2e-5 --warmup 40 --clip 1.0   --holdout 8 --device cuda --save out_qwen3.npz
```

开跑后**第一件事是看这两行**（结构不对就别浪费时间）：

```
权重 310 个；结构 L=28 D=1024 NH=16/KV=8 DH=128 FF=3072 V=151936；tie=True；QK_NORM=True
探针基线（训练前，留出集平均）= 3.203617
```

- `DH` 对 Qwen3-0.6B 必须是 **128**（不是 `1024/16=64`）；`QK_NORM=True`；
- 训练结束会打印 `gnorm 统计：中位数 X` —— **中位数远高于 clip 阈值说明学习率偏大**。

### 10.4 实测参考值（同口径，可直接对照）

| 模型 | 数据 | 步数 | 探针 loss | 降幅 |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | persona 56+4，lr 2e-5，B=2 | 60 | 3.1148 → 1.5576 | **−50.0%** |
| Qwen3-0.6B | persona 56+4，lr 2e-5，B=2 | 60 | 3.2036 → 1.5000 | **−53.2%** |

异质大数据（`sft_big` 1992 条）在 128 截断下只有 −14.8% 且随后回退 ——
已定位为**截断 + 学习率**两个因素，扫参见 `docs/FINETUNE_RESULTS.md`。

### 10.5 两个坑（都已修，但要知道）

| 坑 | 症状 |
| --- | --- |
| **补位 token 计入 loss** | loss 掉得飞快但模型不会说话。检查方法：`dataset_report` 里「答案 token 数 > 总 token 数」是不可能的 |
| **`cfg` 写死结构** | `reshape (1,127,1024) -> (1,127,896)`。结构参数必须从检查点派生（`cfg_from_checkpoint`） |

### 10.6 别指望直接导出 `.TG`

`.TG` 头里**没有 `head_dim`** 字段。Qwen3-0.6B 的 `head_dim=128 ≠ 1024/16`，
所以 `export.save_tg` 会**明确拒绝**导出（而不是产出一个能加载、生成全错的模型）。
训练产物先用 `.npz`；要部署 Qwen3 需要扩展 `.TG` 格式并同步改 C 引擎。

### 10.7 长跑要能续（`--resume`）

训练大模型可能几小时，机器重启/被抢占是常态。存档会自动带上**优化器动量**：

```bash
# 第一次：训 600 步后存档
python examples/qwen25_sft.py --safetensors /path/Qwen3-0.6B --data sft.jsonl \
  --tokenizer /path/Qwen3-0.6B/tokenizer.json --seq-len 256 --steps 600 --batch 4 \
  --lr 2e-5 --clip 1.0 --device cuda --save ck600.npz

# 接着训：参数 + AdamW 动量 + 步号一起恢复
python examples/qwen25_sft.py --safetensors /path/Qwen3-0.6B --data sft.jsonl \
  --tokenizer /path/Qwen3-0.6B/tokenizer.json --seq-len 256 --steps 600 --batch 4 \
  --lr 2e-5 --clip 1.0 --device cuda --resume ck600.npz --save ck1200.npz
```

会打印一行确认：

```
续训：从 step 600 继续（参数 310 个；优化器动量已恢复）
```

> 为什么不能只存参数：AdamW 的 `m/v/t` 丢了就是重启动量，续训后的头几步 loss 会跳。
> 这一点有测试守着：`tests/test_resume.py` 断言"3 步 + 存档 + 续 3 步"与
> "一口气 6 步"的末 3 步**损失一致**。
>
> 老存档（只有参数）仍然能加载，只是会提示"未恢复（老存档只含参数）"。

---

## 11. 十行代码训练（`tgai_go.api.train`）

### 11.1 最小用法

```python
from tgai_go import api

r = api.train(weights="Qwen2.5-0.5B", data="my_sft.jsonl",
              lr=5e-6, batch=2, seq_len=256, steps=300, out="ckpt.npz")
print(r.probe_base, r.probe_final, r.drop_pct, r.verdict)
```

返回的 `TrainingResult` 既是 dict 也能按属性取：

| 字段 | 含义 |
| --- | --- |
| `probe_base` / `probe_final` | 留出集探针 loss（开始 / 结束） |
| `drop_pct` | 降幅（%，正数表示下降） |
| `verdict` | `PASS` / `FAIL`（脚本判定：降幅 >=5% 才算学到） |
| `ckpt` | 检查点路径（传了 `out` 才有） |
| `log` | 完整日志（`quiet=False` 时也打到屏幕） |
| `command` / `returncode` | 实际执行的命令与返回码（便于复现） |

### 11.2 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `weights` | — | safetensors 目录（`tokenizer.json` 可自动找） |
| `data` | — | jsonl 路径，每行 `{"prompt","completion"}`（也支持 `{"text"}` 等，见 `tgai_go.train.data`） |
| `method` | `"sft"` | 只支持 `sft`；`pretrain`/`dpo`/`grpo` 抛 `NotImplementedError` 并说明计划 |
| `seq_len` / `batch` / `steps` | 256 / 1 / 300 | 训练形状 |
| `lr` | 5e-6 | V100 实测推荐值（1e-5~2e-5 之间开始发散，见 FINETUNE_RESULTS §10） |
| `holdout` | 64 | 探针留出集。**别低于 64** —— 8 条的结论会被噪声反转（实测） |
| `out` | None | 检查点路径 |
| `tiny` | False | True 时用随机迷你权重（CPU、几秒），先跑通流程 |
| `quiet` | True | False 时把训练日志实时打出来 |

### 11.3 现在**不是**什么

- **不是进程内训练**：v1 通过子进程驱动 `examples/qwen25_sft.py`（已全面验证的那条路），
  所以要求本机有这个仓库目录；
- **不支持 DPO / GRPO / 预训练**：会明确抛错，不假装支持；
- **不做自动调参**：`lr` 等要自己给（推荐值来自实测扫参，不是猜的）。

### 11.4 想直接改内部行为时

API 只是入口；要改训练细节（融合核、梯度裁剪策略、掩码方式）请直接看
`examples/qwen25_sft.py`，那里的每一步都有中文注释与实测判据。

### 11.5 实测（V100，2026-11-01）

```python
r = api.train(weights="qwen25", data="examples/_alpaca_zh.jsonl",
              lr=5e-6, batch=1, seq_len=256, steps=120, holdout=64,
              out="ck_api.npz")
```

实测输出：`probe 1.9695 → 1.7601（−10.6%）verdict=PASS`，1.2 s/step，检查点 6.5 GB。

> **坑（实测）**：同一配置 `batch=2` 在 V100 上 **OOM** —— 因为产线推理服务常驻占用
> 14.8 GB，batch2/seq256 需要约 16.5 GB，正好卡线。显存紧张时先 `batch=1`。
