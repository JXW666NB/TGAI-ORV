# -*- coding: utf-8 -*-
"""把「训练完 → 导出 .TG → 用内置 TGAI GO 推理」一节插进 skill，并重排后续编号。"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "skills", "orv-engine-dev", "SKILL.md")
s = io.open(P, encoding="utf-8").read()

SECTION = """## 8. 训练完之后：用内置 TGAI GO 推理

ORV 仓库自带 TGAI GO 推理侧（`tgai.py` / `tgai.bat` + C 引擎），训完的模型可以**直接跑推理**，
不需要转成别的框架格式。

### 第 1 步：导出成 `.TG`

```python
from tgai_go.train.export import save_tg

save_tg("my_model.TG", params,          # params: {训练侧名字: numpy 数组}
        n_layers=28, d_model=1024, n_heads=16, d_ff=4096,
        vocab_tokens=tok.vocab, merges=tok.merges,
        max_seq=512, n_experts=8, n_activated=2, rope_theta=10000.0)
```

`params` 用**训练侧命名**（`experts.{e}.gate/up/down`）；导出时会映射成引擎的 `w1/w2/w3`。

### 导出时三个会让模型"能加载但生成崩"的坑

| 坑 | 症状 | 处理 |
| --- | --- | --- |
| **RoPE 的 cos/sin 没填** | 加载正常，`generate` 时 `access violation reading 0x…` —— 引擎**按指针直接读**这块 buffer | `collect()` 已按公式补齐，别绕过它自己拼 cfg |
| **`head_dim != d_model // n_heads`** | **导出必错且不报错**，引擎按 `d_model//n_heads` 解析 | `.TG` 头里**没有 head_dim 字段**，表达不了这种结构。要么改结构，要么换格式 |
| **RoPE theta 与训练侧不一致** | 位置编码错位，输出乱 | 训练用什么 theta，导出就传什么 |

### 第 2 步：用 GO 跑起来

```bash
tgai.bat models                          # 列出可用模型（序号 + 大小）
tgai.bat chat -m my_model.TG             # 直接对话
tgai.bat server -m my_model.TG           # 只开 OpenAI 兼容服务（默认 8080）
tgai.bat server -m my_model.TG --port 11435
tgai.bat chat --server 8080              # 对话 + 同进程开服务
tgai.bat bench                           # 基准测试
```

选择计算后端（加 `--dll`）：

```bash
tgai.bat chat --dll cpu\\build\\libtgai_engine.dll              # 纯 CPU
tgai.bat chat --dll cpu\\build\\libtgai_engine_gpubridge.dll    # GPU 桥
```

对话内命令：`/help` `/params` `/model <路径>` `/server [端口]` `/env` `/stats` `/clear` `/exit`

### 已有 `.pt` / `.gguf` 想转 `.TG`

```bash
tgai.bat convert <args>
```

细节见 `docs/GGUF_TO_TG.md`、`docs/TG_V4_FORMAT.md`。

### 交付前必须验证的三件事

1. **加载不崩**：`tgai.bat models` 能列出来
2. **生成不崩**：问一句话能出完整回复（RoPE buffer 缺失在这一步才暴露）
3. **数值对得上**：同一 prompt 下，训练侧前向的 logits 与引擎输出的 top-k 一致或高度接近

漏掉第 2 步是最常见的翻车方式 —— 加载成功会让人误以为导出正确。

---

"""

# 插到原 §8「排查手册」之前
anchor = "## 8. 排查手册"
if anchor not in s:
    print("找不到锚点 §8 排查手册")
    sys.exit(1)
if "训练完之后：用内置 TGAI GO 推理" in s:
    print("该节已存在，跳过")
    sys.exit(0)

# 先把 >=8 的编号 +1（从大到小替换，避免连锁）
for n in (10, 9, 8):
    s = s.replace("## %d. " % n, "## %d. " % (n + 1))
s = s.replace(anchor.replace("## 8.", "## 9."), SECTION + anchor.replace("## 8.", "## 9."))

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("已插入 §8 并重排编号")
print("现有章节：")
for m in re.finditer(r"(?m)^## (\d+)\. (.+)$", s):
    print("  %s. %s" % (m.group(1), m.group(2)))
