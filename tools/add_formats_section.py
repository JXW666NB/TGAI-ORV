# -*- coding: utf-8 -*-
"""给 skill 加「格式互转」与「用各格式当训练检查点」两节。"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "skills", "orv-engine-dev", "SKILL.md")
s = io.open(P, encoding="utf-8").read()

SECTION = """## 9. 格式互转与「拿别的模型来训练」

支持的格式与方向（接口均已在代码中确认）：

| 格式 | 读（当训练起点） | 写（导出） |
| --- | --- | --- |
| **`.TG`**（TGAI 自有） | ✅ `checkpoints.load_tg` | ✅ `export.save_tg` |
| **GGUF** | ✅ `checkpoints.load_gguf`（仅 F32/F16/BF16） | 见 `tgai-to-gguf` skill |
| **`.safetensors`**（HF 目录） | ✅ `checkpoints.load_safetensors_dir` | — |
| **`.pt`**（torch） | ✅ `interop.from_torch` / `state_dict_from_torch` | ✅ `interop.to_torch` |
| **`.npz`** | ✅ `checkpoints.load_npz` | ✅ `checkpoints.save_npz` |
| **ORV 自己的检查点** | ✅ `ckpt.load_checkpoint` | ✅ `ckpt.save_checkpoint` |

### 9.1 `.pt` / `.gguf` → `.TG`

```bash
tgai.bat convert <args>          # 实际调用 tools/tgai_convert.py
```

命令行细节见 `tgai.bat convert --help`、`docs/GGUF_TO_TG.md`。

Python 侧也可以自己走：

```python
from tgai_go.train.export import save_tg
save_tg("out.TG", params, n_layers=…, d_model=…, n_heads=…, d_ff=…,
        vocab_tokens=vocab, merges=merges, max_experts…)
```

### 9.2 `.TG` → GGUF

走 `tgai-to-gguf` 这套流程（llama.cpp 侧转换）。要点与坑（乱码/循环）见该 skill 与
`docs/GGUF_TO_TG.md`。

### 9.3 把 `.TG` 当训练检查点（继续训练 / 微调）

```python
from tgai_go.train import checkpoints

ck = checkpoints.load_tg("base.TG")            # → Checkpoint（权重 + 结构配置）
# 只想拿几层对拍、省内存时：
ck = checkpoints.load_tg("base.TG", max_layers=4, vocab_limit=32768)
```

`load_tg` 会做 **norm 合理性检查**；确认无误后可传 `allow_suspicious_norms=True` 跳过。

### 9.4 把 GGUF 当训练检查点

```python
ck = checkpoints.load_gguf("model.gguf")
```

**限制（会明确报错，不是静默失败）**：

- **只有 F32 / F16 / BF16 能直接读**；量化类型（Q4_K 等）会抛出 `CheckpointError` 并给出修法
- 需要 `pip install gguf`

### 9.5 把 `.pt` 当训练检查点

```python
import tgai_go.train as T
from tgai_go.train import interop

mine = interop.from_torch(torch_model)          # 结构 + 权重一起搬
# 或只搬权重：
mine.load_state_dict(interop.state_dict_from_torch(torch_sd))
```

`from_torch` 按层类型逐个翻译（`Linear`/`Sequential`/`ModuleList`…）；
遇到它不认识的类型会**报错并列出支持的类型** —— 这是故意的，猜会静默算错。

### 9.6 ORV 自己的检查点（训练中断续跑）

```python
from tgai_go.train import ckpt
ckpt.save_checkpoint(dir, tr, st, opts, head_opt, names, src_tg, step, cfg)
ckpt.load_checkpoint(dir)
ckpt.list_checkpoints(...)        # 配合 rotate() 做滚动保留
```

它会同时保存优化器状态（`opt_fp16mv.save_opt_states`），所以续跑能接上动量。

### 9.7 迁移别人的模型来训练：三个必须核对的点

1. **名字映射**：GGUF 有 `map_gguf_name`、HF 有 `hf_name_to_ours`，都集中在 `checkpoints.py`，
   改架构只改映射表。
2. **Linear 布局**：`.TG` 里有 permute 约定，`checkpoints` 提供 `_tg_perm` / `fix_linear_layout`
   自动修正 —— 但**必须跑一次前向对拍**确认，别信"加载成功"。
3. **norm sanity**：`norm_sanity()` 会检查 norm 层权重是否离谱。跳过了就可能训出一个发散模型。

---

"""

anchor = "## 9. 排查手册"
if anchor not in s:
    print("找不到锚点 ## 9. 排查手册")
    sys.exit(1)
if "格式互转与「拿别的模型来训练」" in s:
    print("已存在，跳过")
    sys.exit(0)

# 从大到小重排 >= 9 的编号
for n in (11, 10, 9):
    s = s.replace("## %d. " % n, "## %d. " % (n + 1))
s = s.replace("## 10. 排查手册", SECTION + "## 10. 排查手册")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("已插入 §9")
for m in re.finditer(r"(?m)^## (\d+)\. (.+)$", s):
    print("  %s. %s" % (m.group(1), m.group(2)))
