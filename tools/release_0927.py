# -*- coding: utf-8 -*-
"""0.9.27 发版：重写 README + 新手四步 API（load_corpus / train_tokenizer）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.27"

ENTRY = """## [0.9.27] - 2026-11-10

重写 README，并补上"从零开始训练自己的模型"这条链路的入口 API。

### README 重写

旧 README 还停留在 0.2.0（结论、数据、功能表全部过时）。新版围绕**一个问题**组织：
"我为什么用 ORV 而不用 PyTorch"，并逐条回答最容易被质疑的点：

- 数学对齐（8 步 loss 相对差 1.44e-07；算子级对照表）
- 能否微调热门模型（Qwen2.5-0.5B 与 HF 相对差 6.29e-06，**并明确标注 7B 以上未实测**）
- 能否训练自己的模型（10 行 + 四步流程）
- 速度（4/8/14 层比 PyT 快 1.41~1.74×；T=512 整步 2038→754 ms）
- **为什么省显存而不是省精度**（逐层重算 + fp16 动量；计算全程 fp32；fp16 动量的余弦相似度 1.000000）
- 与 PyT 的**8 条真实差异**（单卡、自定义 forward 不可自动迁移、`max(dim=)` 不返回索引、
  permute 是真拷贝、`to()` 断梯度链、无 compile、AMP 语义不同、无生态包）

### 新增：新手四步 API

```python
text  = T.load_corpus("my_corpus.txt")          # ① 导入语料（纯文本 / .jsonl）
tok   = T.train_tokenizer(text, vocab_size=4000) # ② 在你自己的语料上训分词表
ids   = tok.encode(text)
data  = T.tokens(ids, seq=128)                   # ③ 切块
model = T.preset.transformer(vocab=tok.vocab_size, d_model=256, n_layers=6, n_heads=8)
hist  = T.fit(model, data, steps=2000, lr=3e-4)  # ④ 预训练
```

- `T.load_corpus(path)`：纯文本或 `.jsonl`（读 `text`/`content` 字段）。
- `T.train_tokenizer(text, vocab_size)` → `T.Tokenizer`（有 `.vocab_size` / `.encode()`）。
- `examples/beginner_pipeline.py` 实跑（纯 CPU，约 140 秒）：**loss 5.5404 → 3.4252（62%）**。

### 补上缺失的顶层导出

`T.SftDataset` / `T.iter_jsonl_texts` / `T.ppo` 之前只在子模块里，README 写 `T.ppo.gae` 会 AttributeError。
现在都在顶层了，并加了 `tests/test_beginner_pipeline.py`（5 条）断言"README 里点名的 API 必须能点出来"。

### 一件必须说清楚的事（没写进卖点）

"V100 单卡 32GB 能全参 AdamW 训 3B" 这条**目前没有跑通实测**：
2.97B 的 fp32 梯度本身就要 11.1 GiB，加上权重 11.39 + fp16 动量 11.1 ⇒ 已超 31.86。
实测确实 OOM。README 里把它标成"尚未验证"，并给出算术，**不当卖点**。

套件：**558 passed / 28 skipped**。
"""


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def main():
    write(os.path.join(ROOT, "pyproject.toml"),
          re.sub(r'(?m)^version\s*=\s*"[^"]+"', 'version = "%s"' % VER,
                 read(os.path.join(ROOT, "pyproject.toml")), count=1))
    ini = os.path.join(ROOT, "python", "tgai_go", "__init__.py")
    write(ini, re.sub(r'(?m)^(\s*__version__\s*=\s*")[^"]+(")',
                      r"\g<1>%s\g<2>" % VER, read(ini)))
    print("版本:", re.findall(r'(?m)^\s*__version__\s*=\s*"([^"]+)"', read(ini)))

    cl = os.path.join(ROOT, "CHANGELOG.md")
    txt = read(cl)
    if "[%s]" % VER not in txt:
        i = txt.find("## [")
        write(cl, (ENTRY + "\n" + txt) if i < 0 else (txt[:i] + ENTRY + "\n" + txt[i:]))
        print("CHANGELOG OK")

    for name in ("dist", "build"):
        q = os.path.join(ROOT, name)
        if os.path.isdir(q):
            shutil.rmtree(q)
    for f in glob.glob(os.path.join(ROOT, "python", "*.egg-info")):
        shutil.rmtree(f, ignore_errors=True)
    r = subprocess.run([sys.executable, "-m", "build", "--outdir", "dist"], cwd=ROOT,
                       capture_output=True, text=True)
    print("build rc=%d" % r.returncode)
    if r.returncode:
        print(r.stdout[-500:], r.stderr[-500:])
        return r.returncode
    pr = os.path.join(ROOT, "publish_ready")
    if os.path.isdir(pr):
        shutil.rmtree(pr)
    os.makedirs(pr)
    files = glob.glob(os.path.join(ROOT, "dist", "*"))
    for f in files:
        shutil.copy2(f, pr)
    ok = True
    for f in files:
        if f.endswith(".whl"):
            with zipfile.ZipFile(f) as z:
                meta = [n for n in z.namelist() if n.endswith("METADATA")][0]
                ok = ok and (("Version: %s" % VER) in z.read(meta).decode("utf-8", "replace"))
                pre = z.read("tgai_go/train/preset.py").decode("utf-8", "replace")
                core = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                readme = z.read("tgai_orv-%s.dist-info/METADATA" % VER).decode("utf-8", "replace") \
                    if ("tgai_orv-%s.dist-info/METADATA" % VER) in z.namelist() else ""
                checks = {
                    "train_tokenizer": "def train_tokenizer(" in pre,
                    "load_corpus": "def load_corpus(" in pre,
                    "Tokenizer": "class Tokenizer(" in pre,
                    "导出 SftDataset": "SftDataset" in core,
                    "导出 ppo": "import ppo as ppo" in core,
                    "README 新": "我为什么用 ORV" in readme,
                }
                for kk, hit in checks.items():
                    print("   %-16s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
