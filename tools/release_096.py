# -*- coding: utf-8 -*-
"""阶段 C 发版：0.9.6（conv）。升版本 → CHANGELOG → docs → 构建 → 拷到 publish_ready。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.6"

ENTRY = """## [0.9.6] - 2026-11-09

阶段 C：卷积算子（CPU）。

### 新增

- `T.conv2d(x, weight, bias, stride, padding, dilation, groups)` / `T.conv1d(...)`，含反向
  （对输入、权重、偏置都正确）。
- `T.Conv2d` / `T.Conv1d` 模块（参数名与 torch 一致：weight/bias）。
- 实现方式：**主机端 im2col + matmul**；`Conv1d` 用可微 reshape 视图委托给 2D 路径。

### 实测（对 torch.nn.functional）

| 配置 | 前向 | dx | dw | db |
| --- | --- | --- | --- | --- |
| stride=1 / 2 | ≤1e-5 | ≤1e-5 | ≤1e-5 | ≤1e-5 |
| padding=1（配合 stride 1/2）| ≤1e-5 | ≤1e-5 | ≤1e-5 | ≤1e-5 |
| dilation=2 | ≤1e-5 | ≤1e-5 | ≤1e-5 | ≤1e-5 |
| groups=2 | ≤1e-5 | ≤1e-5 | ≤1e-5 | ≤1e-5 |
| bias=False | ≤1e-5 | ≤1e-5 | ≤1e-5 | — |
| Conv1d | ≤1e-5 | ≤1e-5 | ≤1e-5 | ≤1e-5 |

### 未完成（如实标注）

- **CUDA 卷积未实现**：im2col 在主机侧构造，GPU 路径需要新的 im2col/col2im 内核。
  当前传 CUDA 张量会抛 `NotImplementedError`（含"修法"），**不会静默算错**；
  测试覆盖了这条报错路径。因此目标 C2 的"GPU 也测"一条**未达成**。
- 性能：主机 im2col 比 cuDNN 慢，大 batch/大 kernel 时差距更明显。
- `groups` 走逐组循环实现（正确但非融合）。
"""

DOCS = """
## 37. 阶段 C：Conv1d / Conv2d（0.9.6，2026-11-09）

### 37.1 实现

`tgai_go.train.conv`：主机端 im2col + matmul，反向手写（dW = gᵀ·cols、dcols = g·W、dx = col2im）。
`Conv1d` 通过**可微 reshape 视图**（`[N,C,L] → [N,C,1,L]`）复用 2D 路径，梯度自动回传。
模块 `T.Conv2d` / `T.Conv1d` 的参数命名与 torch 一致。

### 37.2 实测（`tests/test_conv.py`，判据 1e-5）

stride 1/2、padding 1、dilation 2、groups 2、bias=False、Conv1d ——
**前向与 dx/dw/db 全部 ≤1e-5**（9 项参数化测试全过）。

### 37.3 未达成项（不掩盖）

**GPU 卷积未实现**。im2col 在主机侧构造，要让 GPU 用上必须新写 im2col/col2im 内核；
当前 CUDA 张量会得到 `NotImplementedError`（含修法），并有测试守住"明确报错而不是算错"。
所以目标 C2 中「GPU 也测」这一条**没做到**，只做到 CPU。
性能上主机 im2col 明显慢于 cuDNN，docs 与 docstring 都写明了。
"""


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def main():
    py = os.path.join(ROOT, "pyproject.toml")
    write(py, re.sub(r'(?m)^version\s*=\s*"[^"]+"', 'version = "%s"' % VER, read(py), count=1))
    ini = os.path.join(ROOT, "python", "tgai_go", "__init__.py")
    write(ini, re.sub(r'(?m)^(\s*__version__\s*=\s*")[^"]+(")', r"\g<1>%s\g<2>" % VER, read(ini)))
    print("版本字面量:", re.findall(r'(?m)^\s*__version__\s*=\s*"([^"]+)"', read(ini)))

    cl = os.path.join(ROOT, "CHANGELOG.md")
    txt = read(cl)
    if "[%s]" % VER not in txt:
        i = txt.find("## [")
        write(cl, (ENTRY + txt) if i < 0 else (txt[:i] + ENTRY + txt[i:]))
        print("CHANGELOG 已插入", VER)

    doc = os.path.join(ROOT, "docs", "TRAIN_V27_ORV.md")
    d = read(doc)
    if "## 37." not in d:
        write(doc, d + "\n" + DOCS)
        print("docs 已追加 §37")

    for name in ("dist", "build"):
        p = os.path.join(ROOT, name)
        if os.path.isdir(p):
            shutil.rmtree(p)
    for f in glob.glob(os.path.join(ROOT, "python", "*.egg-info")):
        shutil.rmtree(f, ignore_errors=True)

    r = subprocess.run([sys.executable, "-m", "build", "--outdir", "dist"], cwd=ROOT,
                       capture_output=True, text=True)
    print("build rc=%d" % r.returncode)
    if r.returncode:
        print(r.stdout[-800:], r.stderr[-800:])
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
                blob = b"".join(z.read(n) for n in z.namelist()
                                if n.startswith("tgai_go/train/") and n.endswith(".py"))
                for k, needle in {"conv2d": b"def conv2d", "Conv1d": b"class Conv1d",
                                  "index_select": b"def index_select"}.items():
                    print("   %-13s 包含=%s" % (k, needle in blob))
                    ok = ok and (needle in blob)
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
