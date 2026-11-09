# -*- coding: utf-8 -*-
"""阶段 D 发版：0.9.7（DataLoader）。升版本 → CHANGELOG → docs → 构建 → publish_ready。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.7"

ENTRY = """## [0.9.7] - 2026-11-09

阶段 D：数据加载抽象。

### 新增（`tgai_go.train.loader`，导出为 `T.DataLoader` 等，并挂到 `T.data.*`）

- `Dataset` / `IterableDataset` / `Sampler` / `SequentialSampler` / `RandomSampler` / `BatchSampler`
- `default_collate`：numpy 数组堆叠、标量转数组、dict/list 递归，字符串原样保留
- `DataLoader(dataset, batch_size, shuffle, sampler, drop_last, collate_fn, num_workers, seed)`

### 实测（`tests/test_loader.py`，对 `torch.utils.data.DataLoader`）

| 项 | 结果 |
| --- | --- |
| 顺序批次（bs=1/2/3 × drop_last True/False）| 内容与顺序**逐元素相同** |
| 显式索引分组 | 与 torch 的 `batch_sampler` **完全一致** |
| `len()`（bs=1/2/3/7/10 × drop_last）| 与 torch 相同 |
| `shuffle=True` | 是合法排列（不重不漏）、同 seed 可复现、不同 seed 不同 |
| `num_workers=4` vs `0` | 批次完全相同 |
| `default_collate`（dict + ndarray + 标量）| 与 torch 一致 |
| 空数据集 / 流式数据集 / 错误提示 | 覆盖（报错含"修法"）|

### 与 torch 的差异（如实标注）

- `shuffle=True` 用本库自己的 `np.random.Generator`，**同一 seed 下的顺序不等于 torch**；
  需要严格复现 torch 顺序时请传 `sampler=`。文档与模块说明都写明了。
- `num_workers>0` 用**线程**而非进程（受 GIL 限制，纯 Python 预处理不会真并行）；
  好处是零拷贝、无序列化、顺序天然稳定。

### 过程记录

本轮我先用 `SimpleNamespace` 覆盖了 `T.data`，而它原本是「数据规范化」模块，直接打挂 12 个
数据层测试；已改为把加载 API **挂到同一个模块对象上**，`T.data` 原有函数保持不变。
"""

DOCS = """
## 38. 阶段 D：DataLoader（0.9.7，2026-11-09）

### 38.1 API

`tgai_go.train.loader`：`Dataset`、`IterableDataset`、`Sampler`、`SequentialSampler`、
`RandomSampler`、`BatchSampler`、`default_collate`、`DataLoader`。
导出为 `T.DataLoader` / `T.Dataset` / `T.IterableDataset`，并挂到 `T.data.*`（不覆盖原数据模块）。

### 38.2 实测（`tests/test_loader.py`，14 项全过）

- 顺序批次：bs=1/2/3 × drop_last True/False，与 torch **逐元素相同**（内容与顺序）
- 显式索引分组：与 torch `batch_sampler` 一致
- `len()`：与 torch 相同（5 种 bs × 2 种 drop_last）
- `shuffle=True`：合法排列、同 seed 可复现、不同 seed 不同
- `num_workers=4` 与 `0` 结果相同；`default_collate` 与 torch 一致
- 空数据集、流式数据集、错误提示（含"修法"）均覆盖

### 38.3 差异与代价（写入 docstring，不隐藏）

- `shuffle=True` 用本库 RNG，**同 seed 顺序 ≠ torch**；要严格复现请传 `sampler=`。
- `num_workers>0` 是**线程池**：受 GIL 限制，纯 Python 预处理不会真并行；换来零拷贝与稳定顺序。

### 38.4 一个自己踩的坑

我先用 `SimpleNamespace` 覆盖 `T.data`，而 `T.data` 本是「数据规范化」模块 ⇒ 12 个数据层测试
立刻挂掉。已改成把加载 API 挂到**同一个模块对象**上，`T.data.normalize_*` 等原有接口不变。
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
    write(ini, re.sub(r'(?m)^(\s*__version__\s*=\s*")[^"]+(")', r"\g<1>%s\g<2>" % VER, read(ini)))
    print("版本:", re.findall(r'(?m)^\s*__version__\s*=\s*"([^"]+)"', read(ini)))

    cl = os.path.join(ROOT, "CHANGELOG.md")
    txt = read(cl)
    if "[%s]" % VER not in txt:
        i = txt.find("## [")
        write(cl, (ENTRY + txt) if i < 0 else (txt[:i] + ENTRY + txt[i:]))
        print("CHANGELOG OK")
    doc = os.path.join(ROOT, "docs", "TRAIN_V27_ORV.md")
    d = read(doc)
    if "## 38." not in d:
        write(doc, d + "\n" + DOCS)
        print("docs OK")

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
        print(r.stdout[-600:], r.stderr[-600:])
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
                for k, needle in {"DataLoader": b"class DataLoader",
                                  "BatchSampler": b"class BatchSampler",
                                  "default_collate": b"def default_collate"}.items():
                    hit = needle in blob
                    print("   %-15s %s" % (k, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
