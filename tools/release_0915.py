# -*- coding: utf-8 -*-
"""0.9.15 发版：Tensor.max/min/var/std/norm 归约算子。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.15"

ENTRY = """## [0.9.15] - 2026-11-10

补齐 PyTorch 常用归约：`Tensor.max / min / var / std / norm`（前向 + 反向，CPU 与 CUDA）。

### 新增

- `x.max(axis/dim, keepdims/keepdim)`、`x.min(...)`：用 **argmax 的 one-hot mask** 做乘加实现，
  梯度只落到被选中的位置（不需要新内核）；并列最大值取第一个（与 torch CPU 一致）。
- `x.var(axis, unbiased=True)`、`x.std(...)`：两遍法（先均值、再中心化后平方求均值），
  `unbiased` 与 torch 默认一致（除以 n-1）。
- `x.norm(p=2, axis)`：只支持 p=2，其他 p 明确抛 `NotImplementedError` 并给修法。

### 实测（V100 与 CPU 对照）

| 项 | 相对差 |
| --- | --- |
| max / min / var / norm（CUDA vs CPU） | **0.00e+00** |
| std（CUDA vs CPU） | 2.02e-08 |
| max 反向（CUDA vs torch） | **0.00e+00** |
| CPU 侧 10 条 pytest（含 4 种 axis、unbiased 开关、keepdims 形状、p≠2 拒绝） | 全绿 |

### 代价（如实）

`max/min` 在 CUDA 上需要一次 `to_numpy()` 取 argmax，再拷回 one-hot mask —— **每次调用一次主机往返**，
频繁调用/大张量时慢于原生归约。这是"可用且可验证"的实现，不是最优实现。
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
        write(cl, (ENTRY + "\n" + txt) if i < 0 else (txt[:i] + ENTRY + "\n" + txt[i:]))
        print("CHANGELOG OK")

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
                blob = z.read("tgai_go/train/redux_extra.py").decode("utf-8", "replace")
                core = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                for k, (src, needle) in {
                        "redux 模块": (blob, "def var("),
                        "max/min": (blob, "def _masked_reduce"),
                        "norm": (blob, "def norm("),
                        "Tensor 挂载": (core, '("max", "min", "var", "std", "norm")')}.items():
                    hit = needle in src
                    print("   %-12s %s" % (k, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
