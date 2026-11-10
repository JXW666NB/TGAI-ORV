# -*- coding: utf-8 -*-
"""0.9.23 发版：masked_fill 在 CUDA 上静默产出 nan（修复）+ 组合算子 CUDA 复核。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.23"

ENTRY = """## [0.9.23] - 2026-11-10

修一个**静默算错**的 BUG：`masked_fill` 在 CUDA 上返回 nan。

### 问题

```python
out = Tensor(np.where(m, value, self.data), ...)   # self.data 在 CUDA 上是 None
```

`self.data` 是 CPU 张量的 numpy 镜像；**CUDA 张量没有它**。于是 `np.where` 拿到 `None`，
产出 object 数组，最终变成 **nan** —— 既不报异常，也不算对。
这类"静默算错"比崩溃危险得多（用户的因果掩码会直接毁掉训练，而且看不出原因）。

### 修法

CUDA 上改成组合实现：`x*(1-m) + v*m`。只用已有算子，梯度自动只在未掩码处流，
不需要新内核。

### 实测（V100）

| 算子（CUDA vs CPU 同实现） | 修复前 | 修复后 |
| --- | --- | --- |
| `masked_fill` | **nan** | **0.00e+00** |
| `sqrt`（组合） | — | 5.92e-08 |
| `tanh`（组合） | — | 2.39e-07 |
| `gelu`（组合） | — | 7.78e-08 |
| `repeat_interleave` | — | 0.00e+00（且与 `np.repeat` 逐位一致） |

### 顺带做的系统排查

新增 `tools/audit_data_leak.py`：扫描所有 Tensor 方法，找出"用了 `self.data` 但没有 CUDA 分支"的。
结果只有 `masked_fill` 一处是真问题（`__init__` / `dtype` / `to` 属于正常用途）。
其余没有显式 CUDA 分支的方法（`sqrt`/`tanh`/`gelu`/`scale`/`repeat_interleave`）都是**组合实现**，
所以在 CUDA 上本来就可用 —— 上面表格是实测确认，不是推断。

`tests/test_masked_fill_cuda.py`（4 条）守住"CUDA 上不得出现非有限值"这条线。
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
                core = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                _a = core.find("def masked_fill")
                _b = core.find("def ", _a + 10)
                body = core[_a:_b] if (_a >= 0 and _b > _a) else ""
                checks = {
                    "cuda 分支": 'device == "cuda"' in body,
                    "组合实现": "self * (1.0 - mt)" in body,
                }
                for kk, hit in checks.items():
                    print("   %-12s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
