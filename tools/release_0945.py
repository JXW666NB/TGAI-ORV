# -*- coding: utf-8 -*-
"""0.9.45：把 torch-compat 示例随包分发（0.9.44 的 wheel 里没有它）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.45"
ENTRY = """## [0.9.45] - 2026-11-11

**修正 0.9.44 的分发缺口**：`torch-compat` 示例插件原来只放在 `examples/plugins/`，
**sdist 与 wheel 都没带上** —— `pip install tgai-orv` 的用户拿不到，
这跟"示例要能让所有人跑起来"的初衷矛盾。

已把插件**移入包内**：`tgai_go/train/plugins/torch_compat.py`，随 wheel 与 sdist 一起分发。

```python
import tgai_go.train.plugins.torch_compat   # 导入即登记（只登记，不安装）
import tgai_go.train as T
T.plugin.activate("torch-compat")           # 显式激活
import torch                                # 拿到的其实是 ORV
```

`examples/plugins/` 保留纯算子示例（softplus / gelu_exact / swiglu / squared_relu）。

### 过程中撞到的一条项目纪律

用 PowerShell 的 `Set-Content -Encoding UTF8` 创建 `__init__.py` 时写入了 **BOM 与 CRLF**，
被项目既有的卫生检查（`test_hygiene.py` / `test_cpu.py`）立刻抓红：
"BOM 会让 Python 报 invalid non-printable character"。
已改用无 BOM + LF 重写。**写 Python 源文件不要用 PowerShell 的 Set-Content。**

套件：**635 passed / 33 skipped**。
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
        print(r.stdout[-400:], r.stderr[-400:])
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
                names = z.namelist()
                meta = [n for n in names if n.endswith("METADATA")][0]
                ok = ok and (("Version: %s" % VER) in z.read(meta).decode("utf-8", "replace"))
                checks = {
                    "wheel 含示例插件": "tgai_go/train/plugins/torch_compat.py" in names,
                    "wheel 含 plugins/__init__": "tgai_go/train/plugins/__init__.py" in names,
                }
                for kk, hit in checks.items():
                    print("   %-22s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
