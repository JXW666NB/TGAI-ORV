# -*- coding: utf-8 -*-
"""阶段 E 发版：0.9.8（torch 权重互操作）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.8"

ENTRY = """## [0.9.8] - 2026-11-09

阶段 E：与 `torch.nn` 的**权重级**互操作。

### 新增（`tgai_go.train.interop`，导出为 `T.*`）

- `T.from_torch(module)`：torch 层/`nn.Sequential` → ORV 等价层（参数逐位复制）
- `T.to_torch(module)`：反向转换
- `T.state_dict_from_torch(m)` / `T.state_dict_to_torch(o)`：名字**原样保留**的 state_dict 互转
- `T.OrvSequential`：顺序容器；`T.InteropError`：不支持类型的明确报错

支持层：`nn.Linear`、`nn.Embedding`、`nn.RMSNorm`、`nn.Conv2d`。

### 实测（`tests/test_interop.py`，7 项全过）

| 项 | 结果 |
| --- | --- |
| Sequential(Linear, RMSNorm, Linear) 同输入 logits | ≤1e-5 |
| Embedding / Conv2d 单层 logits | ≤1e-5 |
| `to_torch(from_torch(m))` 参数 | **逐位相同** |
| state_dict 键名（`0.weight`/`0.bias`/…）| 双向保持，值逐位相同 |
| 自定义 Module、LayerNorm | 明确 `InteropError`（含"修法"）|

### 已知缺口（不掩饰）

- **`nn.LayerNorm` 不支持**：ORV 目前没有 LayerNorm 实现。遇到会抛 `InteropError`，
  并有测试守住这条路径。这是与 PyT 对齐的**真实缺口**，下一轮补。
- 容器只支持 `nn.Sequential`/`nn.ModuleList`；自定义 `nn.Module` 子类的 forward 无法自动复现。
- 这是**权重级**互操作：不能在 ORV 图里直接调用 torch 层，也不共享 autograd。
"""

DOCS = """
## 39. 阶段 E：torch 权重互操作（0.9.8，2026-11-09）

### 39.1 API

`tgai_go.train.interop`：`from_torch` / `to_torch` / `state_dict_from_torch` /
`state_dict_to_torch` / `OrvSequential` / `InteropError`，均已导出到 `T.*`。

支持层：`nn.Linear`、`nn.Embedding`、`nn.RMSNorm`、`nn.Conv2d`（本库 `T.Conv2d`）。

### 39.2 实测（`tests/test_interop.py`，7 项全过）

```
Sequential(Linear, RMSNorm, Linear) 同输入 logits  ≤1e-5
Embedding / Conv2d 单层 logits                      ≤1e-5
to_torch(from_torch(m)) 参数                        逐位相同
state_dict 键名与值（0.weight/0.bias/…）            双向逐位相同
自定义 Module / LayerNorm                           明确 InteropError（含修法）
```

### 39.3 明确边界

- 只做**权重级**互转；不做运行期算子互操作（ORV 图里不能直接调 torch 层，也不共享 autograd）。
- 容器仅 `nn.Sequential` / `nn.ModuleList`；自定义 forward 不复现。
- **`nn.LayerNorm` 不支持**（ORV 还没有 LayerNorm 实现），有测试守住"明确报错而非算错"。
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
    if "## 39." not in d:
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
                for k, needle in {"from_torch": b"def from_torch", "to_torch": b"def to_torch",
                                  "state_dict_from_torch": b"def state_dict_from_torch"}.items():
                    hit = needle in blob
                    print("   %-21s %s" % (k, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
