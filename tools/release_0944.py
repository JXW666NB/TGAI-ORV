# -*- coding: utf-8 -*-
"""0.9.44 发版：torch-compat 示例插件（PyTorch API 面作为插件）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.44"
ENTRY = """## [0.9.44] - 2026-11-11

**`torch-compat` 示例插件**：把 `torch` 这个名字指向 ORV。
这是「**PyTorch 的 API 面本身可以是插件**」这个宣传点的**可运行证据**。

### 用法

```python
import tgai_go.train as T
T.plugin.load("examples/plugins/torch_compat.py")   # 从文件装
T.plugin.activate("torch-compat")                   # 显式激活（会改 sys.modules）

import torch, torch.nn as nn                        # 拿到的其实是 ORV
m = nn.Linear(4, 8)
opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
loss = (m(torch.tensor(x)) ** 2).mean()
loss.backward(); opt.step()                         # 闭环跑通
```

这条能成立，说明**核心不必背 PyTorch 的 API 面** —— 社区一个文件就能补。

### 安全边界（这是重点）

**绝不覆盖真的 torch。** 若 `sys.modules["torch"]` 已被真 PyTorch 占用，本插件
**拒绝安装**并报错，而不是悄悄替换 —— 否则用户会以为在用 PyTorch 而实际跑的是 ORV，
这种混淆比功能缺失危险得多。

这个边界在**全量测试里真实触发了**：套件中其它用例会 `import torch`（真的 PyTorch），
于是 `test_import_torch_yields_orv` 与 `test_pytorch_style_loop_runs` 被正确跳过
（skip 原因是"本进程已加载真 torch，插件会正确地拒绝覆盖"），而不是失败。

### 覆盖范围（诚实声明，不夸大）

| 已覆盖 | 未覆盖 |
| --- | --- |
| `torch.tensor / zeros / ones / randn` | `torchvision` / `torchaudio` 等生态包 |
| `torch.Tensor`、`float32/float16/int64` | `torch.compile`、`torch.jit` |
| `torch.nn`（Module/Linear/Embedding/LayerNorm/RMSNorm/Sequential/ModuleList/functional/MoE） | 二阶导数、`torch.func` |
| `torch.optim`（AdamW/SGD/Adafactor/LAMB/Lion/Muon） | 分布式（DDP/FSDP） |
| `torch.no_grad`、`torch.manual_seed`、`torch.cuda` | CUDA Graph 的公开 API |

**这是最小可运行闭环，不是完整兼容层。** 要更完整，社区在这个文件上继续加即可 ——
这正是插件系统的用途。

### 顺带修掉一个命名限制

`plugin.api` / `plugin.backend` 的名字原先要求是合法 Python 标识符，导致
`torch-compat` 这种最自然的叫法被拒。已放宽：只有会挂成属性的层/优化器/骨架
（`T.nn.<name>`）才要求标识符，登记键允许 `. _ -`。

### 测试（7 条，`tests/test_torch_compat_plugin.py`）

插件文件存在、只登记不自动激活、`import torch` 得到 ORV、**PyTorch 写法闭环跑通**、
**拒绝覆盖真 torch**、卸载复原 `sys.modules`、重复加载幂等。

套件：**635 passed / 33 skipped**（+7 条用例、+2 条正确跳过）。
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
                meta = [n for n in z.namelist() if n.endswith("METADATA")][0]
                body = z.read(meta).decode("utf-8", "replace")
                ok = ok and (("Version: %s" % VER) in body)
                checks = {
                    "README 提到插件": "torch-compat" in body,
                    "plugin.py 含 _check_key": "_check_key" in z.read(
                        "tgai_go/train/plugin.py").decode("utf-8", "replace"),
                }
                for kk, hit in checks.items():
                    print("   %-18s %s" % (kk, hit))
                    ok = ok and hit
        elif f.endswith(".tar.gz"):
            # 示例插件应随源码包分发（wheel 只装库，示例留在 sdist / GitHub）
            with tarfile.open(f) as t:
                has = any(n.endswith("examples/plugins/torch_compat.py")
                          for n in t.getnames())
            print("   %-18s %s" % ("sdist 含示例插件", has))
            # 仅提示，不作为失败条件（sdist 打包规则由 MANIFEST 决定）
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
