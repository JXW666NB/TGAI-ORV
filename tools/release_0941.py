# -*- coding: utf-8 -*-
"""0.9.41 发版：插件系统（六条轴，一个命名空间）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.41"
ENTRY = """## [0.9.41] - 2026-11-11

**插件系统统一入口：六条轴，一个 `plugin` 命名空间。** 这是对外宣传的卖点之一。

### 为什么现在做

用户提出的关键判断：**如果优化器也是插件，那么「把优化器改成原地更新」就不是动核心，
而是加一个插件。** 这解掉了一个两难 —— 之前为了拿到分组 GEMM 的 1.39×，
需要改优化器的更新方式（出式 → 原地），那会动所有优化器、风险很高；
有了优化器插件，这件事变成「核心提供共享存储参数原语 + 社区写原地优化器」。

### 六条轴

```python
from tgai_go.train import plugin

@plugin.op("softplus")                     # ① 算子（autograd 自动）
def softplus(x): return (x.exp() + 1.0).log()

@plugin.layer("MyBlock")                   # ② 层
class MyBlock(T.nn.Module): ...

@plugin.arch("my_model")                   # ③ 骨架
def build(d=512): return ...

plugin.optimizer("Lion2", Lion2, works_inplace=True)   # ④ 优化器
plugin.backend("vulkan", VulkanBackend, caps={...})     # ⑤ GPU 后端
plugin.api("torch-compat", install_torch_shim)          # ⑥ API 兼容层
```

登记后**立即可用**：`T.build("my_model")` / `T.optim.Lion2` / `T.nn.MyBlock` /
`T.functional.softplus`。

### 三条设计原则

| 原则 | 做法 |
| --- | --- |
| **简易** | 六条轴共用 `plugin` 对象；支持装饰器；**一个 .py 文件装完全部** |
| **通用** | 插件只组合核心算子 ⇒ autograd 自动、同一份代码适配 CUDA/Vulkan/ROCm |
| **安全** | 能力（`caps` / `works_inplace`）必须如实声明，不足时**当场报错** |

第三条是关键：声明为出式更新的优化器**不能**用于共享存储参数 ——
`p.data = p.data - ...` 会把视图换成新数组，导致权重与视图不同步。
这种 bug 极难查，所以注册阶段就拒绝，而不是等训练跑歪。

### 装配顺序（刻意设计）

- 算子 / 优化器 / 层 / 骨架：**纯登记**，注册后立刻可用（无副作用）；
- 后端 / API 兼容层：**只登记不激活**，需显式 `plugin.activate(name)`。

理由是后者会改 `sys.modules` 或切后端 —— 装插件不该在你不知情时改变进程行为。

### 复用已有的 `ops_plugin`，不造第二套

算子轴转发到已有的 `ops_plugin`（0.7.x 起就有）：组合式注册、自带反向、
`entry_points` 发现、契约校验（`api_version` 主版本匹配 + 声明的算子必须真注册 +
`test_inputs` 数值自检）、`manifest()` 带 sha256 可复现。

### 测试（17 条，`tests/test_plugin_system.py`）

含端到端：**一个 .py 文件把六条轴全用上**，然后验证算子真能算、骨架真能建。
另有能力拒绝、重名需 `replace=True`、非法名字/类型报错、`summary()` 单行等。

套件：**610 passed / 31 skipped**（+17）。
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
                body = z.read(meta).decode("utf-8", "replace")
                ok = ok and (("Version: %s" % VER) in body)
                checks = {
                    "plugin.py 打包": "tgai_go/train/plugin.py" in names,
                    "六轴 op": "def op(" in z.read("tgai_go/train/plugin.py").decode("utf-8", "replace"),
                    "README 插件段": "六条轴" in body,
                    "T.build 导出": "def build(" in z.read("tgai_go/train/__init__.py").decode("utf-8", "replace"),
                }
                for kk, hit in checks.items():
                    print("   %-18s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
