# -*- coding: utf-8 -*-
"""0.9.46 发版：C 端插件优先（原生库 / C ABI / JSON 清单 / NVRTC 现场编译）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.46"
ENTRY = """## [0.9.46] - 2026-11-11

**C 端插件优先**：插件可以是原生库，Python 只是可选外壳。

### 先纠正一个常见误解

「插件是 .py，是不是太慢？」——**.py 从来不是热路径。** 它只在**加载时**执行一次，
做登记与绑定；进入训练循环后调用的是原生代码。所以 `.py` 的问题不是"慢"，
而是"它把插件的**实现**也限制在 Python 里"。本版让实现也能是原生的。

### 四条通道（按原生程度排序）

| 通道 | 形态 | 需要 Python | 热路径 |
| --- | --- | --- | --- |
| `plugin.load_manifest(json)` | 原生库 + **JSON 清单** | ❌ **完全不需要** | 原生 |
| `plugin.op_c(lib, symbol)` | 预编译库 + C ABI | 一行绑定 | 原生 |
| `plugin.op_native(source, entry)` | CUDA C 源码，**ORV 用 NVRTC 现场编译** | 一行绑定 | 原生 |
| `plugin.op(fn)` | 组合已有算子 | 是 | Python 胶水 |

### 实测（真 V100，`tools/native_plugin_check.py`）

```
n=1000      与 numpy 相对差 4.39e-08  ⇒ 一致
n=1048576   与 numpy 相对差 6.94e-08  ⇒ 一致
再次调用未重复编译: True
原生内核 n=1M: 0.0192 ms
```

**NVRTC 现场编译通道已实证可用**，不是"设计好了"。

### 过程中连修三个真 BUG（都在真实 GPU 上暴露）

1. `_compile(source=...)` 收 **str**，我传了 `bytes` → `AttributeError`；
2. `_compile(arch=...)` 收 `"sm_70"` 字符串，我传了整数 `70` →
   `NVRTC: invalid value for --gpu-architecture`；
3. `_compile(want_cubin=True)` 返回的是 **cubin 字节**，不是 CUmodule →
   直接当 module 传会得到 `rc=400 invalid resource handle`；
   必须再 `cuModuleLoadData`。

另外 `launch(args=...)` 要 `(值, ctype)` **对**，不是裸值。

### 关于跨后端可移植性（重要，不是本版能解决的）

C 端插件**天生绑定后端** —— CUDA C 不可能跑在 Vulkan 或国产卡上，这是物理事实。
解法是三档可移植性 + DSL，已写成设计：`docs/PLUGIN_DSL.md`

| 档 | 形态 | 可移植 | 速度 |
| --- | --- | --- | --- |
| A 组合式 | 用已有算子拼 | ✅ 全后端 | 看拼法 |
| B 声明式内核（DSL） | 逐元素/归约/广播描述 | ✅ 全后端自动生成 | 近原生 |
| C 后端原生 | 手写 CUDA C / GLSL / 厂商语言 | ❌ 只该后端 | 原生 |

纪律：**C 档插件必须同时提供 A 或 B 档回退**，否则它在别的后端上应是"不存在"，
而不是"跑出错结果"。DSL 分 S1–S5 五阶段，**S1（逐元素 + CPU/CUDA 两路生成）尚未开工**。

### 安全声明

**加载原生库等于执行任意机器码。** ORV 不做沙箱、不校验来源。只加载信任的库。

### 测试（14 条，`tests/test_plugin_native.py`）

C ABI 包装与缺符号报错、`load_lib` 缓存与缺文件报错、参数声明解析（含越界/坏类型）、
JSON 清单字段校验与端到端绑定、`op_native` 源码校验与无 CUDA 时报错、文档存在性检查。

套件：**649 passed / 33 skipped**（+14）。
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
                pl = z.read("tgai_go/train/plugin.py").decode("utf-8", "replace")
                checks = {
                    "op_native": "def op_native(" in pl,
                    "op_c": "def op_c(" in pl,
                    "load_lib": "def load_lib(" in pl,
                    "load_manifest": "def load_manifest(" in pl,
                    "plugin_manifest 打包": "tgai_go/train/plugin_manifest.py" in names,
                    "cuModuleLoadData 修正": "cuModuleLoadData" in pl,
                }
                for kk, hit in checks.items():
                    print("   %-22s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
