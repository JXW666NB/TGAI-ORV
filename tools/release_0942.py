# -*- coding: utf-8 -*-
"""0.9.42 发版：能力模型 + 算子分派（Vulkan 接入的架构前提）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.42"
ENTRY = """## [0.9.42] - 2026-11-11

**能力模型 + 算子分派**：一个插件文件适配全部后端。这是接 Vulkan 的架构前提。

### 机制

插件**针对能力写，不针对后端写**：

```python
plugin.op("my_silu", fast_impl, backend="cuda", caps={"async_copy": True}, min_cc=(8, 0))
plugin.op("my_silu", slow_impl, backend="*")     # 任何后端的通用回退
```

分派按当前后端的**真实能力**（`caps`）与计算能力（`cc`）选实现。
`backend="*"` 的回退参与任何后端，所以 Vulkan 在插件尚未提供专用实现时仍能跑（只是慢），
而不是直接崩。

### 实测分派开销（硬约束的量化）

`dispatch.benchmark_overhead()`（20 万次）：

| 项 | ns/次 |
| --- | --- |
| 直接调用 | 75 |
| 经分派（缓存命中） | 701 |
| **分派净开销** | **约 627** |

| 场景 | 单次耗时 | 分派占比 | 结论 |
| --- | --- | --- | --- |
| 大 GEMM（2048×2048×6144） | 约 400,000 ns | **0.16%** | 可接受 |
| 小 elemwise（[128,4096]） | 约 15,000 ns | **4.2%** | **超标** |

⇒ 因此提供 **`plugin.bind()` 静态绑定**：解析一次后直接调用、零分派开销。
**热路径必须用 bind**，这条写进了 `resolve_op` 的文档、README 与
`docs/PLUGIN_DISPATCH.md`，不是隐藏知识。

### 三条纪律

1. **单次 dict 查表**：键 `(op, backend, dtype, cc_major, cc_minor, caps)`，结果进缓存；
   不做多级 fallback 链、不用反射。
2. **能力不足必须报错**：抛 `DispatchError` 并逐条说明每个候选缺什么
   （缺哪个能力 / 计算能力不满足 / dtype 不符）。
3. **通配回退**保证新后端可用。

### 过程中抓到两个真 BUG

1. **缓存键漏了 `caps`**：先按 `{"async_copy": True}` 解析出快路径，再按 `{}` 查询时
   命中了同一条缓存，**错误地返回快路径**。已把 caps 加入键（字符串化后排序）。
2. **提前返回挡住了通配回退**：只登记了 `backend="*"` 的后端（如 Vulkan）被判为"无实现"。
   已改为先把具体后端与通配两类候选都取到再判空。

### 测试（12 条，`tests/test_plugin_dispatch.py`）

能力门控、计算能力范围、通配覆盖任意后端、dtype 过滤、优先级、缓存命中、
新登记使缓存失效、静态绑定、无实现时报错含原因、开销量化、以及
「条件化实现不进 functional 命名空间 / 无条件算子弟照进」。

套件：**622 passed / 31 skipped**（+12）。
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
                dp = z.read("tgai_go/train/dispatch.py").decode("utf-8", "replace")
                pl = z.read("tgai_go/train/plugin.py").decode("utf-8", "replace")
                checks = {
                    "dispatch.py 打包": "tgai_go/train/dispatch.py" in names,
                    "能力门控": "_caps_ok" in dp,
                    "通配回退": '("*")' in dp or '"*"' in dp,
                    "缓存含 caps": "caps_key" in dp,
                    "plugin.bind": "def bind(" in pl,
                    "README 分派段": "单文件怎么适配全部后端" in body,
                }
                for kk, hit in checks.items():
                    print("   %-18s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
