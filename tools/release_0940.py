# -*- coding: utf-8 -*-
"""0.9.40 发版：MoE 容量因子 + 残差旁路。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.40"
ENTRY = """## [0.9.40] - 2026-11-11

**MoE 支持容量因子与残差旁路**（Switch / GShard / Mixtral / DeepSeek 的工程必配项）。

### 语义

```
C = ceil(α · T · k / N)        每个专家最多接收 C 个 token
```

超过容量的 token 在本层**拿不到该专家的贡献**（MoE 输出记为 0），
但它在 block 里的残差 `x` 仍原样流过去 —— 这就是 **residual bypass**。
Switch / Mixtral / DeepSeek 都是这个做法：丢的是「专家加工」，不是「整个 token」。

多卡 Expert Parallel 必须设（要静态分配缓冲）；单卡可不设。

### 新增 API

```python
T.nn.MoE(..., capacity_factor=1.25)   # 默认 None = 不限制，与旧行为逐位相同
```

### ⚠️ 一个必须知悉的限制（实测发现，已写进代码注释）

**开启容量限制后，CPU 与 CUDA 保留的 token 集合可能不同。**
两边都严格遵守 cap，但"丢哪些"不一致 —— 因为 CPU 的排序序与 CUDA `moe_compact`
的到达序不同。

实测（cap=24，96 token）：

```
截断前 counts = [17, 21, 30, 27, 26, 16, 26, 29]
截断后 counts = [17, 21, 24, 24, 24, 16, 24, 24]     ← cap 生效，两边一致
CPU 有 3 个 token 两个专家全被丢；CUDA 为 0
```

⇒ **用容量限制时请固定用同一个后端。** 要做到跨后端一致，需要按门控权重排序后再截断
（GShard 的优先路由），那需要一次排序内核，本版未做。

### 顺带修掉一个真 BUG

`_clamp_offsets` 里用了 Python 内置 `min`，但 **ORV 在模块命名空间里把 `min`
遮蔽成了张量 API**（`_api_min`），导致在 CUDA 上抛「max/min 尚未实现」。
已加 `_builtin_min = min` 别名（与已有的 `_builtin_max` 对称）。

这个坑值得记下来：ORV 模块内**不能直接用 `min`/`max` 做标量比较**。

### 测试（6 条，`tests/test_moe_capacity.py`）

1. `capacity_factor=None` 与旧行为逐位相同；
2. 容量公式 `ceil(α·T·k/N)` 正确（多组 T/α 交叉验证）；
3. 容量收紧时确实有 token 被丢（MoE 输出整行为 0）；
4. 被丢 token 的反向正常（无 nan 梯度）；
5. **共享专家不受容量限制**（它不参与路由）；
6. `capacity_factor ≤ 0` 报错。

套件：**593 passed / 31 skipped**（+6）。
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
                ok = ok and (("Version: %s" % VER) in z.read(meta).decode("utf-8", "replace"))
                core = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                i0 = core.find("class MoE")
                moe = core[i0:] if i0 >= 0 else ""
                checks = {"capacity_factor": "capacity_factor" in moe,
                          "_capacity": "def _capacity" in moe,
                          "_clamp_offsets": "_clamp_offsets" in moe,
                          "_builtin_min 别名": "_builtin_min = min" in core}
                for kk, hit in checks.items():
                    print("   %-18s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
