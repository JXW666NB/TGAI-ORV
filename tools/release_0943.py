# -*- coding: utf-8 -*-
"""0.9.43 发版：共享存储参数原语（T.ParameterGroup）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.43"
ENTRY = """## [0.9.43] - 2026-11-11

**共享存储参数**（`T.ParameterGroup`）：多个参数共用同一块底层缓冲。

### 为什么需要

MoE 专家权重若连续存成 `[G, N, K]`，一次 `SgemmStridedBatched` 就能算完全部专家；
否则每次都要先把 G 份权重复制成连续缓冲 —— 实测打包开销比省下的还多
（`docs/MOE_MODERN_ARCH.md` 3.1：V2.7 慢 1.30×、256 专家慢 2.22×）。
共享存储参数是「权重连续存」的数据模型基础。

```python
g = T.ParameterGroup([(4096, 1024)] * 8)        # 8 个专家权重
y = x @ g.views[3].T                            # 视图可直接用于前向
opt = T.optim.AdamW(g.parameters(), lr=1e-3)    # 优化器只拿底层那一块
```

### 契约（不遵守会静默出错，所以在**构造时**就拦）

1. **只有 `works_inplace=True` 的优化器能用于本组** —— 出式更新
   （`p.data = p.data - ...`）会把 `.data` 换成新数组、**断开视图**，
   于是"权重更新了、视图还是旧值"。用 `group.optimizer_class(name)` 取优化器时
   由 `plugin.require_inplace` 直接拒绝。
2. **视图不能单独交给优化器**：`group.parameters()` 只返回底层那一块；
   `group.assert_optimizer_ok(opt)` 会检查并报错。
3. `group.assert_shared()` 自检视图确实共享内存。

### 平台状态（如实，不含糊）

| 平台 | 状态 |
| --- | --- |
| CPU（numpy） | ✅ **可用**，视图是切片、天然共享 |
| CUDA | ❌ **本版不支持**：`GpuBuf` 支持共享 `ptr` 的视图，但**没有 offset 字段**；做偏移视图需新建 `ptr+offset` 缓冲，会触碰按 ptr 索引的缓存/提交逻辑。**明确报错，不静默返回错误结果。** |

CUDA 支持是「零打包 → 分组 GEMM 收益」的前置条件，属后续工作。

### 测试（8 条，`tests/test_param_group.py`）

视图共享内存、紧密排布、`parameters()` 不含视图、`works_inplace` 强制、
优化器拿到视图被拒、**实证"出式赋值会断开共享"**（用它来证明风险真实存在）、
输入校验、`repr`。

套件：**630 passed / 31 skipped**（+8）。
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
                pg = z.read("tgai_go/train/param_group.py").decode("utf-8", "replace")
                init = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                checks = {
                    "param_group 打包": "tgai_go/train/param_group.py" in names,
                    "ParameterGroup 类": "class ParameterGroup" in pg,
                    "works_inplace 强制": "require_inplace" in pg,
                    "视图检查": "assert_optimizer_ok" in pg,
                    "导出": "ParameterGroup" in init,
                }
                for kk, hit in checks.items():
                    print("   %-18s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
