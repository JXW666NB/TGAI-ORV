# -*- coding: utf-8 -*-
"""0.9.48：ParameterGroup 支持 CUDA 偏移视图（BUG #1）+ 根目录整理。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.48"
ENTRY = """## [0.9.48] - 2026-11-11

**`ParameterGroup` 支持 CUDA**（偏移视图），并**修掉导出/移动文件时的两处误用**。

### 之前的状态

0.9.43 只有 CPU 可用，CUDA 直接报错 —— 当时判断是「`GpuBuf` 没有 offset 字段」。
实际不成立：`GpuBuf` 的视图可以**指定任意 `ptr`**，所以偏移视图不需要改 `GpuBuf`。

### 现在

新增 `CudaRuntime.sub_view(buf, shape, elem_offset)`：在底层缓冲内做偏移视图，
共享同一段显存（`view_of` 会挂 `keepalive`、清 `_token`，原缓冲先死不会被误释放），
越界报错不静默。

`ParameterGroup.to_cuda()` 用它把整组搬上设备，各视图落在正确偏移。

### 真 V100 实测（`tools/param_group_gpu_check.py`）

```
1) 偏移视图指针正确: True
2) 视图读到正确切片: True（最大偏差 0.0e+00 / 0.0e+00 / 0.0e+00）
3) 视图参与 GEMM 与独立张量一致: True（最大差 0.0e+00）
4) 原地改底层后视图同步: True
```

### 过程中连踩三个坑（都只有上真卡才暴露）

1. **默认设备是 cuda 时 `Parameter(ndarray)` 立刻建了设备副本**，`.data` 变成 None，
   `h2d(None)` 只得到 1 个元素 ⇒ `sub_view` 报越界。修：构造期间强制 CPU，
   并明确**`self._arr` 是唯一权威主机副本**。
2. **"复用已有的 `packed._gpu`"** 会读到陈旧内容（主机副本已被改）。已删除该优化。
3. **缺少安全写入整组数据的方法** —— `packed.data = ...` 是出式赋值、会断开视图。
   新增 `group.load(array)`：切片原地写、共享不断，已上设备时同步刷新。

### 测试

`tests/test_param_group.py` 增 3 条：`load` 原地写并保持共享、元素数不符报错、
`assert_shared_gpu` 在 CPU 上是无操作。套件：**655 passed / 33 skipped**。

### 附：仓库根目录整理

32 个测量输出（`.txt` / `.log`）原先散在仓库根目录 —— 它们是 README 与
`ROADMAP_3B_PERF.md` 里数字的**原始证据**，所以**移动**到 `docs/measurements/`
并生成索引，而不是删除。`.gitignore` 增加 `/*.txt`、`/*.log` 防止复发。
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
                cu = z.read("tgai_go/train/cuda.py").decode("utf-8", "replace")
                pg = z.read("tgai_go/train/param_group.py").decode("utf-8", "replace")
                checks = {
                    "sub_view 已加": "def sub_view(" in cu,
                    "to_cuda 用 _arr": "np.ascontiguousarray(self._arr" in pg,
                    "load 已加": "def load(self, array)" in pg,
                    "构造期强制 CPU": '_pk._DEVICE = "cpu"' in pg,
                }
                for kk, hit in checks.items():
                    print("   %-18s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
