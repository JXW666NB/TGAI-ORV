# -*- coding: utf-8 -*-
"""0.9.32 发版：分组 GEMM 从"必然崩溃"修成"失败即安全回退"。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.32"
ENTRY = """## [0.9.32] - 2026-11-11

修掉分组 GEMM 的**必然崩溃**，并把它改成**失败即安全回退**。性能收益尚未拿到，如实记录。

### 修了什么

`cuda.py::_grouped_ready()` 加载的是**非 Ex** 版 `cublasSgemmGroupedBatched`，但 `argtypes`
按 Ex 版声明成了指针数组：

```
真签名：... int m, int n, int k, alpha, A[], lda, B[], ldb, beta, C[], ldc, batchCount, groupSize[]
旧声明：... int* m, int* n, int* k, alpha, ...
```

驱动把指针低 32 位当 `m` 读 ⇒ 天文数字 ⇒ **必然非法访存（rc=700）**。
旧注释把它归因为"某专家行数 >64 越界"，是误判 —— 只要分组路径被走到就必崩。

现在：
- `argtypes` 改为正确的标量版；
- `_grouped_sgemm` 先检查**所有组 m/n/k 是否一致**（非 Ex 版的硬要求），不一致就安全回退；
- 任何非 0 返回码都**静默回退**到逐次 `sgemm`，只提示一次，绝不抛异常。

### 实测（`tools/grouped_fix_check.py`）

| 场景 | 修复前 | 修复后 |
| --- | --- | --- |
| 形状一致，走分组 | **rc=700 崩溃** | rc=7（参数校验不过）⇒ 自动回退 |
| 形状不一致 | **rc=700 崩溃** | 回退，**0.00e+00 逐位一致** |
| 库的可靠性 | 一调就崩 | **永不崩、结果永远正确** |

### 未达成的目标（如实）

**MoE 反向 332 ms → ≤200 ms 这一条没做到。** 分组路径现在返回 `rc=7
"parameter number 2 had an illegal value"`（第 2 个参数，即 `transa` 位置），
说明这个 libcublas 版本的非 Ex 版符号**参数布局与文档不一致**，还需要进一步定位
（怀疑：该版本要求 `m/n/k` 在 `transa/transb` 之前，或 `alpha/beta` 需设备指针）。

在调通之前，性能仍是逐专家循环的老路径，**不做任何性能宣称**。

### 顺带确认的事实

- 运行时 libcublas **只有** `cublasSgemmGroupedBatched`，没有 `cublasGemmGroupedBatchedEx`
  与 `cublasSgemmGroupedBatchedEx`（`tools/grouped_ex_probe.py` 实测），所以方案 B（Ex 版）
  在这台机器上不可用。

套件：**567 passed / 28 skipped**。
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
                k = z.read("tgai_go/train/cuda.py").decode("utf-8", "replace")
                checks = {
                    "标量签名": "ctypes.c_int,             # m" in k,
                    "一致性检查": "uniform = all(" in k,
                    "失败即回退": "_grouped_fail_note" in k,
                }
                for kk, hit in checks.items():
                    print("   %-14s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
