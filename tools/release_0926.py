# -*- coding: utf-8 -*-
"""0.9.26 发版：bf16 成为可编程 AMP dtype。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.26"

ENTRY = """## [0.9.26] - 2026-11-10

`bf16` 接入可编程 AMP（此前只有 `TG_BF16` 环境变量）。

### 变化

- `T.amp(dtype="bf16")` / `T.amp_on(dtype="bf16")` / `T.amp_off(dtype="bf16")` 均可用了。
- `T.amp_state()` 现在分别报告 `fp16` 与 `bf16` 两个开关。
- 退出上下文时**分别恢复**两个开关：原来开 fp16 就恢复 fp16，原来关着就保持关。

### 实现说明（不是新写内核）

GPU 侧 `_sgemm` 早就有 bf16 快路径（`_bf16_wanted` → `cublasGemmEx(CUDA_R_16BF, COMPUTE_32F)`），
只是入口只有环境变量。本次把它接到公开 API，并保留原有的**失败即回退**语义：
任何 cuBLAS 错误都会关掉 bf16 并只警告一次，之后的 GEMM 走原路径（fp16 或 fp32）。

### 实测（V100）

`tools/bf16_check.py`：

```
amp 状态: bf16=True fp16=False
结果与 fp32 **完全一致** ⇒ 运行期已回退 fp32
```

**V100 没有 bf16 硬件**，cuBLAS 报错后按设计回退，结果与 fp32 逐位相同 —— 不存在
"既不是 bf16 精度也不是 fp32 结果"的静默算错。**A100/H100 上才会真正加速，这一点我没有硬件实测**，
所以在 README 里只能写"API 已支持"，不能写"bf16 加速已验证"。

### 测试

`tests/test_amp_api.py` 改为断言 bf16 是合法 dtype、非法 dtype 仍报错。
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
        print(r.stdout[-500:], r.stderr[-500:])
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
                api = z.read("tgai_go/train/amp_api.py").decode("utf-8", "replace")
                checks = {
                    "bf16 dtype": '"bf16"' in api,
                    "_bf16_state": "_bf16_state" in api,
                    "分别恢复": 'get("bf16")' in api,
                }
                for kk, hit in checks.items():
                    print("   %-12s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
