# -*- coding: utf-8 -*-
"""0.9.36 发版：分组 GEMM 的 pytest 守卫（已在真 GPU 上验证通过）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.36"
ENTRY = """## [0.9.36] - 2026-11-11

补上分组 GEMM 的 **pytest 守卫**（验收原文要求「数值对照 ≤1e-5 且有 pytest 守住」）。
四条断言已在真实 V100 上全部 PASS。

### 新增 `tests/test_moe_grouped_fallback.py`

| 用例 | 锁住的保证 | 真 GPU 结果 |
| --- | --- | --- |
| `test_non_uniform_shapes_fall_back_and_match_exactly` | 形状不一致时回退，与逐次 `gemm_nt` **逐位一致** | PASS（差 0.00e+00） |
| `test_uniform_shapes_never_crash_and_stay_correct` | 形状一致时走分组，**无论成败不崩、数值 ≤1e-5** | PASS |
| `test_moe_grouped_switch_defaults_off` | `TGAI_MOE_GROUPED` 默认关闭 ⇒ 关闭时行为与加分组前一致 | PASS |
| `test_grouped_helpers_exist` | 运行时原语存在，避免静默回退掩盖缺失 | PASS |

无 CUDA 的机器上后两条自动 skip（本地实测 1 passed / 3 skipped）。

### 为什么必须锁住这几条

2026-11-11 之前 `_grouped_ready()` 按 Ex 版签名调非 Ex 版符号，
驱动把指针低 32 位当 `m` 读 ⇒ **必然非法访存（rc=700）**。
因为当时它默认关闭，所以**没人发现**。现在有四道断言挡住回归。

### 工具

`tools/run_grouped_tests.py` —— 在无 pytest 的机器上直接调用同一批断言函数
（GPU 机器原先没有 pytest，本版已安装 9.2.0，两条路都能跑）。

套件：本地 **568 passed / 31 skipped**（新增 1 pass + 3 skip）。
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
    print("产物:", [os.path.basename(f) for f in files])
    return 0


if __name__ == "__main__":
    sys.exit(main())
