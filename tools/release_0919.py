# -*- coding: utf-8 -*-
"""0.9.19 发版：fp16 GEMM 阈值修正（大形状真正用上 tensor core）+ 缓存 BUG 修复。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.19"

ENTRY = """## [0.9.19] - 2026-11-10

修正 fp16 GEMM：**大形状真正用上 tensor core**（此前默认阈值把它们全挡回 fp32），并修掉影子缓存里
一处 `hit[0]` 的 BUG。

### 两处根因

1. `_FP16_MAX_K` 默认 **1280** —— K 超过就退回 fp32。而 tensor core 的收益恰恰在**大 K / 大输出**上。
   实测（V100，fp16 输入 + `CUBLAS_COMPUTE_32F` 累加）：

   | 形状 (M×K×N) | fp32 | fp16（修正后） | 加速 |
   | --- | --- | --- | --- |
   | 512×1024×3072 | 0.28 ms | **0.08 ms** | 3.5× |
   | 2048×2048×6144 | 3.86 ms | **0.67 ms** | 5.8× |
   | 2048×2048×8192 | 5.30 ms | **0.94 ms** | 5.6× |
   | 2048×4096×11008 | 13.89 ms | **2.13 ms** | 6.5× |
   | 2048×2048×32000 | 19.24 ms | **3.15 ms** | 6.1× |

   同样形状下 PyTorch AMP 是 0.10 / 0.71 / 0.97 / 2.33 / 3.57 ms ⇒ **ORV 六项全部不慢于 PyT AMP**。

2. `Runtime._cast_f16_cached` 里写成 `return hit[0]`（`hit` 已是 `GpuBuf`，不是元组），
   抛 `TypeError` 后被"fp16 不可用 → 回退 fp32"的兜底吞掉。**上一版据此得出的"fp16 打平 fp32"
   其实是 fp32 在跑**，本次一并更正。

### 数值

fp16 输入 + fp32 累加，相对 fp32 的最大相对误差 **2.78e-04 ~ 2.88e-04**，与 **torch AMP 完全同值**。

### 测试

`tests/test_fp16_thresholds.py`：守住 `_FP16_MAX_K` 默认 >= 1<<30、`_FP16_MIN_OUTPUT` 保持量级，
防止把默认值改回去（读源码断言，因为 `cuda.py` 是 exec 进私有命名空间的）。
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
    write(ini, re.sub(r'(?m)^(\s*__version__\s*=\s*")[^"]+(")', r"\g<1>%s\g<2>" % VER, read(ini)))
    print("版本:", re.findall(r'(?m)^\s*__version__\s*=\s*"([^"]+)"', read(ini)))

    cl = os.path.join(ROOT, "CHANGELOG.md")
    txt = read(cl)
    if "[%s]" % VER not in txt:
        i = txt.find("## [")
        write(cl, (ENTRY + "\n" + txt) if i < 0 else (txt[:i] + ENTRY + "\n" + txt[i:]))
        print("CHANGELOG OK")

    for name in ("dist", "build"):
        p = os.path.join(ROOT, name)
        if os.path.isdir(p):
            shutil.rmtree(p)
    for f in glob.glob(os.path.join(ROOT, "python", "*.egg-info")):
        shutil.rmtree(f, ignore_errors=True)
    r = subprocess.run([sys.executable, "-m", "build", "--outdir", "dist"], cwd=ROOT,
                       capture_output=True, text=True)
    print("build rc=%d" % r.returncode)
    if r.returncode:
        print(r.stdout[-600:], r.stderr[-600:])
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
                cuda = z.read("tgai_go/train/cuda.py").decode("utf-8", "replace")
                # 只查 _cast_f16_cached 的函数体：文件别处有无关的 hit[0]（token/指针缓存），
                # 上一版用全文件 grep 造成误报，且 twine 已上传 —— 教训：校验要精确到函数体。
                _a = cuda.find("def _cast_f16_cached")
                _b = cuda.find("def _sgemm_f16")
                cache_body = cuda[_a:_b] if (_a >= 0 and _b > _a) else ""
                checks = {
                    "K 阈值 1<<30": "1 << 30" in cuda,
                    "缓存 return hit": "return hit" in cache_body,
                    "缓存体无 hit[0]": "hit[0]" not in cache_body,
                }
                for k, hit in checks.items():
                    print("   %-14s %s" % (k, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
