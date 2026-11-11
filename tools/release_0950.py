# -*- coding: utf-8 -*-
"""0.9.50：修 P100 实测暴露的首装 DLL 问题 + 记录 P100 评估。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.50"
ENTRY = """## [0.9.50] - 2026-11-12

**修首装失败（NVRTC 依赖 DLL 找不到）**，并把一份**外部 P100 实测报告**的发现记入文档。

### 修复：`nvrtc-builtins64_*.dll` 找不到

外部用户在 **Tesla P100（sm_60，Windows，只有 PyTorch cu118，无 CUDA Toolkit）** 上首装即失败：

```
CudaError: NVRTC 编译失败 arch=compute_60
nvrtc: error: failed to open nvrtc-builtins64_118.dll.
```

`nvrtc64_*.dll` 与 `nvrtc-builtins64_*.dll` 就在 `site-packages/torch/lib` 下，但进程找不到。
用户的规避办法是「PATH 前置 + `os.add_dll_directory` **两者都做**」。

原来 ORV 只扫 `sys.path/torch/lib` 并调用 `add_dll_directory` —— 不够。现在：

1. 把 pip 装的 CUDA 轮子目录 `site-packages/nvidia/*/lib`、`nvidia/*/bin` 一并纳入搜索；
2. `_prepare_dir()` 除了 `add_dll_directory`，**同时把目录前置进 `PATH`**
   （这正是用户实测有效的做法）。

### 一份重要的外部实测：P100 上 ORV 在 16GB 卡上反而 OOM

同一份报告里还有更重的发现，已完整收入 `docs/REVIEW_P100_2026-11-12.md`：

| 发现 | 影响 |
| --- | --- |
| 400M 稠密模型（20 层/d1280/B8/S2048）**PyTorch 能跑（14.18 GB），ORV 开检查点仍 OOM** | **直接削弱"省显存"这一核心卖点**；ORV 的检查点对激活的回收现在弱于 PyTorch |
| fp16 **只覆盖前向 GEMM**，反向仍 fp32 | P100 无 Tensor Core，收益远小于 V100；而 torch autocast 前反向都覆盖 |
| 缺 `cos`/`sin`、SDPA 参数约定与 torch 不同、MoE 36 专家 top-6 需重写 | 迁移成本高于预期 |
| sm_60 本身支持，张量/模型/AdamW 能正常训练 | 硬件不是障碍，是软件缺口 |

**README 已加适用范围声明**：`"比 PyTorch 省显存"目前只在 3B MoE / 32GB V100 这个已知组合上有证据`，
不能外推。四处缺口已记入 `docs/DESIGN.md` 技术债表。

### 说明

本版**只修了** NVRTC DLL 定位。梯度检查点显存回收、反向 fp16、`cos`/`sin`、SDPA 约定
这四项**尚未修**，已在技术债表与 README 里明确标注。

套件：**656 passed / 36 skipped**（本次改动不影响数值路径）。
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
                checks = {
                    "nvidia/*/lib 纳入": '"nvidia"' in cu or "'nvidia'" in cu,
                    "PATH 前置": 'os.environ["PATH"] = path +' in cu,
                    "add_dll_directory 保留": "add_dll_directory" in cu,
                }
                for kk, hit in checks.items():
                    print("   %-22s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
