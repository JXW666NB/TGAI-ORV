# -*- coding: utf-8 -*-
"""0.9.20 发版：可编程 AMP —— T.amp() / amp_on / amp_off / GradScaler。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.20"

ENTRY = """## [0.9.20] - 2026-11-10

把 `TG_FP16` 环境变量升级成**可编程 AMP**：`with T.amp():`，不用再重启进程。

### 新增

- `T.amp(enabled=True)` 上下文管理器；`T.amp_on() / T.amp_off() / T.amp_state()`；`T.amp_invalidate()`。
- `T.GradScaler`：为 PyT 迁移代码保留（`scale_loss / unscale_ / step(opt) / update()`），
  但 **ORV 默认 AMP 不需要它** —— 见下面的设计说明。
- `T.amp` 同时保留旧语义：它以前是 bf16 工具模块（`T.amp.to_bf16_bits`、`T.amp.BF16_KERNELS`），
  现做成"可调用 + 属性代理"的命名空间，**旧代码不破**（`tests/test_amp_bf16.py` 全绿）。

### 设计与 PyT autocast 的差别

| 项 | ORV | PyTorch autocast |
| --- | --- | --- |
| 参数 / 激活 / 梯度 | **全程 fp32** | 激活可为 fp16 |
| GEMM 输入 | fp16（带权重影子缓存） | fp16 |
| GEMM 累加 | fp32 | fp32 |
| 需要 loss scaling | **不需要** | 需要 |

因为梯度和激活始终 fp32，所以不存在下溢；`GradScaler` 只是兼容层。

### 实测（V100）

| 项 | 结果 |
| --- | --- |
| GEMM 2048×2048×6144 | fp32 **4.33 ms** → amp **0.67 ms（6.47×）** |
| 真实 8 层训练步 | loss 4219.40 → **4219.45**（相对差 1.25e-05），梯度有限 |
| 退出上下文 | 状态正确复原（`on=False`，缓存关闭） |
| 数值（大 GEMM） | 相对 fp32 误差 2.78e-04~2.88e-04，与 torch AMP 同值 |

### 边界

- 只支持 fp16（V100 无 bf16；`_sgemm_bf16` 内核已在，接进 Tensor 层属于后续工作）。
- `amp()` 是**全局开关**（GEMM 级），不是 PyT 的逐算子策略；conv/attention 走同一套 GEMM 路由。
- 自定义优化器更新参数后要调 `T.amp_invalidate()`（内置 `Fp16MomAdamW` 已自动）。
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
                api = z.read("tgai_go/train/amp_api.py").decode("utf-8", "replace")
                core = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                checks = {
                    "amp_api": "class amp(object)" in api,
                    "GradScaler": "class GradScaler" in api,
                    "命名空间代理": "_AmpNamespace" in core,
                    "旧 bf16 兼容": "from . import amp as _amp_module" in core,
                }
                for k, hit in checks.items():
                    print("   %-14s %s" % (k, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
