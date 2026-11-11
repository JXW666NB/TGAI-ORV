# -*- coding: utf-8 -*-
"""0.9.51：四个 P100 迁移缺口里的三项落地 + 一份干净头对头实测。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.51"
ENTRY = """## [0.9.51] - 2026-11-12

按 P100 实测报告（`docs/REVIEW_P100_2026-11-12.md`）逐项处理四个迁移缺口。

### 1. 梯度检查点：查明「回收弱于 PyTorch」的真因并修复

**真因不是检查点算法，而是引擎的一个底层缺陷**：所有算子在构造时**无条件**传
`_parents=(self, other)`，连 `no_grad()` 里也一样。于是"不建图"只是没设 `_backward`，
**中间激活仍被 `_parents` 引用住、释放不掉**。

旧实现用「克隆输出」硬切断引用链，代价是每段检查点多占一份段输出
（20 层/d=1280/B=8/S=2048 ⇒ 每层 84 MB，共约 1.7 GB）—— 量与 16GB 卡上的差距同阶。

现在改为**直接清空 `_parents`**：既切断引用，又不拷贝。另加一条显式检查：
若 `fn` 直接返回了某个输入（ptr 相同），报错而不是把输入当检查点输出。

| dense 检查点峰值显存 | 修前 | 修后 |
| --- | --- | --- |
| ORV | 0.857 GiB | **0.602 GiB（−30%）** |

**但没消灭差距**：与 PyTorch 的比值只从 1.89× 降到 1.83×，因为 torch 也降了 28%。
剩余约 0.28 GiB 属 ORV 的基础开销，已记入技术债。

### 2. SDPA 接受 torch 风格的 4 维输入

torch 习惯 `[B, H, S, Dh]`，而 ORV 内部要 `[B*H, S, Dh]`，照 torch 写会报维度错。
现在两种都接受：4 维自动展平、返回时还原（还原只有一处，不会被漏掉）。
参数名本来就兼容（`is_causal=`）。

全部数值对拍：4 维与手工展平的 3 维结果**完全相同**（相对差 0.0），
反向梯度形状也是 4 维。新增 `tests/test_sdpa_4d.py`（4 条）。

### 3. 新增 `Tensor.cos` / `Tensor.sin`

P100 报告指出缺这两个导致 RoPE 类模型必须自己用 numpy 预生成表。
现在 CPU 走 numpy、CUDA 走新内核（`ewise_cos` / `ewise_sin` / `ewise_cos_bwd` /
`ewise_sin_bwd`），反向分别是 `-sin(x)` 与 `cos(x)`。
新增 `tests/test_cos_sin.py`（6 条，含 RoPE 旋转形态的组合求导）。

### 4. 反向 fp16：**已是既有能力，报告这条判断需要修正**

报告说「fp16 只覆盖前向 GEMM，反向仍 fp32」。实测查证：fp16 分支在
**共享的 GEMM 核心**里（`_sgemm_common`），反向同样会走，只是被形状门槛挡住：

```python
if m*n >= _FP16_MIN_OUTPUT and k <= _FP16_MAX_K and fp16_on:
```

`_FP16_MAX_K` 默认 `1<<30`（等于不限制），可由 `TG_FP16_MAX_K` 调整。
代码注释记录了长 K 的代价：dW 类形状会让参数梯度相对误差从 3e-4 恶化到 2e-2
（600 步 loss 偏 10%）。用户 B=8/S=2048 时 `k=16384`，正落在这个风险区。

**本次未改默认值** —— 那是性能与精度的权衡，改了会影响所有现有用户。
真机实测（dense/fp32，V100）：`TG_FP16=1` 让 18.63 ms → **16.85 ms（1.11×）**，
loss 差 ~1e-07，说明小 K 形状下数值无碍。**Pascal 无 Tensor Core，收益远小于 V100**，
这一点现在写进了 README 与 skill。

### 附：一份自包含的干净头对头实测

新增 `tools/h2h_clean.py`（同一份权重、两侧独立进程、CUDA event、5 次取均值）
与 `docs/BENCHMARK_H2H_2026-11-12.md`。核心结论：

| 配置 | ORV | PyTorch | 速度 | 显存 |
| --- | --- | --- | --- | --- |
| dense / fp32 | 16.55 ms | 42.73 ms | ORV 快 **2.58×** | ORV 多 1.89× |
| MoE / fp32 | 83.04 ms | 139.37 ms | ORV 快 **1.68×** | ORV 多 1.56× |

**ORV 更快，但更费显存（1.2~1.9×）**。README 已按此改写：不再暗示普适省显存。

套件：**665 passed / 37 skipped**（新增 10 条测试；GPU 上新增的 13 条亦全过）。
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
                cu = z.read("tgai_go/train/cuda.py").decode("utf-8", "replace")
                core = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                checks = {
                    "cos/sin 内核": "ewise_cos" in cu and "ewise_sin" in cu,
                    "cos/sin rt 方法": "def cos(self, a):" in cu,
                    "Tensor.cos": "    def cos(self):" in core,
                    "SDPA 4 维包装": "def _sdpa_impl(" in core,
                    "检查点清 _parents": "res._parents = ()" in core,
                    "检查点去掉 clone": "clone(out._gpu)" not in core,
                }
                for kk, hit in checks.items():
                    print("   %-20s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
