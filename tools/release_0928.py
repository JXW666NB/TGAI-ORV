# -*- coding: utf-8 -*-
"""0.9.28 发版：Muon 显存修复（3B 全参跑通）+ README 三段式重写。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.28"

ENTRY = """## [0.9.28] - 2026-11-10

**3B MoE 全参训练在 V100 32GB 上跑通**（峰值 18.93 GiB，剩余 12.93 GiB，不用主机内存）。
关键是把 Muon 的状态占用从 33.3 GiB 降到 5.5 GiB。

### 修复：Muon 给每个参数分配了三份状态

旧实现无条件为**每个参数**分配 `buf + _adam_m + _adam_v`：

```
2.97B × 3 份 × 4 字节 = 33.3 GiB     （比 AdamW 的 22.2 GiB 还差，3B 上完全不可用）
```

现在按形状只分配真正用得上的：

| 参数形状 | 走哪条路 | 状态 |
| --- | --- | --- |
| 2D（投影、专家矩阵、embedding） | Muon 正交化 | **1 份动量** |
| 其它（1D 的 RMSNorm、标量） | AdamW | m + v |

再接上 `fp16_mom`（动量存 fp16，内核仍是 fp32 计算，逐张量转换）：

```
2.97B：fp32 动量 11.1 GiB → fp16 动量 5.5 GiB
```

同时新增 `state_bytes()` / `bytes_per_param()`，让"能不能装下"可以有可靠答案而不是估算。

### 新增：分层即时更新（`step_layers`）

反向是逐层的：层 i 的 `dL/dx_i` 算完之后，层 i-1 的反向**不再需要层 i 的权重**。
因此可以此时就用同一个 `t` 更新层 i 并释放它的梯度 —— 同时驻留的梯度只需一层（约 0.4 GiB
而不是 11.1 GiB）。

`set_layer_groups()` 会**强制校验覆盖完整**：绑定权重（`tie_embedding` 时 `embed.weight` 与
`lm_head.weight` 是同一对象）在参数列表里会出现两次，按出现次数更新会让同一参数被更新两遍
（实测差异 1.285e-03）；分组漏掉 `embed/norm/head` 则会**静默少训练**（实测差异 1.1e-02）。
两处都已在代码里拦住，并自动把未覆盖参数收进 `-1` 组。

### 实测（V2.7 28 层 2.97B，V100 32GB）

| 配置 | 优化器状态 | 峰值 | 结果 |
| --- | --- | --- | --- |
| AdamW fp32 动量 | 22.2 GiB | ≥46 GiB 需求 | 不可行 |
| AdamW fp16 动量 + 全梯度驻留 | 11.1 GiB | 33.6 GiB 需求 | **实测 OOM** |
| **Muon fp16 动量 + 逐层即时更新** | **5.5 GiB** | **18.93 GiB** | ✅ 跑通，剩 12.93 GiB |

### README 重写

按"① 这是什么 → ② 优势与疑问 → ③ 新手入门"三段式重写，全文使用项目第三人称
（此前的第二人称"你的显卡"不适合对外宣传）。同步更新显存实测表。

### 已知不足（本轮未完成）

- Muon 的**单元测试尚未补齐**（GPU 实跑验证了分配、fp16、`step_layers`，
  但仓库里还没有对应的 pytest）。下一次发版前补上。
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
                body = z.read(meta).decode("utf-8", "replace")
                ok = ok and (("Version: %s" % VER) in body)
                core = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                i0 = core.find("class Muon")
                mu = core[i0:core.find("\nclass ", i0 + 10)] if i0 >= 0 else ""
                checks = {
                    "Muon 按形状分配": "self._is2d.append" in mu,
                    "Muon fp16 动量": "fp16_mom" in mu,
                    "Muon state_bytes": "def state_bytes" in mu,
                    "Muon step_layers": "def step_layers" in mu,
                    "README 三段式": "第一部分：这是什么" in body,
                    "README 无第二人称": "你" not in body.split("第一部分")[0][:2000],
                }
                for kk, hit in checks.items():
                    print("   %-18s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
