# -*- coding: utf-8 -*-
"""0.9.53：检查点按 PyTorch CheckpointFunction 重做，P100 16GB 可以训了。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.53"
ENTRY = """## [0.9.53] - 2026-11-12

**梯度检查点重做（按 PyTorch `CheckpointFunction` 的做法），P100 16GB 从"训不了"变成"能跑"。**

### 问题

P100 用户反馈：同形状 PyTorch 能跑、ORV 开检查点仍 OOM。我在 V100 上复现并用逐块探针定位：

| 指标 | 修复前 |
| --- | --- |
| 前向结束活张量（L=6/S=2048/B=2）| 0.133 GiB |
| **第一段反向开始时** | **8.474 GiB** ← 所有层激活同时驻留 |
| 反向峰值 | 10.80 GiB |
| 每层斜率 | **1.68 GiB/层** |

### 真因（读 `torch/utils/checkpoint.py` 对照得出）

上一版为了释放中间激活把 `res._parents` 清成 `()`，导致**外层图在段与段之间断开**。
反向时只能靠 `bw` 里的 `re.backward(g)` 递归钻进上一段（上一段输出的 `_backward` 正是它自己的 `bw`），
于是**一次调用把整条链全部重算**，所有段的激活同时存在。

PyTorch 是两件事一起做：

```python
detached_inputs = detach_variable(tuple(inputs))   # ① 输入变成叶子，嵌套反传停在这里
with torch.enable_grad():
    outputs = ctx.run_function(*detached_inputs)
torch.autograd.backward(outputs_with_grad, args_with_grad)
grads = tuple(inp.grad for inp in detached_inputs)
return (None, None) + grads                        # ② 梯度作为返回值交还外层引擎
```

⇒ 外层图**保持连通**，中间激活靠"输出不引用中间张量"释放，梯度由外层引擎路由。

### 改法

* `res._parents = tuple(x for x in inputs if x.requires_grad)` —— 只连输入，保持连通，不引用中间激活；
* `bw` 里用**共享缓冲的 detached 叶子**重算（不拷贝），嵌套反传在其上停住；
* 算完把 `detached.grad` 通过 `_acc` 交还外层 `grads`。

### 效果（同一组配置，V100 实测）

| L | 修复前反向峰值 | **修复后** |
| --- | --- | --- |
| 2 | 4.080 GiB | **2.517** |
| 4 | 7.439 GiB | **2.752** |
| 6 | 10.799 GiB | **2.987** |

每层斜率 **1.68 GiB → 0.117 GiB（降 14 倍）**；剩余的 0.117 GiB/层是叶子权重梯度累积，属正常。
"关检查点"的对照组数值不变（4.00/7.28/10.57），说明改动只影响检查点路径。

**P100 用户的目标配置**（20 层 / d=1280 / FF=5120 / dh=128 / B=2 / S=2048，606M 参数）：

| | 峰值活张量 | 16 GiB |
| --- | --- | --- |
| 修复前 | 31.29 GiB（OOM）| 跑不了 |
| **修复后** | **7.56 GiB**（含池缓存 8.66）| **能跑** |

### 其它

* `tests/test_checkpoint_cuda.py` 里原先断言"输出不能连父节点"，那个方向是错的，
  改为断言"**恰好连到输入**、不引用中间张量"，并加数值等价校验。
* CI 修好（此前每次推送都红）：`tools/_*.txt` 临时文件带 BOM 已从索引移除并加入 gitignore；
  `test_plugin_native.py` 在 Linux 上把 `find_library("c")` 返回的 soname `libc.so.6` 当路径用，
  改为先解析成真实绝对路径（顺带让这两个用例在 Windows 上从"跳过"变成"真跑"）。

套件：**670 passed / 40 skipped**。
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
                core = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                checks = {
                    "parents 连到输入": "res._parents = tuple(x for x in ts if x.requires_grad)" in core,
                    "detached 重算": "d._backward = None" in core,
                    "梯度交还外层": "gg._gpu if isinstance(gg, Tensor)" in core,
                }
                for kk, hit in checks.items():
                    print("   %-18s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
