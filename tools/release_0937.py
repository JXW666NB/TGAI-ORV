# -*- coding: utf-8 -*-
"""0.9.37 发版：MoE 共享专家（DeepSeek-V3 / Llama 4 结构）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.37"
ENTRY = """## [0.9.37] - 2026-11-11

**MoE 支持共享专家**（`n_shared`）—— 现代主流 MoE 的标配，此前完全不支持。

### 为什么补这个

调研现代 MoE 配置（`docs/MOE_MODERN_ARCH.md`）：

| 模型 | 路由专家 | top-k | 共享专家 |
| --- | --- | --- | --- |
| Mixtral 8x7B | 8 | 2 | 无 |
| DeepSeek-V2 | 160 | 6 | 有 |
| **DeepSeek-V3** | **256** | **8** | **1** |
| Llama 4 Scout / Maverick | 16 / 128 | 1 | **1** |

ORV 之前 `shared` 在代码里出现 **0 次**，意味着训不了 DeepSeek-V3 / Llama 4 这类结构。

### 语义（DeepSeekMoE）

```
y = Σ_{i∈TopK} g_i · E_i(x)  +  Σ_j S_j(x)
    └─ 路由部分（加权）─┘      └ 共享部分（不加权、全 token）┘
```

共享专家**恒定激活、不参与路由、不带门控权重**，作用是吸收通用知识，
让路由专家专注特化。

### 实现选择：加在 `_forward_flat` 这一层

好处是 **CPU 与 CUDA 两条路径自动同时生效**，不会出现"一条实现、一条没实现"
的数值不一致（这是 ORV 一直守的纪律）。已实测确认。

### 实测（真 V100，`tools/shared_gpu_check.py`）

| 项 | 结果 |
| --- | --- |
| CPU vs CUDA 相对差（n_shared=1） | **6.09e-07** ⇒ 一致 |
| n_shared=0 前向 | 3.10 ms |
| n_shared=1 前向 | **3.25 ms（+0.15，约 +5%）** |
| n_shared=2 前向 | 3.35 ms |
| GPU 上共享专家拿到梯度 | ✅ |

**每个共享专家只增加约 5% 前向耗时**，代价很小。

### 新增 API

```python
T.nn.MoE(d_model, d_ff, n_experts, top_k=2,
         n_shared=1,          # 共享专家数量，默认 0 = 与旧行为完全一致
         shared_d_ff=None)    # 共享专家的 d_ff，默认等于 d_ff
```

### 测试（7 条，`tests/test_moe_shared_experts.py`）

1. `n_shared=0` 与旧行为**逐位相同**（向后兼容）；
2. 输出 == 路由输出 + 共享输出，且共享部分**不带门控权重**（相对差 ≤1e-6）；
3. 把路由专家清零后输出只剩共享部分 ⇒ 证明共享专家**看到全部 token**；
4. 梯度能流到共享专家；
5. 参数按每个共享专家 3 个矩阵线性增长；
6. `shared_d_ff` 可独立设置；
7. `n_shared < 0` 报错。

套件：**575 passed / 31 skipped**（+7）。
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
                core = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                i0 = core.find("class MoE")
                moe = core[i0:] if i0 >= 0 else ""
                checks = {"n_shared 参数": "n_shared=0" in moe,
                          "shared_d_ff": "shared_d_ff" in moe,
                          "_add_shared": "_add_shared" in moe,
                          "接入 _forward_flat": "self._add_shared" in moe}
                for kk, hit in checks.items():
                    print("   %-18s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
