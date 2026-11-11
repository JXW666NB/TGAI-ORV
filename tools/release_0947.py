# -*- coding: utf-8 -*-
"""0.9.47：修掉容量因子的 CPU/CUDA 分叉（BUG #2）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.47"
ENTRY = """## [0.9.47] - 2026-11-11

**修复：容量因子在 CPU / CUDA 上丢弃的 token 不一致**（会真实影响用户）。

### 问题

0.9.40 用「截断每个专家的偏移量」实现容量限制，这依赖后端 `moe_compact` 的顺序。
实测两条路径丢的不是同一批 token：

| | 修复前 | 修复后 |
| --- | --- | --- |
| `cf=1.0` 相对差 | **7.57e-01** | **5.88e-07** |
| 丢弃 token | CPU=3，**GPU=0** | CPU=3，**GPU=3** |
| 集合一致 | **False** | **True** |

后果：用户在 CPU 上调试、在 GPU 上训练时，会看到**不该出现的分歧**。

### 修法：改用门控权重掩码

丢弃一个 (token, 专家) 对，等价于把该对的门控权重置 0。
掩码**只依赖 `idx`**（两条路径都从同一份 logits 选出），所以天然一致。

```python
def _capacity_mask(idx, n, cap, n_experts, top_k):
    # 稳定排序 ⇒ 同一专家内部仍是 (token, slot) 升序，与 CPU 扫描顺序完全相同
    order = np.argsort(flat, kind="stable")
    ...
```

同时**删掉**了 CPU 路径里旧的 `tok/slot` 截断 —— 否则两套规则叠加会再次分叉。

### 这个修法不需要新内核

之前判断"需要排序内核，暂不修"是过度悲观：`off`/`tok` 本就在主机侧，
用 `argsort(kind="stable")` 就够。已在 `docs/DESIGN.md` 的技术债表里更正。

### 测试（9 条，+3）

新增三条直接验证掩码规则：
`test_capacity_mask_keeps_first_cap_per_expert`（构造已知分布逐位核对）、
`test_capacity_mask_is_deterministic`（稳定排序）、
`test_capacity_mask_respects_cap_exactly`。

GPU 校验也恢复成**严格判据**（要求丢弃集合一致），不再是"限制下不要求一致"。
真 V100 结果：`cf=None/2.0/1.0` 三档全部 `集合一致=True`。

套件：**652 passed / 33 skipped**。
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
                checks = {
                    "_capacity_mask": "def _capacity_mask" in moe,
                    "旧截断已删": "_clamp_offsets" not in core,
                    "CPU 用掩码": "w = w * _as_tensor(keep" in moe,
                    "CUDA 用掩码": "w = w * mask_t" in moe,
                }
                for kk, hit in checks.items():
                    print("   %-16s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
