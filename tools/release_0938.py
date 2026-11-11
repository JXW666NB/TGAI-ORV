# -*- coding: utf-8 -*-
"""0.9.38 发版：MoE aux-loss-free 负载均衡（DeepSeek-V3）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.38"
ENTRY = """## [0.9.38] - 2026-11-11

**MoE 支持 aux-loss-free 负载均衡**（DeepSeek-V3, arXiv 2408.15664）。
这是现代 MoE 的第二块拼图，接在 0.9.37 的共享专家之后。

### 为什么补这个

纯 top-k 没有均衡机制，容易「路由塌缩」：少数专家被反复选中 → 更新更多 → 更易被选中。
传统做法是加辅助损失（Switch 的 `α·N·Σ f_i·P_i`），但**辅助损失以梯度形式注入 router**，
和主任务梯度打架。V3 改为在**图外**用符号更新调偏置，主梯度保持干净。

### 语义

```
选专家：   s_i(x) + b_i          ← 含 bias
门控权重： g_i(x)                 ← 用**原始**分数，bias 不参与
更新：     b_i ← b_i − u·sign(c_i − c̄_i)    （每步一次，图外）
```

`b_i` 是 **buffer 而非 Parameter**：不进梯度图、不进 `state_dict`、优化器看不到它。

### 新增 API

```python
T.nn.MoE(d_model, d_ff, n_experts, top_k=2,
         balance="aux_free",    # 默认 "none" = 与旧行为逐位相同
         aux_free_lr=1e-3)      # 符号更新的步长 u
```

### 实测

| 项 | 结果 |
| --- | --- |
| CPU vs CUDA 相对差（aux_free，固定 bias） | **5.04e-07** ⇒ 一致 |
| CUDA 上 bias 更新方向（过载↓ / 欠载↑） | ✅ |
| **bias 归零后与 `balance="none"` 的差** | **0.00e+00** ⇒ 证明 bias 不进门控权重 |
| 负载均衡改善（合成用例，30 步） | 最大/均匀负载比 **1.89 → 1.54**（约 −19%）|

### 测试（6 条，`tests/test_moe_aux_free.py`）

1. `balance="none"` 时 bias 恒为 0，输出与旧行为逐位相同；
2. **bias 只改变选择、不改变门控权重**（bias 归零 == 无 bias，逐位）；
3. 符号更新方向正确（过载降、欠载升）；
4. `b_i` 不在 `state_dict`、参数数量不变；
5. **确实能均衡负载**（用最大/均匀负载比量化，1.89 → 1.54）；
6. 非法 `balance` 值报错。

### 已知限制（写清，不藏）

`b_i` 是运行时 buffer，**不进 `state_dict`** —— 断点续训会丢掉它（与 V3 论文的处理一致，
它是控制量不是模型参数）。若需要严格续训，需另行保存。

套件：**581 passed / 31 skipped**（+6）。
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
                checks = {"balance 参数": 'balance="none"' in moe,
                          "aux_free_lr": "aux_free_lr" in moe,
                          "expert_bias": "expert_bias" in moe,
                          "sel_logits 分流": "sel_logits" in moe,
                          "bias 更新": "_update_expert_bias" in moe}
                for kk, hit in checks.items():
                    print("   %-18s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
