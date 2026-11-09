# -*- coding: utf-8 -*-
"""阶段二发版：0.9.0（DPO + GRPO）。升版本 → CHANGELOG → 构建 → 校验。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.0"

ENTRY = """## [0.9.0] - 2026-11-09

阶段二：对齐训练形态（DPO / GRPO）。torch 核心不含这些损失，参考实现按论文公式手写并逐项对照。

### 新增（`tgai_go.train.align`，导出为 `T.align.*` 与顶层函数）

- `logsigmoid(x)`：数值稳定（`−softplus(−x)` 形式）
- `seq_logprob(logits, labels, ignore_index)`：回答段 logπ **之和**，保持可微
- `dpo_pair_loss(pw, pl, rw, rl, beta)` / `dpo_batch_loss(...)`：DPO
- `group_advantages(rewards, eps)`：GRPO 组内归一化优势（总体标准差）
- `token_logprobs(logits, labels, ignore_index)`：**逐 token** logπ（可微）
- `grpo_loss(policy_logps, old_logps, ref_logps, advantages, masks, clip_eps, beta, kl)`：
  token 级 PPO 裁剪 + KL（k1 / k3），返回 (loss, stats)

### 实测对照（手写 torch 参考，独立实现同一公式）

| 项 | 判据 | 实测 |
| --- | --- | --- |
| `logsigmoid`（含 ±40）| 1e-6 | ≤1e-6 |
| `seq_logprob`（含 ignore_index）| 1e-5 | ≤1e-5 |
| DPO 单对 loss / 对两侧 logits 的梯度 | 1e-5 | ≤1e-5 |
| DPO batch 平均（beta ∈ {0.01,0.1,0.5}）| 1e-5 | ≤1e-5 |
| `token_logprobs` | 1e-5 | ≤1e-5 |
| GRPO loss / 梯度（beta=0 / 0.1·k3 / 0.1·k1）| 1e-5 | ≤1e-5 |
| GRPO 裁剪触发统计 | — | ratio 远离 1 时 clip_frac>0 |

### 已知限制（写进 docstring，未隐藏）

后端没有 gather 算子，因此：
- `seq_logprob` 逐序列调用（batch 内多几次 kernel）；
- `token_logprobs` 需要一份 [T, V] 的 one-hot，词表大时显存开销明显。

两处都留了"等 gather 就绪再向量化"的说明，接口不需要变。
"""


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def main():
    py = os.path.join(ROOT, "pyproject.toml")
    s = re.sub(r'(?m)^version\s*=\s*"[^"]+"', 'version = "%s"' % VER, read(py), count=1)
    write(py, s)

    ini = os.path.join(ROOT, "python", "tgai_go", "__init__.py")
    t = re.sub(r'(?m)^(\s*__version__\s*=\s*")[^"]+(")',
               r"\g<1>%s\g<2>" % VER, read(ini))
    write(ini, t)
    print("版本字面量:", re.findall(r'(?m)^\s*__version__\s*=\s*"([^"]+)"', read(ini)))

    cl = os.path.join(ROOT, "CHANGELOG.md")
    txt = read(cl)
    if "[%s]" % VER not in txt:
        i = txt.find("## [")
        txt = (ENTRY + txt) if i < 0 else (txt[:i] + ENTRY + txt[i:])
        write(cl, txt)
        print("CHANGELOG 已插入", VER)

    for d in ("dist", "build"):
        p = os.path.join(ROOT, d)
        if os.path.isdir(p):
            shutil.rmtree(p)
    for f in glob.glob(os.path.join(ROOT, "python", "*.egg-info")):
        shutil.rmtree(f, ignore_errors=True)

    r = subprocess.run([sys.executable, "-m", "build", "--outdir", "dist"], cwd=ROOT,
                       capture_output=True, text=True)
    print("build rc=%d" % r.returncode)
    if r.returncode != 0:
        print(r.stdout[-1200:], r.stderr[-1200:])
        return r.returncode

    wheels = glob.glob(os.path.join(ROOT, "dist", "*.whl"))
    print("产物:", [os.path.basename(x) for x in wheels])
    want = {"dpo": b"def dpo_pair_loss", "grpo": b"def grpo_loss",
            "token_logprobs": b"def token_logprobs", "advantages": b"def group_advantages"}
    ok = True
    for w in wheels:
        with zipfile.ZipFile(w) as z:
            names = z.namelist()
            meta = [n for n in names if n.endswith("METADATA")]
            ver_ok = (("Version: %s" % VER) in z.read(meta[0]).decode("utf-8", "replace")) if meta else False
            blob = b"".join(z.read(n) for n in names
                            if n.startswith("tgai_go/train/") and n.endswith(".py"))
            print("wheel %s 版本正确=%s" % (os.path.basename(w), ver_ok))
            for k, needle in want.items():
                hit = needle in blob
                print("   %-14s 包含=%s" % (k, hit))
                ok = ok and hit
            ok = ok and ver_ok
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
