# -*- coding: utf-8 -*-
"""0.9.25 发版：PPO/RLHF + preset 批量 + min 平局 BUG 修复。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.25"

ENTRY = """## [0.9.25] - 2026-11-10

补齐 **PPO / RLHF**（此前只有 DPO/GRPO），修一个**会让策略往反方向学**的梯度 BUG，
并让 `T.preset` 支持 batch>1。

### 新增：PPO / RLHF

- `T.ppo.gae(rewards, values, dones, gamma, lam)`：广义优势估计（含 done 的 bootstrap 终止）。
- `T.ppo.ppo_loss(...)`：PPO 裁剪目标 + 可选的 value 损失与熵项。
- `T.ppo.discounted_returns(...)`：没有 critic 时的最简基线。
- `T.PPOTrainer`：actor-critic 训练器；`value_params` 会自动并进参数集合（否则 critic 静默不训练）。

于是四种训练模式齐了：**预训练 / SFT / DPO / GRPO / PPO(RLHF)**。

### 修复：min 平局导致梯度符号翻转（PPO 与 GRPO 都中招）

`min(s1, s2)` 原先用 `-max(-s1, -s2)` 组合实现。当 `ratio == 1` 时 `s1 == s2` 是**平局**，
`maximum` 把梯度**整份给第二个参数** ⇒ 代理损失对 logπ 的梯度符号翻转
（优势为正的样本，logπ 反而被压下去）。

实测（`tools/debug_ppo_grad.py`）：

| 情形 | 修复前 | 修复后 |
| --- | --- | --- |
| adv=+1, ratio=1 | 梯度 +1（错） | **-1**（对） |
| adv=-1, ratio=1 | 梯度 -1（错） | **+1**（对） |
| adv=+1, ratio=2（应被 clip） | — | **0**（对） |

现在改用**显式掩码**挑较小者（平局时两者相等，取哪边导数都正确）。
`align.grpo_loss` 里同一处写法一并修掉。

### preset 支持 batch>1

`T.preset.transformer(...)` 现在接受 `[seq]` 与 `[batch, seq]`，内部按 `[B*S, D]` 算、
注意力按 `[B*H, S, Dh]` 分组。测试里加了**批次独立性**断言：同一条序列在批量与单独前向中结果一致
（若 batch 维串了会立刻失败）。

### 测试

- `tests/test_ppo.py`（9 条）：GAE 与手写递推一致、done 终止 bootstrap、与 torch 参考的损失与梯度、
  clip 行为、value/熵项系数。
- `tests/test_ppo_trainer.py`（4 条）：step 跑通、critic 真的进参数集合、可学任务上优势高样本的 logπ 上升。
- `tests/test_preset_quickstart.py` 增 3 条批量用例。

套件：**553 passed / 28 skipped**。
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
                ppo = z.read("tgai_go/train/ppo.py").decode("utf-8", "replace")
                al = z.read("tgai_go/train/align.py").decode("utf-8", "replace")
                rl = z.read("tgai_go/train/rl_trainer.py").decode("utf-8", "replace")
                pre = z.read("tgai_go/train/preset.py").decode("utf-8", "replace")
                core = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                checks = {
                    "gae": "def gae(" in ppo,
                    "ppo_loss": "def ppo_loss(" in ppo,
                    "显式掩码": "take1" in ppo and "take1" in al,
                    "PPOTrainer": "class PPOTrainer" in rl,
                    "导出": "PPOTrainer" in core,
                    "preset 批量": "def forward(self, x, B=None, S=None)" in pre,
                }
                for kk, hit in checks.items():
                    print("   %-12s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
