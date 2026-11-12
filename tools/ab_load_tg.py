# -*- coding: utf-8 -*-
"""A/B：同一个模型，v3 与 v4 版本号下，训练侧 `load_tg` 的行为是否一致。

## 为什么这样测

2026-11-12 实测发现：v4 文件（v3 布局 + version=4）用 `load_tg` 取
`token_embedding.weight` 时会去解 q4_0 并申请 288 GiB。这可能是**夹具本身的命名/形状不匹配**，
也可能是 v4 读取有问题。用**同一个文件的 v3 / v4 两份**做 A/B，就能把两者分开：
* 两边报同样的错 ⇒ 与版本号无关，是夹具问题；
* 只有 v4 报错 ⇒ v4 读取确实有问题，必须修。

用法：`python ab_load_tg.py <base.tg>`
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "python"))

CHILD = r"""
import sys
sys.path.insert(0, sys.argv[1])
from tgai_go.train.checkpoints import load_tg
try:
    r = load_tg(sys.argv[2])
    print("OK", type(r).__name__, len(r) if hasattr(r, "__len__") else "")
except Exception as exc:
    msg = "%s: %s" % (type(exc).__name__, str(exc)[:150])
    print("ERR", msg)
"""


def run(path):
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", CHILD,
                        os.path.join(REPO, "python"), path],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip() or (r.stderr or "").strip().splitlines()[-1]
    return out[:200]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    base = sys.argv[1]
    tmpdir = tempfile.mkdtemp(prefix="abtg_")
    try:
        v4 = os.path.join(tmpdir, "as_v4.tg")
        r = subprocess.run([sys.executable, "-X", "utf8",
                            os.path.join(HERE, "set_tg_version.py"),
                            base, v4, "--version", "4"],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print("生成 v4 失败:", (r.stdout + r.stderr)[:200])
            return 1
        print("v3:", run(base))
        print("v4:", run(v4))
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
