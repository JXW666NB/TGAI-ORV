# -*- coding: utf-8 -*-
"""文档与代码的一致性：文档里写的命令行参数必须真实存在。

为什么值得单独测：我们这一轮改了很多开关（`--resume`、`--seq-len`、`--clip` 默认值…），
文档很容易写着写着就"引用了一个不存在的参数"。用户照着抄会直接报
`unrecognized arguments` —— 而这种错**在 CI 里不会暴露**，因为没人跑文档里的命令。

做法：只检查**同一行里既出现 `qwen25_sft.py` 又出现 `--xxx`** 的参数，
避免把别的工具的参数（`check_package.py --build`、`verify_all.py --quick`）也算进来。
"""
import io
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _example_flags():
    p = os.path.join(REPO, "examples", "qwen25_sft.py")
    s = io.open(p, encoding="utf-8").read()
    return set(re.findall(r'add_argument\(\s*"(--[A-Za-z0-9_-]+)"', s))


def _doc_flags(*docs):
    """只看那些同时在命令行里出现 qwen25_sft.py 的 `--xxx`。"""
    flags = {}
    for rel in docs:
        p = os.path.join(REPO, rel)
        if not os.path.isfile(p):
            continue
        for lineno, line in enumerate(io.open(p, encoding="utf-8"), 1):
            if "qwen25_sft.py" not in line:
                continue
            for f in re.findall(r"(--[A-Za-z0-9_-]+)", line):
                flags.setdefault(f, []).append("%s:%d" % (rel, lineno))
    return flags


def test_documented_train_flags_exist():
    have = _example_flags()
    assert have, "没从 examples/qwen25_sft.py 解析到任何参数，正则可能失效"
    used = _doc_flags("docs/QUICKSTART_TRAIN.md", "README.md",
                      "docs/FINETUNE_RESULTS.md", "docs/STATUS.md")
    bad = {f: v for f, v in used.items() if f not in have}
    assert not bad, ("文档里引用了示例脚本不存在的参数（照着抄会报 unrecognized arguments）：\n  "
                     + "\n  ".join("%s 出现在 %s" % (f, ", ".join(v[:3])) for f, v in bad.items()))


def test_core_flags_are_documented():
    """反过来：几个关键开关必须在 QUICKSTART 里出现过，否则用户不知道它们存在。"""
    doc = io.open(os.path.join(REPO, "docs", "QUICKSTART_TRAIN.md"),
                  encoding="utf-8").read()
    for f in ("--safetensors", "--seq-len", "--save", "--resume", "--holdout", "--clip"):
        assert f in doc, "QUICKSTART_TRAIN.md 里没提到 %s" % f
