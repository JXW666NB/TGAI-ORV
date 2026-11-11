# -*- coding: utf-8 -*-
"""核实：格式转换与"把 .TG / .pt / GGUF 当训练检查点"到底支持到什么程度。

不猜，逐项从代码里找。
"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "python", "tgai_go", "train")


def rd(rel):
    p = os.path.join(ROOT, rel)
    return io.open(p, encoding="utf-8").read() if os.path.isfile(p) else ""


print("=== 1. 检查点相关模块的公开函数 ===")
for name in ("ckpt.py", "checkpoints.py", "interop.py"):
    s = rd(os.path.join("python", "tgai_go", "train", name))
    if not s:
        print("  [缺] %s" % name)
        continue
    fns = re.findall(r"(?m)^def (\w+)", s)
    print("  %-16s %s" % (name, fns))

print()
print("=== 2. 训练侧是否出现 TG / GGUF 读写 ===")
for name in ("ckpt.py", "checkpoints.py", "export.py", "interop.py", "__init__.py"):
    s = rd(os.path.join("python", "tgai_go", "train", name))
    if not s:
        continue
    for kw in ("GGUF", "gguf", "save_tg", "load_tg", "from_tg", "pack import TG",
               "TG(", "read_tg"):
        c = s.count(kw)
        if c:
            print("  %-18s %-14s %d" % (name, kw, c))

print()
print("=== 3. pack 模块（.TG 格式）公开接口 ===")
s = rd(os.path.join("python", "tgai_go", "pack", "__init__.py"))
if not s:
    for cand in ("pack.py",):
        s2 = rd(os.path.join("python", "tgai_go", cand))
        if s2:
            s = s2
print("  ", re.findall(r"(?m)^(?:def |class )(\w+)", s)[:20])

print()
print("=== 4. 是否存在 GGUF 读写实现 ===")
hits = []
for base, _, files in os.walk(os.path.join(ROOT, "python")):
    if "__pycache__" in base:
        continue
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(base, f)
        try:
            t = io.open(p, encoding="utf-8").read()
        except Exception:
            continue
        if "gguf" in t.lower():
            hits.append(os.path.relpath(p, ROOT))
print("  提到 gguf 的 py 文件:", hits or "无")

print()
print("=== 5. tgai.py 的 convert 子命令 ===")
t = rd("tgai.py")
i = t.find("convert")
while i > 0:
    seg = t[max(0, i - 90):i + 160].replace("\n", " ")
    print("  ...", seg.strip()[:200])
    i = t.find("convert", i + 1)
    if t.count("convert") > 6:
        break
