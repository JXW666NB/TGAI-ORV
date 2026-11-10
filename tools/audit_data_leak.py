# -*- coding: utf-8 -*-
"""排查"用了 self.data 但没有 CUDA 分支"的方法（按行扫描，避免正则切片出错）。

`self.data` 在 CUDA 张量上是 None，所以任何 `np.xxx(self.data)` 都会产出 None/object 数组，
最终变成 nan 或抛异常。已经踩到 `masked_fill`（CUDA 上返回 nan，不是报错）。
"""
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lines = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "__init__.py"),
                encoding="utf-8").read().splitlines()

blocks = []          # [(name, [lines])]
cur_name, cur_body, in_class = None, [], False
for ln in lines:
    if ln.startswith("class ") or ln.startswith("def "):
        if cur_name:
            blocks.append((cur_name, cur_body))
        cur_name, cur_body = None, []
        in_class = ln.startswith("class ")
        continue
    if ln.startswith("    def ") and in_class:
        if cur_name:
            blocks.append((cur_name, cur_body))
        m = ln.strip()
        cur_name = m[4:m.find("(")] if "(" in m else m
        cur_body = [ln]
        continue
    if cur_name is not None:
        cur_body.append(ln)
if cur_name:
    blocks.append((cur_name, cur_body))

bad = []
for name, body in blocks:
    text = "\n".join(body)
    if "self.data" not in text:
        continue
    if 'device == "cuda"' in text:
        continue
    bad.append(name)

print("方法里「用 self.data 且无 CUDA 分支」的：%d 个" % len(bad))
for n in bad:
    print("   ", n)
