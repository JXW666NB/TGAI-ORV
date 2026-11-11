# -*- coding: utf-8 -*-
"""取 save_tg 签名与 GO 的用法提示，供 skill 文档准确引用。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.isfile(p):
        return ""
    return io.open(p, encoding="utf-8").read()


s = read(os.path.join("python", "tgai_go", "train", "export.py"))
i = s.find("def save_tg")
print("=== save_tg 签名与开头 ===")
print(s[i:i + 500])

print()
print("=== tgai.py print_help 内容 ===")
t = read("tgai.py")
j = t.find("def print_help")
print(t[j:j + 900])
