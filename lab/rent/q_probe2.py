# -*- coding: utf-8 -*-
"""探针 2：远端 ORV 包真实版本 + CSA 修复痕迹 + 关键实现签名。"""
import importlib.metadata as md
import os
import re
import sys

P = r"D:\orv_pip_0962\Lib\site-packages"
sys.path.insert(0, P)

for name in ("tgai-orv", "tgai_orv", "tgai-go"):
    try:
        print("dist %-10s -> %s" % (name, md.version(name)))
    except Exception as e:
        print("dist %-10s -> (%s)" % (name, type(e).__name__))

import tgai_go
print("tgai_go.__file__ =", tgai_go.__file__)
print("tgai_go.__version__ =", getattr(tgai_go, "__version__", "?"))

f = os.path.join(P, "tgai_go", "train", "v3_csa.py")
txt = open(f, encoding="utf-8", errors="replace").read()
print("v3_csa.py bytes =", len(txt))
print("slot_doc occurrences =", txt.count("slot_doc"))
for m in re.finditer(r"slot_doc", txt):
    a = max(0, m.start() - 90)
    print("   ...", txt[a:m.start() + 90].replace("\n", " | "))
print("has '1.0.7' string:", "1.0.7" in txt)
print("has '1.0.6' string:", "1.0.6" in txt)

# 全包扫一遍版本字符串（判明 wheel 真实版本）
hits = []
for root, dirs, files in os.walk(os.path.join(P, "tgai_go")):
    for fn in files:
        if fn.endswith(".py"):
            p = os.path.join(root, fn)
            if os.path.getsize(p) > 2_000_000:
                continue
            try:
                t = open(p, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            for ver in ("1.0.7", "1.0.6", "0.9.97"):
                if ver in t:
                    hits.append((os.path.relpath(p, P), ver))
print("version-string hits:", hits[:20])
