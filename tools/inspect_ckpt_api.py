# -*- coding: utf-8 -*-
"""取 load_tg / load_gguf / from_torch / save_checkpoint 的签名与首段说明。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rd(rel):
    p = os.path.join(ROOT, rel)
    return io.open(p, encoding="utf-8").read() if os.path.isfile(p) else ""


def show(rel, fn, n=420):
    s = rd(rel)
    i = s.find("def %s(" % fn)
    if i < 0:
        print("[未找到] %s::%s" % (rel, fn))
        return
    # 取到 def 行结束（可能跨行）
    j = s.find(":", i)
    while j > 0 and s[j - 1] not in ")\n" and "->" not in s[i:j]:
        j = s.find(":", j + 1)
    print("=== %s :: %s ===" % (rel, fn))
    print(s[i:i + n])
    print()


CK = os.path.join("python", "tgai_go", "train", "checkpoints.py")
show(CK, "load_tg")
show(CK, "load_gguf")
show("python/tgai_go/train/interop.py", "from_torch")
show("python/tgai_go/train/ckpt.py", "save_checkpoint")

print("=== checkpoints.py 顶部说明（前 700 字）===")
print(rd(CK)[:700])
