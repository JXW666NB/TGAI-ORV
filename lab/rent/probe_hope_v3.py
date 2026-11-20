# -*- coding: utf-8 -*-
"""对齐 HOPE V3（_src15b）与 orv_bench 里那份 model_v3 的同一性，并读出配置字段与数据格式。"""
import hashlib
import io
import os
import re
import sys

A = r"D:\DSH-Workspace\TGAI-对话档案\_src15b"
B = r"D:\orv_bench"


def md5(p):
    try:
        return hashlib.md5(open(p, "rb").read()).hexdigest()
    except OSError:
        return None


print("=== md5 对比 ===")
for nm in ("model_v3.py", "data_v3.py", "optim_v3.py", "train_v3.py"):
    a, b = md5(os.path.join(A, nm)), md5(os.path.join(B, nm))
    same = "同一份" if (a and a == b) else ("B 缺失" if b is None else "**不同**")
    print("  %-14s A=%s  B=%s  -> %s" % (nm, (a or "-")[:8], (b or "-")[:8], same))

print("\n=== model_v3.py 的配置默认值（前 30 项）===")
sys.path.insert(0, A)
try:
    import dataclasses
    import model_v3 as M
    fs = dataclasses.fields(M.TGAIV3Config)
    for f in fs[:30]:
        print("  %-26s = %r" % (f.name, f.default))
    print("  ... 共 %d 项" % len(fs))
except Exception as e:
    print("  导入失败:", type(e).__name__, e)

print("\n=== data_v3.py 里的 npz 键名 ===")
src = io.open(os.path.join(A, "data_v3.py"), encoding="utf-8", errors="replace").read()
for m in sorted(set(re.findall(r'["\'](\w+)["\']\s*\]', src)))[:10]:
    print("  key?", m)
for i, l in enumerate(src.splitlines()):
    if re.search(r"np\.(load|savez)|\.npz|files\[|--data", l):
        print("  L%d: %s" % (i + 1, l.strip()[:110]))
