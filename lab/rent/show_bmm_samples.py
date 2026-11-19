# -*- coding: utf-8 -*-
"""查看 probe_bmm_sites.py 采样到的输入取值范围。"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"lab\rent\bmm_sites.json"
d = json.load(open(path, encoding="utf-8"))
for k, v in sorted(d["samples"].items()):
    print(k)
    if "err" in v:
        print("   ERR", v["err"])
        continue
    for i, b in enumerate(v["bufs"]):
        print("   arg%d n=%d min=%.4g max=%.4g absmax=%.4g std=%.4g"
              % (i, b["n"], b["min"], b["max"], b["absmax"], b["std"]))
    print("   stack: " + " <- ".join(reversed(v["stack"][:4])))
