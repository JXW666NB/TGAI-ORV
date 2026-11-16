# -*- coding: utf-8 -*-
"""探测 opt_fp16mv 模块与 T.optim 的完整成员。"""
import inspect

import tgai_go.train as T

print("T.optim 成员:", [n for n in dir(T.optim) if not n.startswith("_")])

for modname in ("opt_fp16mv", "int8_adamw"):
    try:
        m = __import__("tgai_go.train." + modname, fromlist=["*"])
        names = [n for n in dir(m) if not n.startswith("_")]
        print("%s -> %s" % (modname, names[:12]))
        for n in names:
            cls = getattr(m, n)
            if inspect.isclass(cls) and "Adam" in n or (inspect.isclass(cls) and "Opt" in n):
                try:
                    print("   %-20s %s" % (n, inspect.signature(cls.__init__)))
                except Exception:
                    pass
    except Exception as e:
        print("%s -> 不可用 (%s)" % (modname, type(e).__name__))
