# -*- coding: utf-8 -*-
"""探测 Muon 的构造签名、文档与是否自带分组策略。"""
import inspect

import tgai_go.train as T

cls = T.optim.Muon
print("signature:", inspect.signature(cls.__init__))
print("--- doc ---")
print((cls.__doc__ or "")[:1200])
print("--- members ---")
print([n for n in dir(cls) if not n.startswith("_")])
