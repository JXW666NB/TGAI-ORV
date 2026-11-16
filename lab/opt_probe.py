# -*- coding: utf-8 -*-
"""探测 ORV 优化器的构造签名与超参（在 V100 上用 D:\\orv_pip_0962 跑）。"""
import inspect

import tgai_go.train as T

for name in ("AdamW", "Fp16MomAdamW", "Adafactor", "SGD", "LAMB", "Lion"):
    cls = getattr(T.optim, name, None)
    if cls is None:
        print("%-14s <不存在>" % name)
        continue
    try:
        sig = inspect.signature(cls.__init__)
        print("%-14s %s" % (name, sig))
    except Exception as e:
        print("%-14s <签名不可读: %s>" % (name, e))
    doc = (cls.__doc__ or "").strip().splitlines()
    if doc:
        print("               doc: %s" % doc[0][:110])
