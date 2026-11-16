# -*- coding: utf-8 -*-
"""探测 ORV 的 AMP 接口：T.amp_on / f16 缓存 / amp_mlp。"""
import tgai_go.train as T

rt = T._cuda()
print("T.amp_* :", [n for n in dir(T) if n.startswith("amp")])
print("rt.f16_*:", [n for n in dir(rt) if "f16" in n.lower()])

for n in ("amp_on", "amp_off", "amp_state"):
    f = getattr(T, n, None)
    if f is None:
        print("%-12s <不存在>" % n)
        continue
    try:
        import inspect
        print("%-12s %s" % (n, inspect.signature(f)))
    except Exception as e:
        print("%-12s <签名不可读 %s>" % (n, e))

try:
    from tgai_go.train import amp_mlp
    print("amp_mlp:", [n for n in dir(amp_mlp) if not n.startswith("_")][:12])
except Exception as e:
    print("amp_mlp: 不可用", type(e).__name__)

try:
    st = T.amp_state()
    print("amp_state():", {k: st[k] for k in list(st)[:8]})
except Exception as e:
    print("amp_state() 失败:", e)
