# -*- coding: utf-8 -*-
"""查清 --grad-ckpt 为什么没生效：参数解析到了吗？main 里 cfg 从哪来？"""
import inspect

from tgai_go.train import v3_pretrain as P

ap = P.build_arg_parser()
a = ap.parse_args(["--data", r"D:\TGAI-V3-1.5B\01_corpus\packed512\s512",
                   "--steps", "1", "--grad-ckpt"])
print("a.grad_ckpt =", getattr(a, "grad_ckpt", "<缺失>"))

# main 里出现的 cfg 相关行
src = inspect.getsource(P.main)
lines = [l.rstrip() for l in src.splitlines() if "cfg" in l]
print("\nmain() 里与 cfg 有关的行（前 12 条）:")
for l in lines[:12]:
    print("   " + l.strip()[:110])

# 找出构建 cfg 的函数，并直接调用它验证
for name in ("_build_cfg", "build_cfg", "_cfg_from_args"):
    fn = getattr(P, name, None)
    if fn is not None:
        try:
            cfg = fn(a)
            print("\n%s(a) -> cfg.get('grad_ckpt') = %r" % (name, cfg.get("grad_ckpt")))
        except Exception as e:
            print("\n%s 调用失败: %r" % (name, e))
