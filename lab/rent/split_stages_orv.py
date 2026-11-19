# -*- coding: utf-8 -*-
"""ORV 的流程拆解：forward / backward / optimizer 三段。

做法：包装模块级 `T.backward`（训练器每步调用一次）与优化器的 `step`，
前后各 sync 一次计时；总步时取训练器自己报告的 `step_s_steady`，
forward = 总 − backward − optimizer（差值法，避免重复计算同步开销）。
注意力（csa_forward）另外单独计，用于看它在 forward 里的占比。
"""
import atexit
import json
import sys
import time

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_pretrain as TR

rt = T._cuda()
acc = {"bwd": 0.0, "opt": 0.0, "attn": 0.0, "moe": 0.0, "steps": 0,
       "bwd_n": 0, "opt_n": 0, "attn_n": 0, "moe_n": 0}

# 训练器调的是 `loss.backward()`（Tensor 的方法），不是模块级函数 ⇒ 要包类方法
_T = None
for _nm in ("Tensor", "_Tensor"):
    if hasattr(T, _nm):
        _T = getattr(T, _nm)
        break
if _T is None:
    for _v in vars(T).values():
        if isinstance(_v, type) and hasattr(_v, "backward"):
            _T = _v
            break
if _T is None:
    raise SystemExit("找不到 Tensor 类，无法挂 backward 计时")
_bwd = _T.backward


def _bwd_timed(*a, **kw):
    rt.sync()
    t0 = time.perf_counter()
    out = _bwd(*a, **kw)
    rt.sync()
    acc["bwd"] += time.perf_counter() - t0
    acc["bwd_n"] += 1
    return out


_T.backward = _bwd_timed

# MoE（专家 + 路由）单独计时：forward 里的大头之一
_MoE = getattr(T, "MoE", None)
if _MoE is not None:
    _moe_fwd = _MoE.forward

    def _moe_timed(self, *a, **kw):
        rt.sync()
        t0 = time.perf_counter()
        out = _moe_fwd(self, *a, **kw)
        rt.sync()
        acc["moe"] += time.perf_counter() - t0
        acc["moe_n"] += 1
        return out

    _MoE.forward = _moe_timed


_attn = C.csa_forward


def _attn_timed(*a, **kw):
    rt.sync()
    t0 = time.perf_counter()
    out = _attn(*a, **kw)
    rt.sync()
    acc["attn"] += time.perf_counter() - t0
    acc["attn_n"] += 1
    return out


C.csa_forward = _attn_timed


def _report():
    print("[split] " + json.dumps(acc), flush=True)


atexit.register(_report)

if __name__ == "__main__":
    TR.main(sys.argv[1:])
