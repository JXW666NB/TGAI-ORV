# -*- coding: utf-8 -*-
"""ORV 组件拆解：给 `v3_csa.csa_forward` 挂计时包装，再跑**真实训练器**。

要点：调用方是 `C.csa_forward(...)`（属性查找发生在调用时），所以 monkeypatch 生效；
包装里前后各 sync 一次，只用来算**注意力占比**，不把总步时当真。
用法：python split_orv.py --data D:\orv_bench\data2048 --seq 1024 --layers 12 ...
"""
import atexit
import json
import sys
import time

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_pretrain as TR

rt = T._cuda()
acc = {"attn": 0.0, "n": 0}
_orig = C.csa_forward


def _timed(*a, **kw):
    rt.sync()
    t0 = time.perf_counter()
    out = _orig(*a, **kw)
    rt.sync()
    acc["attn"] += time.perf_counter() - t0
    acc["n"] += 1
    return out


C.csa_forward = _timed


def _report():
    print("[split] " + json.dumps({"attn_calls": acc["n"],
                                   "attn_s_total": acc["attn"]}), flush=True)


atexit.register(_report)

if __name__ == "__main__":
    TR.main(sys.argv[1:])
