# -*- coding: utf-8 -*-
"""抓 (2816,1024) 的分配现场。

retention.py 显示 4 层 / seq512 的 forward 结束时，形状 (2816,1024) 有 **120 块不同显存**
（专家权重只应有 40 块），且 step 0 就已如此（不是优化器状态）。
这里给 `rt.empty` 挂桩，只对这一个形状记录**调用栈里第一个库内非 cuda.py 的帧**。
"""
import argparse
import collections
import traceback

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()
CLS = type(rt)

TARGET = (2816, 1024)
SITES = collections.Counter()
ON = [False]

_o_empty = CLS.empty


def site():
    for fr in reversed(traceback.extract_stack()[:-2]):
        fn = fr.filename.replace("\\", "/")
        if "/tgai_go/train/cuda.py" in fn:
            continue
        if "/tgai_go/train/" in fn:
            return "%s:%d" % (fn.rsplit("/", 1)[-1], fr.lineno)
        return "ext:%s:%d" % (fn.rsplit("/", 1)[-1], fr.lineno)
    return "<cuda-only>"


def empty_hook(self, shape, *a, **kw):
    if ON[0]:
        try:
            shp = tuple(int(s) for s in shape)
        except Exception:
            shp = None
        if shp == TARGET:
            SITES[site()] += 1
    return _o_empty(self, shape, *a, **kw)


CLS.empty = empty_hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    a = ap.parse_args()

    cfg = dict(d=1024, d_ff=2816, E=4, layers=a.layers, n_q=16, n_kv=2,
               vocab=96000, seq=512, attn_res=False, attn_res_block=2,
               top_k=2, csa=4, n_kv_attn=16, rope_theta=500000.0,
               csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    rng = np.random.default_rng(0)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    params = CLI.params_of(P)
    opt = T.optim.AdamW(params, lr=3e-4)
    csac = cfg["csa_cfg"]
    plan = C.make_plan(csac, 1, 512)
    cos, sin = C.rope_tables(csac, 512)
    ids = np.random.default_rng(1).integers(0, cfg["vocab"], 512).astype(np.int64)
    tgt = np.roll(ids, -1)

    for _ in range(2):
        ON[0] = False
        SITES.clear()
        ON[0] = True
        logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
        ON[0] = False
        loss = T.functional.cross_entropy(logits.reshape(512, cfg["vocab"]), tgt)
        for p in params:
            p.grad = None
        loss.backward()
        opt.step()
        rt.sync()

    print("=== forward 里分配 (2816,1024) 的调用点（共 %d 次）===" % sum(SITES.values()))
    for s, n in SITES.most_common(10):
        print("  %-30s %5d" % (s, n))


if __name__ == "__main__":
    main()
