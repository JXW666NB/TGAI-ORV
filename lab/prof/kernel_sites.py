# -*- coding: utf-8 -*-
"""按**调用点**归因内核派发次数：这 1166 个内核分别是谁发的。

内核总数只知道"多"，不知道"该改哪里"。本脚本给 `launch` 挂桩，
记录每个内核的调用栈里**第一个不在 tgai_go/train 下的帧**，
从而把内核归到模型/损失/优化器的具体代码位置。
"""
import argparse
import collections
import os
import traceback

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()

BY_SITE = collections.Counter()
BY_PAIR = collections.Counter()
_orig_launch = rt.launch


def _site():
    """找**库里最内层的语义调用者**。

    从内往外走：跳过 `cuda.py`（那是内核发射器本身），
    停在第一个位于 `tgai_go/train` 下、但不是 `cuda.py` 的帧 ——
    也就是 `__init__.py` / `v3_ops.py` / `v3_moe.py` 里的具体行。
    """
    for fr in reversed(traceback.extract_stack()[:-2]):
        fn = fr.filename.replace("\\", "/")
        if "/tgai_go/train/cuda.py" in fn:
            continue
        if "/tgai_go/train/" in fn:
            return "%s:%d" % (os.path.basename(fn), fr.lineno)
        return "ext:%s:%d" % (os.path.basename(fn), fr.lineno)   # 逃出库了
    return "<cuda-only>"


def launch(name, grid, block, args=None, *ar, **kw):
    s = _site()
    BY_SITE[s] += 1
    BY_PAIR[(s, str(name))] += 1
    return _orig_launch(name, grid, block, args, *ar, **kw)


rt.launch = launch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--d_ff", type=int, default=2816)
    a = ap.parse_args()

    cfg = dict(d=a.d, d_ff=a.d_ff, E=4, layers=a.layers, n_q=16, n_kv=2,
               vocab=96000, seq=a.seq, attn_res=False, attn_res_block=2,
               top_k=2, csa=4, n_kv_attn=16, rope_theta=500000.0,
               csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    rng = np.random.default_rng(0)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    params = CLI.params_of(P)
    opt = T.optim.AdamW(params, lr=3e-4)
    seq = int(cfg["seq"])
    csac = cfg["csa_cfg"]
    plan = C.make_plan(csac, 1, seq)
    cos, sin = C.rope_tables(csac, seq)
    ids = np.random.default_rng(1).integers(0, cfg["vocab"], seq).astype(np.int64)
    tgt = np.roll(ids, -1)

    def one_step(clear):
        if clear:
            BY_SITE.clear()
            BY_PAIR.clear()
        logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
        n_fwd = sum(BY_SITE.values())
        loss = T.functional.cross_entropy(logits.reshape(seq, cfg["vocab"]), tgt)
        for p in params:
            p.grad = None
        loss.backward()
        opt.step()
        rt.sync()
        return n_fwd

    one_step(False)                       # 预热
    n_fwd = one_step(True)                # 计数这一步
    total = sum(BY_SITE.values())

    print("=== 一步共 %d 个内核（前向 %d / 反向+优化器 %d）==="
          % (total, n_fwd, total - n_fwd))
    print("\n按调用点（前 16）：")
    for site, n in BY_SITE.most_common(16):
        print("  %-26s %5d  (%.1f%%)" % (site, n, n * 100.0 / total))
    print("\n(调用点, 内核) 最多的 14 组：")
    for (site, name), n in BY_PAIR.most_common(14):
        print("  %-26s %-20s %4d" % (site, name, n))


if __name__ == "__main__":
    main()
