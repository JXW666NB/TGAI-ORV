# -*- coding: utf-8 -*-
"""测：`_acc` 的多次贡献分布在多少个**不同节点**上。

这决定"梯度累加能否批量化"：
* 若集中在少数节点（每节点 10+ 次）⇒ 可把该节点的 N 次 add 合成 1 个多源求和内核；
* 若分散在很多节点（每节点 2 次）⇒ 每节点都必须在它被消费前算完，
  按反向拓扑序无法跨节点批量，省不下来。
"""
import argparse
import collections

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()

HITS = collections.Counter()
_orig = T._acc


def acc(grads, node, g):
    HITS[id(node)] += 1
    return _orig(grads, node, g)


T._acc = acc


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
        HITS.clear()
        logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
        loss = T.functional.cross_entropy(logits.reshape(512, cfg["vocab"]), tgt)
        for p in params:
            p.grad = None
        loss.backward()
        opt.step()
        rt.sync()

    total = sum(HITS.values())
    multi = {k: v for k, v in HITS.items() if v >= 2}
    extra = sum(v - 1 for v in multi.values())
    hist = collections.Counter(HITS.values())
    print("=== _acc 调用分布（一步）===")
    print("  总调用次数        : %d" % total)
    print("  不同节点数        : %d" % len(HITS))
    print("  被多次累加的节点数: %d" % len(multi))
    print("  累加次数合计(超出首份的部分): %d   ← 这就是 fused_add_ip 的次数" % extra)
    print("\n  每节点贡献次数分布（次数 -> 节点数）：")
    for c in sorted(hist):
        print("    %2d 次 -> %3d 个节点" % (c, hist[c]))
    if multi:
        top = sorted(multi.items(), key=lambda kv: -kv[1])[:5]
        print("\n  贡献最多的 5 个节点（每节点次数）：%s"
              % ", ".join(str(v) for _k, v in top))


if __name__ == "__main__":
    main()
