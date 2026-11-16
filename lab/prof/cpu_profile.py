# -*- coding: utf-8 -*-
"""CPU 侧热点：一步训练在 Python 里把时间花在哪。

前置结论：1 层模型的 fwd+bwd GPU 时间只有 6.3 ms，墙钟 199 ms
⇒ 97% 是 CPU。本脚本用 cProfile 找出具体热点函数。

跑法：
    D:\\orv_pip_0962\\Scripts\\python.exe -m cProfile -s tottime cpu_profile.py --layers 1
"""
import argparse

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--n_q", type=int, default=16)
    ap.add_argument("--d_ff", type=int, default=2816)
    ap.add_argument("--vocab", type=int, default=96000)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--csa", type=int, default=4)
    a = ap.parse_args()

    cfg = dict(d=a.d, d_ff=a.d_ff, E=a.E, layers=a.layers, n_q=a.n_q, n_kv=2,
               vocab=a.vocab, seq=a.seq, attn_res=False, attn_res_block=2,
               top_k=2, csa=a.csa, n_kv_attn=a.n_q, rope_theta=500000.0,
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

    def step():
        logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
        loss = T.functional.cross_entropy(logits.reshape(seq, cfg["vocab"]), tgt)
        for p in params:
            p.grad = None
        loss.backward()
        opt.step()
        rt.sync()

    # 给 h2d / d2h 打尺寸日志（找出是谁在搬大数组）
    import collections
    H2D = collections.Counter()
    _orig_h2d = rt.h2d
    TRACED = [0]
    def h2d(buf, *ar, **kw):
        try:
            shp = tuple(getattr(buf, "shape", ()))
            H2D[(shp, str(getattr(buf, "dtype", "")))] += 1
            if len(shp) == 2 and shp[0] > 50000 and TRACED[0] < 2:
                TRACED[0] += 1
                import traceback
                print("### 大数组 h2d %s 的调用栈 ###" % (shp,), flush=True)
                traceback.print_stack()
        except Exception:
            pass
        return _orig_h2d(buf, *ar, **kw)
    rt.h2d = h2d

    step()                      # 预热（build / NVRTC 编译都在这里，不能被算进 profile）
    print("### WARM DONE ###", flush=True)

    # 只 profile 第二个 step：用 Profile 对象显式开关，
    # 避免 `-m cProfile` 把预热期的 build(3.2 s) 与 NVRTC 编译(2.35 s) 也算进来
    import cProfile
    import io
    import pstats

    pr = cProfile.Profile()
    pr.enable()
    step()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(22)
    print(s.getvalue())
    print("### h2d 调用明细 ###")
    for k, c in H2D.most_common(12):
        print("  %-30s x%d" % (str(k), c))


if __name__ == "__main__":
    main()
