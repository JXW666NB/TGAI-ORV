# -*- coding: utf-8 -*-
"""驻留量探针：`loss.backward()` **之前**，显存里还活着的缓冲按形状分布。

为什么需要它：之前用"前向创建过的张量"统计（`act_shapes.py`）得 1.93 GiB，
而 `pool.peak_live` 是 6.80 GiB —— 那个口径把视图和已释放的中间量都算进去了，
**不能用来决定"该量化谁"**。

做法：给 `GpuBuf` 的 `__init__`/`__del__` 挂钩子维护登记表（id -> (shape, bytes)），
在 backward 之前聚合打印。不改库，纯脚本。
"""
import argparse
import collections

import numpy as np

import tgai_go.train as T
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI

T.set_default_device("cuda")
rt = T._cuda()
CLS = type(rt)
GB = type(rt.zeros((1,), dtype=np.float32))

REG = {}
_o_init = GB.__init__
_o_del = GB.__del__


def g_init(self, *a, **kw):
    _o_init(self, *a, **kw)
    try:
        REG[id(self)] = (tuple(int(s) for s in self.shape), int(self.n) * 4,
                         int(self.ptr))
    except Exception:
        pass


def g_del(self):
    REG.pop(id(self), None)
    return _o_del(self)


GB.__init__ = g_init
GB.__del__ = g_del


def snapshot(tag, top=14):
    # ★ 必须按**指针**去重：`sub_view`/`reshape` 会在同一块显存上创建多个 GpuBuf
    #   包装对象，按对象计数会把同一块内存重复算好几次（我第一版就是这么错的，
    #   结果看到"120 个权重"这种不可能的数字）。
    uniq = {}
    for (shp, b, ptr) in REG.values():
        uniq[ptr] = (shp, b)
    by_shape = collections.Counter()
    cnt = collections.Counter()
    tot = 0
    for shp, b in uniq.values():
        by_shape[shp] += b
        cnt[shp] += 1
        tot += b
    gi = float(1 << 30)
    print("  [%s] 活缓冲 %d 个对象 / **去重后 %d 块**，合计 %.3f GiB"
          % (tag, len(REG), len(uniq), tot / gi))
    for shp, b in by_shape.most_common(top):
        print("      %-26s x%-4d %9.2f MB" % (str(shp), cnt[shp], b / (1 << 20)))
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    a = ap.parse_args()

    cfg = dict(d=1024, d_ff=2816, E=4, layers=a.layers, n_q=16, n_kv=2,
               vocab=96000, seq=a.seq, attn_res=False, attn_res_block=2,
               top_k=2, csa=4, n_kv_attn=16, rope_theta=500000.0,
               csa_compress=4, csa_conv_kernel=3, csa_local_window=0)
    rng = np.random.default_rng(0)
    P = CLI.build(cfg, rng)
    kinds = CLI.layer_kinds(cfg)
    params = CLI.params_of(P)
    opt = T.optim.AdamW(params, lr=3e-4)
    csac = cfg["csa_cfg"]
    seq = int(cfg["seq"])
    plan = C.make_plan(csac, 1, seq)
    cos, sin = C.rope_tables(csac, seq)
    ids = np.random.default_rng(1).integers(0, cfg["vocab"], seq).astype(np.int64)
    tgt = np.roll(ids, -1)

    for step in range(2):
        for p in params:
            p.grad = None
        logits = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
        loss = T.functional.cross_entropy(logits.reshape(seq, cfg["vocab"]), tgt)
        rt.sync()
        if step == 0:
            # ★ 必须取 step 0：此时 AdamW 的 m/v 还没分配（它们在第 1 次 step() 才建）。
            #   我第一版在 step 1 取，结果把 m/v 当成"每权重三份拷贝"，
            #   误判成"权重被复制 3~4 份"。
            print("=== 4 层 / seq=%d：backward 之前（m/v 未分配）的驻留 ===" % seq)
            snapshot("backward 前")
        loss.backward()
        opt.step()
        rt.sync()
    print("\n  pool peak_live = %.3f GiB" % (rt.pool_stats()["peak_live_bytes"] / float(1 << 30)))


if __name__ == "__main__":
    main()
