# -*- coding: utf-8 -*-
"""CUDA Graph 探针：捕获「前向+反向」，与逐算子对拍步时与 loss。"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE  # noqa: E402
from tgai_go.train.graph_train import GraphStep  # noqa: E402
from v3_orv_train import build, params_of  # noqa: E402


class Adapter:
    """把函数式模型包成 GraphStep 需要的 model 接口。"""

    def __init__(self, P, cos, sin, d_k, n_q, n_kv, S, d, vocab, top_k=2):
        self.P, self.cos, self.sin = P, cos, sin
        self.d_k, self.n_q, self.n_kv = d_k, n_q, n_kv
        self.S, self.d, self.vocab, self.top_k = S, d, vocab, top_k

    def forward(self, ids):
        P, S, d = self.P, self.S, self.d
        x = P["emb"][ids].reshape(1, S, d)
        for L in P["layers"]:
            x = x + V.attention_orv(V.rmsnorm_orv(x, L["ln1"]), L["wq"], L["wk"],
                                    L["wv"], L["wo"], self.cos, self.sin,
                                    self.d_k, self.n_q, self.n_kv)
            h = x
            hn = V.rmsnorm_orv(h, L["ln2"]).reshape(S, d)
            e, g, _ = MOE.route_topk(hn, L["router"], self.top_k)
            mo = MOE.moe_topk_orv(hn, e, g, L["W1"], L["W2"], L["W3"])
            mo = mo + V.situ_glu_orv(hn, L["S1"], L["S2"], L["S3"])
            x = h + mo.reshape(1, S, d)
        lg = V.rmsnorm_orv(x, P["ln_f"]).reshape(S, d) @ P["head"].transpose(0, 1)
        return lg.reshape(1, S, self.vocab)

    def params(self):
        return params_of(self.P)


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    d, S, E, L, n_q, n_kv, vocab = 512, 2048, 8, 8, 8, 2, 4096
    cfg = dict(d=d, d_ff=4 * d, E=E, layers=L, n_q=n_q, n_kv=n_kv, vocab=vocab,
               seq=S, steps=1)
    rng = np.random.default_rng(0)
    P = build(cfg, rng)
    cos, sin = V.rope_cos_sin(S, d // n_q, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    ids = rng.integers(0, vocab, (1, S)).astype(np.int64)
    tgt = np.roll(ids, -1, axis=1)

    opt = T.optim.AdamW(params_of(P), lr=1e-4, betas=(0.9, 0.95), eps=1e-8,
                        weight_decay=0.0)
    model = Adapter(P, cos, sin, d // n_q, n_q, n_kv, S, d, vocab)

    gs = GraphStep(model, opt, (1, S), (1, S), vocab, arena_gib=8.0)

    # 逐算子基准
    for _ in range(2):
        gs.step_eager(ids, tgt)
    rt.sync()
    t0 = time.time()
    for _ in range(3):
        gs.step_eager(ids, tgt)
    rt.sync()
    eager = (time.time() - t0) / 3
    print("逐算子  %.3f s/步" % eager)

    # 捕获
    t0 = time.time()
    nodes = gs.capture(warmup=2)
    rt.sync()
    print("捕获完成 nodes=%s 耗时 %.1f s" % (nodes, time.time() - t0))

    # 图重放
    for _ in range(2):
        gs.step(ids, tgt)
    rt.sync()
    t0 = time.time()
    for _ in range(3):
        gs.step(ids, tgt)
    rt.sync()
    graph = (time.time() - t0) / 3
    print("图重放  %.3f s/步  加速 %.2fx" % (graph, eager / graph))
    print("live=%.0f MiB" % (rt.pool_stats()["live_bytes"] / 2**20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
