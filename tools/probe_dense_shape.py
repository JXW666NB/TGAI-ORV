# -*- coding: utf-8 -*-
"""复现 P100 群友的形状（稠密 Transformer），并消融出 attention / FFN 各占多少。

形状：21 层 / d=1152 / 18 头 × 64 / FF=2560 / S=2048 / B 可变。
目的：回答"ORV 在大 batch 上为什么不随 batch 伸缩"。
"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V  # noqa: E402


def build_layer(rt, d, ff, n_q, n_kv, seed):
    rng = np.random.default_rng(seed)
    s = 1.0 / np.sqrt(d)
    mk = lambda *sh: T.tensor(rng.normal(0, s, sh).astype(np.float32))
    return dict(
        wq=mk(d, d), wk=mk(d, d), wv=mk(d, d), wo=mk(d, d),
        W1=mk(ff, d), W2=mk(ff, d), W3=mk(d, ff),
        ln1=T.tensor(np.ones(d, np.float32)), ln2=T.tensor(np.ones(d, np.float32)))


def main():
    T.set_default_device("cuda")
    rt = T._cuda()
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    L = int(sys.argv[3]) if len(sys.argv) > 3 else 21
    d, ff, n_q, n_kv = 1152, 2560, 18, 6
    d_k = d // n_q
    print("形状 S=%d B=%d L=%d d=%d ff=%d heads=%d dh=%d"
          % (S, B, L, d, ff, n_q, d_k), flush=True)

    layers = [build_layer(rt, d, ff, n_q, n_kv, seed=i) for i in range(L)]
    cos, sin = V.rope_cos_sin(S, d_k, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)

    def attn(x, Lw):
        h = V.rmsnorm_orv(x, Lw["ln1"])
        return h + V.attention_orv(h, Lw["wq"], Lw["wk"], Lw["wv"], Lw["wo"],
                                   cos, sin, d_k, n_q, n_kv)

    def ffn(x, Lw):
        h = V.rmsnorm_orv(x, Lw["ln2"]).reshape(-1, d)
        y = V.situ_glu_orv(h, Lw["W1"], Lw["W2"], Lw["W3"])
        return h.reshape(1, -1, d) + y.reshape(1, -1, d)

    def step(use_attn=True, use_ffn=True, ckpt=True):
        x = T.tensor(np.zeros((1, B * S, d), np.float32))
        for Lw in layers:
            if use_attn:
                if ckpt:
                    x = x + T.checkpoint(lambda h, l=Lw: V.attention_orv(
                        V.rmsnorm_orv(h, l["ln1"]), l["wq"], l["wk"], l["wv"],
                        l["wo"], cos, sin, d_k, n_q, n_kv), x)
                else:
                    x = attn(x, Lw)
            if use_ffn:
                if ckpt:
                    x = x + T.checkpoint(lambda h, l=Lw: V.situ_glu_orv(
                        V.rmsnorm_orv(h, l["ln2"]).reshape(-1, d),
                        l["W1"], l["W2"], l["W3"]
                    ).reshape(h.shape), x)
                else:
                    x = ffn(x, Lw)
        return (x * x).sum()

    def bench(label, **kw):
        try:
            l = step(**kw)
            l.backward()
            rt.sync()
            rt.empty_cache()
            t0 = time.time()
            l = step(**kw)
            l.backward()
            rt.sync()
            dt = time.time() - t0
            tok = B * S / dt
            live = rt.pool_stats()["live_bytes"] / 2**20
            print("%-18s %.2f s   %6.0f tok/s   live=%.0f MiB" % (label, dt, tok, live),
                  flush=True)
        except Exception as exc:
            print("%-18s 失败：%s" % (label, str(exc)[:110]), flush=True)

    bench("全量(检查点)")
    bench("只 attention", use_ffn=False)
    bench("只 FFN", use_attn=False)
    bench("全量(无检查点)", ckpt=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
