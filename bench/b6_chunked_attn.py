# -*- coding: utf-8 -*-
"""b6：用**纯 ORV 算子**的分块注意力替代物化 S²，在群友形状上对比显存与速度。

背景：P100 上融合内核编不出来（68 KB smem > 48 KB 上限），只能物化 S²，
B=4 时占 4.18 GB。但"不物化 S²"不一定非要写内核 —— 把 query 切块，
每块只与截至自己的 key 相乘，分数矩阵就从 B·H·S² 降到 B·H·C·S。

对比四档：物化 / 融合 / **分块(C=256)** / 分块(C=512)
"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import functional as Fn  # noqa: E402


def _rect_mask(rows, cols, q_offset):
    r = np.arange(rows)[:, None] + int(q_offset)
    c = np.arange(cols)[None, :]
    return np.where(c <= r, 0.0, -np.inf).astype(np.float32)[None, :, :]


def chunked(q, k, v, chunk):
    """q,k,v: [BH, S, DH]（3D，与 sdpa 一致）。按 query 块算，掩码矩形可加。"""
    BH, S, DH = q.shape
    outs = []
    C = int(chunk)
    for st in range(0, S, C):
        en = min(st + C, S)
        qb = q[:, st:en, :]
        kb = k[:, :en, :]
        vb = v[:, :en, :]
        m = _rect_mask(en - st, en, st)                 # (1, rows, cols)
        am = np.repeat(m, BH, axis=0)
        ob = Fn.scaled_dot_product_attention(
            qb, kb, vb, causal=False, attn_mask=T.tensor(np.ascontiguousarray(am)))
        outs.append(ob)
    return outs[0] if len(outs) == 1 else T.cat(outs, dim=1)


def bench(kind, BH, S, DH, chunk=None, reps=5):
    rng = np.random.default_rng(0)
    q = T.tensor(rng.normal(0, .3, (BH, S, DH)).astype(np.float32), requires_grad=True)
    k = T.tensor(rng.normal(0, .3, (BH, S, DH)).astype(np.float32), requires_grad=True)
    v = T.tensor(rng.normal(0, .3, (BH, S, DH)).astype(np.float32), requires_grad=True)
    rt = T._cuda()

    def one():
        if kind == "materialized":
            y = Fn.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=False)
        elif kind == "fused":
            y = Fn.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=True)
        else:
            y = chunked(q, k, v, chunk)
        (y * y).sum().backward()

    one()
    rt.sync()
    rt.empty_cache()
    t0 = time.time()
    for _ in range(reps):
        one()
    rt.sync()
    dt = (time.time() - t0) / reps
    st = rt.pool_stats()
    label = kind if chunk is None else "%s(C=%d)" % (kind, chunk)
    print("  %-18s %6.3f s  峰值 %6.2f GB" % (label, dt, st["peak_live_bytes"] / 2**30),
          flush=True)


def main():
    T.set_default_device("cuda")
    B, S, NH, DH = 4, 2048, 12, 64
    BH = B * NH
    print("注意力形状：BH=%d S=%d DH=%d（B=4, 12 头）" % (BH, S, DH), flush=True)
    bench("materialized", BH, S, DH)
    try:
        bench("fused", BH, S, DH)
    except Exception as exc:
        print("  fused              失败：%s" % str(exc)[:70], flush=True)
    for c in (256, 512):
        try:
            bench("chunked", BH, S, DH, chunk=c)
        except Exception as exc:
            print("  chunked(C=%d)      失败：%s" % (c, str(exc)[:70]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
