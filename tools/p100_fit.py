# -*- coding: utf-8 -*-
"""按 P100 用户的真实配置测 ORV 的峰值显存，判断 16GB 上能不能训。

## 用户的配置（来自其评估报告）

20 层 / d_model=1280 / FFN=5120 / 10 头（⇒ d_head=128）/ B=2..8 / S=2048，
开梯度检查点，400M 级参数。

他实测（0.9.49/0.50）：B=2 时 ORV 活张量已达 15.15 GiB ⇒ 16GB 卡 OOM；
而 torch fp16 同形状只要 8.99 GB。

## 本脚本要回答的问题

**0.9.52（修了检查点的 `_parents` 保留问题）之后，同配置的峰值是多少？能不能塞进 16 GiB？**

本机是 V100 32GB，所以即使超 16 GiB 也能测出来（不会因 OOM 中断），
这样得到的数字才谈得上"能不能"。

用法：python p100_fit.py [batch]
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_pkg_root(start):
    """向上找含 python/tgai_go 的目录，避免写死相对层数。"""
    cur = start
    for _ in range(6):
        if os.path.isdir(os.path.join(cur, "python", "tgai_go")):
            return os.path.join(cur, "python")
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    raise SystemExit("找不到 python/tgai_go（从 %s 向上找了 6 层）" % start)


sys.path.insert(0, _find_pkg_root(HERE))

OUT = os.path.join(HERE, "p100fit.txt")

#: 用户配置
VOCAB, D, L, NH, FF, S = 32000, 1280, 20, 10, 5120, 2048
DH = D // NH          # 128 —— 注意 sdpa_lowmem 只支持 64
NE, TOPK = 36, 6      # 他的 MoE 是 36 专家 top-6；这里先用 dense 对齐，MoE 另测


def main():
    bsz = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    import torch
    import tgai_go.train as T
    from tgai_go.train import functional as Fn

    T.set_default_device("cuda")
    rt = T._cuda()
    rep = []

    rng = np.random.default_rng(0)

    def dev(a):
        return T.Tensor._from_gpu(rt.h2d(np.ascontiguousarray(a, np.float32)))

    p = {"embed": dev(rng.normal(0, 0.02, (VOCAB, D))),
         "head": dev(rng.normal(0, 0.02, (VOCAB, D)))}
    p["attn"] = [dict(qkv=dev(rng.normal(0, 0.02, (3 * D, D))),
                      o=dev(rng.normal(0, 0.02, (D, D)))) for _ in range(L)]
    p["mlp"] = [dict(gate=dev(rng.normal(0, 0.02, (FF, D))),
                     up=dev(rng.normal(0, 0.02, (FF, D))),
                     down=dev(rng.normal(0, 0.02, (D, FF)))) for _ in range(L)]
    allp = [p["embed"], p["head"]] + [v for d in p["attn"] for v in d.values()] \
        + [v for m in p["mlp"] for v in m.values()]
    for t in allp:
        t.requires_grad = True

    n_par = sum(int(np.prod(tuple(t.shape))) for t in allp)
    rep.append("参数 %.1fM（含 embed+head %.1fM）"
               % (n_par / 1e6, (VOCAB * D * 2) / 1e6))
    rep.append("配置：L=%d D=%d NH=%d dh=%d FF=%d S=%d B=%d"
               % (L, D, NH, DH, FF, S, bsz))

    ids = rng.integers(0, VOCAB, (bsz, S)).astype(np.int64)
    emb_layer = T.nn.Embedding(VOCAB, D, weight=T.Parameter(np.zeros((VOCAB, D), np.float32)))
    emb_layer.weight._gpu = p["embed"]._gpu
    emb = emb_layer(ids.reshape(-1)).reshape(bsz, S, D)

    def block(x, li):
        B, Sq, _ = x.shape
        qkv = x.reshape(B * Sq, D) @ p["attn"][li]["qkv"].T
        q, k, v = qkv.reshape(B, Sq, 3, NH, DH).permute(2, 0, 3, 1, 4)
        q = q.reshape(B * NH, Sq, DH)
        k = k.reshape(B * NH, Sq, DH)
        v = v.reshape(B * NH, Sq, DH)
        # is_causal=True；lowmem 会自动判断（dh=128 ⇒ 走物化路径）
        o = Fn.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=True)
        o = o.reshape(B, NH, Sq, DH).permute(0, 2, 1, 3).reshape(B * Sq, D)
        x = x + (o @ p["attn"][li]["o"].T).reshape(B, Sq, D)
        flat = x.reshape(B * Sq, D)
        m = p["mlp"][li]
        h = Fn.silu(flat @ m["gate"].T) * (flat @ m["up"].T)
        h = h @ m["down"].T
        return x + h.reshape(B, Sq, D)

    def step():
        x = emb
        for li in range(L):
            x = T.checkpoint(lambda h, _l=li: block(h, _l), x)
        lg = (x.reshape(bsz * S, D) @ p["head"].T).reshape(bsz, S, VOCAB)
        return Fn.cross_entropy(lg.reshape(-1, VOCAB), ids.reshape(-1))

    st0 = rt.pool_stats()
    rep.append("起始池占用 %.2f GiB（活 %.2f）"
               % (st0["occupied_bytes"] / (1 << 30), st0["live_bytes"] / (1 << 30)))
    try:
        loss = step()
        loss.backward()
        rt.sync()
        st = rt.pool_stats()
        live = st["peak_live_bytes"] / (1 << 30)
        incl = st["peak_bytes"] / (1 << 30)
        rep.append("一步后：峰值活张量 **%.2f GiB**（含池缓存 %.2f GiB）" % (live, incl))
        rep.append("16 GiB 卡结论：%s" % ("**能跑**" if incl < 16.0 * 0.95 else
                                          "**跑不了**（超出 16 GiB 的安全余量）"))
        rep.append("loss=%r" % float(np.asarray(loss.to_numpy(), np.float32)))
    except Exception as ex:      # noqa: BLE001
        st = rt.pool_stats()
        rep.append("失败：%s: %s" % (type(ex).__name__, str(ex)[:200]))
        rep.append("失败时活张量 %.2f GiB" % (st["live_bytes"] / (1 << 30)))

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
