# -*- coding: utf-8 -*-
"""b5：把 v3exp2 的每一块**单独拆出来**跑，看显存到底被谁吃掉。

用法：
    python bench/b5_module_memory.py <模块> [B]
    模块 ∈ {emb, attn, dense, moe, head, all}

为什么要一个模块一个进程：
    ORV 的分配器有 live/peak 计数，但**没有 reset**。同一进程里先后测多块，
    peak 是单调不减的，后测的模块会"继承"前面模块的峰值 —— 读数没有意义。
    所以每块单独起一个进程，读到的是"只有这一块时"的真实峰值。

形状取自群友的 v3exp2（与他脚本一致）：
    V=36864, D=1152, S=2048, 12 头 × 64（dh=64）, MoE 36 专家 top_k=5 n_shared=1, FF=2560
"""
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import functional as Fn  # noqa: E402

V, D, S, H, DH = 36864, 1152, 2048, 12, 64
INNER = H * DH
FF = 2560
N_EXP, TOP_K, D_FF, N_SHARED = 36, 5, 256, 1


def weights(rt, which):
    """按模块只建它需要的权重 —— 这样峰值里不会有别的模块的权重。"""
    rng = np.random.default_rng(0)

    def dev(n, m):
        return T.Tensor._from_gpu(rt.h2d(rng.normal(0, 0.02, (n, m)).astype(np.float32)))

    w = {}
    if which in ("emb", "all"):
        w["emb"] = T.nn.Embedding(V, 768)
        w["up0"] = dev(D, 768)
    if which in ("attn", "all"):
        w.update(q=dev(INNER, D), kv=dev(256, D), ku=dev(INNER, 256),
                 vu=dev(INNER, 256), o=dev(D, INNER))
    if which in ("dense", "all"):
        w.update(w1=dev(FF, D), w2=dev(D, FF))
    if which in ("moe", "all"):
        w["moe"] = T.nn.MoE(D, D_FF, N_EXP, top_k=TOP_K, n_shared=N_SHARED,
                            shared_d_ff=576)
    if which in ("head", "all"):
        w["head"] = dev(V, D)
    return w


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    T.set_default_device("cuda")
    rt = T._cuda()
    w = weights(rt, which)
    for v in w.values():
        if hasattr(v, "requires_grad"):
            v.requires_grad = True
        elif hasattr(v, "parameters"):
            for p in v.parameters():
                p.requires_grad = True
    rt.sync()
    w_live = rt.pool_stats()["live_bytes"] / 2**20

    rng = np.random.default_rng(1)
    ids = T.tensor(rng.integers(0, V, (B, S)).astype(np.int64))
    tgt = T.tensor(rng.integers(0, V, (B, S)).astype(np.int64))
    x0 = T.tensor(rng.normal(0, 0.02, (B, S, D)).astype(np.float32))

    def run():
        if which == "emb":
            x = w["emb"](ids.reshape(-1)).reshape(B, S, 768)
            y = (x.reshape(-1, 768) @ w["up0"].T).reshape(B, S, D)
        elif which == "attn":
            x = x0
            z = x.reshape(-1, D)
            q = (z @ w["q"].T).reshape(B, S, H, DH).permute(0, 2, 1, 3)
            kv = z @ w["kv"].T
            k = (kv @ w["ku"].T).reshape(B, S, H, DH).permute(0, 2, 1, 3)
            v = (kv @ w["vu"].T).reshape(B, S, H, DH).permute(0, 2, 1, 3)
            yy = Fn.scaled_dot_product_attention(q.reshape(B * H, S, DH),
                                                 k.reshape(B * H, S, DH),
                                                 v.reshape(B * H, S, DH),
                                                 is_causal=True, lowmem=False)
            yy = yy.reshape(B, H, S, DH).permute(0, 2, 1, 3).reshape(-1, INNER)
            y = x + (yy @ w["o"].T).reshape(B, S, D)
        elif which == "dense":
            f = x0.reshape(-1, D)
            y = x0 + (Fn.silu(f @ w["w1"].T) @ w["w2"].T).reshape(B, S, D)
        elif which == "moe":
            y = x0 + w["moe"](x0.reshape(-1, D)).reshape(B, S, D)
        elif which == "head":
            y = (x0.reshape(-1, D) @ w["head"].T)
        else:  # all
            x = w["emb"](ids.reshape(-1)).reshape(B, S, 768)
            x = (x.reshape(-1, 768) @ w["up0"].T).reshape(B, S, D)
            z = x.reshape(-1, D)
            q = (z @ w["q"].T).reshape(B, S, H, DH).permute(0, 2, 1, 3)
            kv = z @ w["kv"].T
            k = (kv @ w["ku"].T).reshape(B, S, H, DH).permute(0, 2, 1, 3)
            v = (kv @ w["vu"].T).reshape(B, S, H, DH).permute(0, 2, 1, 3)
            yy = Fn.scaled_dot_product_attention(q.reshape(B * H, S, DH),
                                                 k.reshape(B * H, S, DH),
                                                 v.reshape(B * H, S, DH),
                                                 is_causal=True, lowmem=False)
            yy = yy.reshape(B, H, S, DH).permute(0, 2, 1, 3).reshape(-1, INNER)
            x = x + (yy @ w["o"].T).reshape(B, S, D)
            x = x + (Fn.silu(x.reshape(-1, D) @ w["w1"].T) @ w["w2"].T).reshape(B, S, D)
            y = x.reshape(-1, D) @ w["head"].T
        if which == "head" or which == "all":
            loss = Fn.cross_entropy(y, tgt.reshape(-1))
        else:
            loss = (y * y).sum()
        loss.backward()
        return loss

    run()                      # 预热（首次会编内核）
    rt.sync()
    rt.empty_cache()
    base = rt.pool_stats()["live_bytes"] / 2**20
    t0 = time.time()
    run()
    rt.sync()
    dt = time.time() - t0
    st = rt.pool_stats()

    peak = st["peak_live_bytes"] / 2**20
    print("%-6s B=%d  权重 %6.2f GB  峰值 %6.2f GB  激活 %6.2f GB  %.3f s"
          % (which, B, w_live / 1024, peak / 1024, (peak - w_live) / 1024, dt),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
