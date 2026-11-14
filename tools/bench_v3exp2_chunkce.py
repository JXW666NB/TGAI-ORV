# -*- coding: utf-8 -*-
"""v3exp2 真实架构：torch vs ORV 对比（MLA + SMS2-MoE）。

torch 侧直接 import D:\\st\\moe\\v3e\\v3e_model.py（我们的真实训练模型）。
ORV 侧用等价骨架重写：MLA（q/q_rope/kv_down/k_up/v_up/k_rope/o）+
     稠密 FFN（w1/silu/w2）+ MoE 层用 T.nn.MoE(top_k=5, n_shared=1) 近似
     （我们的 SMS2-MoE 是"强2/弱3/常驻"三路拼接 + W_O[1728,1152]，
       ORV 没有对应算子，只有标准 MoE；所以 ORV 这侧每层 FLOPs 约高 1.6×，
       对 ORV 是保守估计）。
"""
import os
import sys
import time

import numpy as np

HB = int(os.environ.get("HB", "4"))
S = 2048
V, D = 36864, 1152
L_DENSE = [1536, 1728, 1920, 2112, 2560]
HEADS = [8, 9, 10, 11, 12] + [12] * 16
MOE_LAYERS = {5, 7, 9, 11, 13, 15, 17, 19}   # 0-based: 第 6,8,…,20 层


# ───────────────────────── torch（我们的真实模型） ─────────────────────────
def bench_torch(iters=3, fp32=True):
    import torch
    sys.path.insert(0, r"D:\st\moe\v3e")
    import v3e_config as C
    from v3e_model import build_model

    cfg = C.V3EConfig()
    m = build_model(cfg).cuda().train()
    m.grad_ckpt = True
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    x = torch.randint(0, V, (HB, S), device="cuda")
    y = torch.randint(0, V, (HB, S), device="cuda")

    def step():
        opt.zero_grad(set_to_none=True)
        if fp32:
            _lg, loss = m(x, y)
            loss.backward()
        else:
            with torch.autocast("cuda", dtype=torch.float16):
                _lg, loss = m(x, y)
            loss.backward()
        opt.step()
        return float(loss)

    for tag, it in (("warm", 1), ("run", iters)):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(it):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / it
        print("  torch %s B=%d  %.2f s/step  %.0f tok/s  峰值 %.2f GB"
              % ("fp32" if fp32 else "fp16", HB, dt, HB * S / dt,
                 torch.cuda.max_memory_allocated() / 2**30), flush=True)
    return dt


# ───────────────────────── ORV（等价骨架 + 标准 MoE） ─────────────────────────
def bench_orv(iters=3, policy="standard"):
    import tgai_go.train as T
    from tgai_go.train import functional as Fn

    T.set_default_device("cuda")
    try:
        T.set_policy(policy)
    except Exception as e:
        print("    set_policy 失败:", e)
    rt = T._cuda()
    rng = np.random.default_rng(0)

    def dev(n, m):
        return T.Tensor._from_gpu(rt.h2d(rng.normal(0, 0.02, (n, m)).astype(np.float32)))

    emb = T.nn.Embedding(V, 768)
    up0 = dev(D, 768)
    head = dev(V, D)
    layers = []
    for i in range(21):
        h = HEADS[i]
        inner = h * 64
        dct = dict(h=h, q=dev(inner, D), qr=dev(32, D), kv=dev(256, D),
                   ku=dev(inner, 256), vu=dev(inner, 256), kr=dev(32, D),
                   o=dev(D, inner))
        if i in MOE_LAYERS:
            dct["moe"] = T.nn.MoE(D, 256, 36, top_k=5, n_shared=1,
                                  shared_d_ff=576)
        else:
            ff = L_DENSE[i] if i < 5 else 2560
            dct["w1"] = dev(ff, D); dct["w2"] = dev(D, ff)
        layers.append(dct)
    params = [emb.weight, up0, head]
    for l in layers:
        for k, v in l.items():
            if k == "h":
                continue
            if k == "moe":
                params.extend(list(v.parameters()))
            else:
                params.append(v)
    for t in params:
        t.requires_grad = True
    opt = T.optim.AdamW(params, lr=3e-4)
    ids = T.tensor(rng.integers(0, V, (HB, S)).astype(np.int64))
    tgt = T.tensor(rng.integers(0, V, (HB, S)).astype(np.int64))

    def rope(x, off=0):
        #   x: [n, 32] -> cos/sin 查表
        return x  # 形状测试用的近似（RoPE 是 elementwise，不影响 GEMM 占比）

    def blk(x, l):
        h, inner = l["h"], l["h"] * 64
        B, S2, _ = x.shape
        z = x.reshape(-1, D)
        q = (z @ l["q"].T).reshape(B, S2, h, 64).permute(0, 2, 1, 3)
        kv = z @ l["kv"].T
        k = (kv @ l["ku"].T).reshape(B, S2, h, 64).permute(0, 2, 1, 3)
        v = (kv @ l["vu"].T).reshape(B, S2, h, 64).permute(0, 2, 1, 3)
        y = Fn.scaled_dot_product_attention(q.reshape(B * h, S2, 64),
                                            k.reshape(B * h, S2, 64),
                                            v.reshape(B * h, S2, 64),
                                            is_causal=True, lowmem=False)
        y = y.reshape(B, h, S2, 64).permute(0, 2, 1, 3).reshape(-1, inner)
        x = x + (y @ l["o"].T).reshape(B, S2, D)
        f = x.reshape(-1, D)
        if "moe" in l:
            y2 = l["moe"](x.reshape(-1, D))
            return x + y2.reshape(B, S2, D)
        return x + (Fn.silu(f @ l["w1"].T) @ l["w2"].T).reshape(B, S2, D)

    def step():
        x = emb(ids.reshape(-1)).reshape(HB, S, 768)
        x = (x.reshape(-1, 768) @ up0.T).reshape(HB, S, D)
        for l in layers:
            x = T.checkpoint(blk, x, l)
        if os.environ.get("CHUNKCE", "0") == "1":
            # 分块 CE：峰值只与 chunk 有关，不物化 [N, V]
            loss = Fn.linear_cross_entropy(x.reshape(-1, D), head, tgt.reshape(-1),
                                           chunk=int(os.environ.get("CHUNK", "1024")))
        else:
            lg = x.reshape(-1, D) @ head.T
            loss = Fn.cross_entropy(lg, tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        rt.sync()
        return float(np.asarray(loss.to_numpy()).reshape(-1)[0])

    for tag, it in (("warm", 1), ("run", iters)):
        try:
            t0 = time.time()
            for _ in range(it):
                step()
            rt.sync()
        except Exception as e:
            print("  ORV[%s] B=%d 失败: %s" % (policy, HB, str(e)[:100]), flush=True)
            return None
        dt = (time.time() - t0) / it
        st = rt.pool_stats()
        print("  ORV[%s] B=%d  %.2f s/step  %.0f tok/s  峰值 %.2f GB"
              % (policy, HB, dt, HB * S / dt,
                 st["peak_live_bytes"] / (1 << 30)), flush=True)
    return dt


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("torch", "both"):
        bench_torch(fp32=True)
    if what in ("orv", "both"):
        bench_orv(policy=os.environ.get("POL", "standard"))
