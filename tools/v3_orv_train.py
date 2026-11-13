# -*- coding: utf-8 -*-
"""在 ORV 上跑 V3 架构的**真实训练步**（完整 AdamW、权重真更新）。

结构 = V3 的骨架（pre-norm 残差 + GlobalAttention + MoE(E 专家, top_k=1) + SiTU-GLU）：

```
x = emb[ids]
for layer: h = x + attn(rmsnorm(x));  x = h + moe(rmsnorm(h))
loss = CE(head(rmsnorm(x)), targets);  loss.backward(); opt.step()
```

用 `scatter_add` 做专家分派（每个专家只算自己的 token）。
权重随机初始化（同构对比用；V3 的 ORV 格式权重尚无导出器）。

指标：峰值显存（nvidia-smi 采样）、每步耗时、tok/s、逐 step loss。
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V  # noqa: E402
from tgai_go.train import v3_moe as MOE  # noqa: E402


def _vram_peak(stop, base, out):
    peak = base
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
            peak = max(peak, int(o.splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.2)
    out["peak"] = peak


def build(cfg, rng):
    """随机初始化参数（同构对比用）。返回值都是 requires_grad 的 ORV 张量。"""
    d, d_ff, E, L = cfg["d"], cfg["d_ff"], cfg["E"], cfg["layers"]
    n_q, n_kv = cfg["n_q"], cfg["n_kv"]
    d_k = d // n_q
    def p(*shape, scale=0.02):
        return T.tensor(rng.normal(0, scale, shape).astype(np.float32), requires_grad=True)
    P = {"emb": p(cfg["vocab"], d), "ln_f": T.tensor(np.ones(d, np.float32), requires_grad=True),
         "head": p(cfg["vocab"], d), "layers": []}
    for _ in range(L):
        P["layers"].append(dict(
            ln1=T.tensor(np.ones(d, np.float32), requires_grad=True),
            ln2=T.tensor(np.ones(d, np.float32), requires_grad=True),
            # router 是**独立参数**（参考实现 `nn.Linear(d_model, n_experts)`），
            # 不能每步用 cat([W1[e]…]) 现拼（既偏架构，又多拷 E×d_ff×d 个元素）
            router=p(E, d),
            wq=p(n_q * d_k, d), wk=p(n_kv * d_k, d), wv=p(n_kv * d_k, d), wo=p(d, n_q * d_k),
            # ★ 专家权重用**独立参数列表**，不用三维张量再 [e] 索引：
            #   实测 `W3d[e]` 每次都拷贝（4 MiB/次），MoE 每层索引 24 次（8 专家 × 3 权重），
            #   拷出来的副本还会被自动微分保留到反向 —— 一层约 200 MiB 的纯浪费。
            #   列表索引不产生任何 GPU 拷贝。
            W1=[p(d_ff, d) for _ in range(E)],
            W2=[p(d_ff, d) for _ in range(E)],
            W3=[p(d, d_ff) for _ in range(E)],
            S1=p(d_ff, d), S2=p(d_ff, d), S3=p(d, d_ff),     # 共享专家的 FFN（对齐 PyT 的 n_shared_experts=1）
        ))
    return P


def params_of(P):
    out = [P["emb"], P["ln_f"], P["head"]]
    for L in P["layers"]:
        for k, v in L.items():
            if isinstance(v, list):      # 专家权重是列表
                out.extend(v)
            else:
                out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_q", type=int, default=4,
                    help="注意力查询头数；d/n_q 即 dh。dh==64 时 sdpa_lowmem 可用")
    ap.add_argument("--lowmem", action="store_true",
                    help="走不物化 P 的注意力内核（需 dh==64 + causal + CUDA）")
    ap.add_argument("--chunk", type=int, default=0,
                    help=">0 时用分块注意力（照 PyTorch mem-efficient 的思路：不物化 S×S）")
    ap.add_argument("--flash", action="store_true",
                    help="块化在线 softmax 注意力（不物化 S×S，反向重算）")
    ap.add_argument("--no-shared", action="store_true",
                    help="关闭共享专家。PyT 侧是 n_shared_experts=1，默认开启以保持同构")
    a = ap.parse_args()
    a.shared = not a.no_shared          # 默认开启（与 PyT 对齐）

    T.set_default_device("cuda")
    base = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.splitlines()[0])
    cfg = dict(d=a.d, d_ff=4 * a.d, E=a.E, layers=a.layers,
               n_q=a.n_q, n_kv=2, vocab=4096, seq=a.seq, steps=a.steps)

    rng = np.random.default_rng(0)
    P = build(cfg, rng)
    cos, sin = V.rope_cos_sin(a.seq, a.d // cfg["n_q"], 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    opt = T.optim.AdamW(params_of(P), lr=a.lr)

    stop = threading.Event(); prof = {}
    th = threading.Thread(target=_vram_peak, args=(stop, base, prof), daemon=True); th.start()

    ids_all = rng.integers(0, cfg["vocab"], (a.steps + 1, a.seq)).astype(np.int64)
    losses, times = [], []
    for step in range(a.steps):
        ids = ids_all[step]
        tgt = np.roll(ids, -1)
        t0 = time.time()
        x = P["emb"][ids].reshape(1, a.seq, a.d)          # 嵌入（整数索引）→ (1,T,d)
        for L in P["layers"]:
            if a.chunk > 0:
                from tgai_go.train import v3_attn_chunked as CH
                attn = CH.attention_orv_chunked(V.rmsnorm_orv(x, L["ln1"]),
                                                L["wq"], L["wk"], L["wv"], L["wo"],
                                                cos, sin, a.d // cfg["n_q"],
                                                cfg["n_q"], cfg["n_kv"], chunk=a.chunk)
            else:
                attn = V.attention_orv(V.rmsnorm_orv(x, L["ln1"]),
                                       L["wq"], L["wk"], L["wv"], L["wo"],
                                       cos, sin, a.d // cfg["n_q"], cfg["n_q"], cfg["n_kv"],
                                       lowmem=a.lowmem, flash=a.flash)
            h = x + attn
            hn3 = V.rmsnorm_orv(h, L["ln2"])
            hn = hn3.reshape(a.seq, a.d)                  # MoE 按行分派，用二维视图
            # ★ 必须用真正的 router 参数（[E, d]）。原先这行用 `cat([W1[e]])` 当占位，
            #   那会造出 [E·d_ff, d] = [16384, 512] 的矩阵，`hn @ routerᵀ` 得到 **[S, 16384]**，
            #   softmax 与掩码各再一张 ⇒ 每层约 384 MiB 的纯浪费（S=2048 时）。
            #   参考实现的 router 是 `nn.Linear(d_model, n_experts)`，只有 [E, d]。
            eids, g, _s = MOE.route_top1(hn, L["router"])
            mo = MOE.moe_dispatch_orv(hn, eids, g, L["W1"], L["W2"], L["W3"])
            # ★ 共享专家：参考实现是 `n_activated=1, n_shared_experts=1`，即
            #   路由专家 top_k=1 的加权和 **再加上** 一个对所有 token 都生效的 FFN。
            #   之前 ORV 缺这一支，导致两侧参数不等（少 25.2 M），比较不是同构的。
            if a.shared:
                mo = mo + V.situ_glu_orv(hn, L["S1"], L["S2"], L["S3"])
            x = h + mo.reshape(1, a.seq, a.d)
        # head 是张量（vocab,d），不是模块 ⇒ 矩阵乘，别当函数调用
        logits = V.rmsnorm_orv(x, P["ln_f"]).reshape(a.seq, a.d) @ P["head"].transpose(0, 1)
        loss = T.functional.cross_entropy(logits, tgt)
        lv = loss.to_numpy() if hasattr(loss, "to_numpy") else loss
        lv = float(lv) if isinstance(lv, (int, float)) else float(np.asarray(lv).reshape(-1)[0])
        opt.zero_grad(); loss.backward(); opt.step()
        times.append(time.time() - t0); losses.append(lv)
        print("step %d loss=%.4f %.2fs" % (step, lv, times[-1]), flush=True)

    stop.set(); th.join(timeout=3)
    res = dict(cfg=cfg, losses=losses, step_s=float(np.mean(times)),
               tok_s=a.seq / float(np.mean(times)),
               peak_mib=prof.get("peak"), base_mib=base,
               # ★ 精确高水位：运行时池自己记的峰值，不受 nvidia-smi 采样间隔影响
               pool=T._cuda().pool_stats())
    print("RESULT", json.dumps({k: v for k, v in res.items() if k != "losses"}, ensure_ascii=False))
    print("losses", losses)
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
