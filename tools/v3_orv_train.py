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
            wq=p(n_q * d_k, d), wk=p(n_kv * d_k, d), wv=p(n_kv * d_k, d), wo=p(d, n_q * d_k),
            W1=p(E, d_ff, d), W2=p(E, d_ff, d), W3=p(E, d, d_ff),
        ))
    return P


def params_of(P):
    out = [P["emb"], P["ln_f"], P["head"]]
    for L in P["layers"]:
        out.extend(L.values())
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
    a = ap.parse_args()

    T.set_default_device("cuda")
    base = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.splitlines()[0])
    cfg = dict(d=a.d, d_ff=4 * a.d, E=a.E, layers=a.layers,
               n_q=4, n_kv=2, vocab=4096, seq=a.seq, steps=a.steps)

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
            h = x + V.attention_orv(V.rmsnorm_orv(x, L["ln1"]),
                                    L["wq"], L["wk"], L["wv"], L["wo"],
                                    cos, sin, a.d // cfg["n_q"], cfg["n_q"], cfg["n_kv"])
            hn3 = V.rmsnorm_orv(h, L["ln2"])
            hn = hn3.reshape(a.seq, a.d)                  # MoE 按行分派，用二维视图
            router = T.cat([L["W1"][e] for e in range(a.E)], dim=0)   # 占位：见 route_top1 注释
            eids, g, _s = MOE.route_top1(hn, router)
            mo = MOE.moe_dispatch_orv(hn, eids, g, L["W1"], L["W2"], L["W3"])
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
