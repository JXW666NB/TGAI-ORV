# -*- coding: utf-8 -*-
"""层级梯度检查点版本：只保留层边界，反向**逐层重算**前向。

## 为什么

实测：分块注意力无效（掩码与分数仍留在图里，峰值同量级）。
真正吃掉显存的是"**所有中间激活都被保留到反向**"。
PyTorch 那份 1.5B 基线靠 `grad_ckpt=1` 才进 29 GB —— 这里做同一件事：

```
前向（no_grad）：只把每层输出留成"边界张量"（leaf，不接图）
反向：从最后一层往前，用边界值重算该层前向 → y.backward(上游梯度)
      ⇒ 拿到该层参数梯度 + 上一层的边界梯度
```

代价是**多一次前向**（约 +33% 时间），换来激活显存从 O(层数) 降到 O(1)。
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V  # noqa: E402
from tgai_go.train import v3_moe as MOE  # noqa: E402
from v3_orv_train import build, params_of  # noqa: E402


def vram_peak(stop, base, prof):
    peak = base
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
            peak = max(peak, int(o.splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.05)
    prof["peak"] = peak


def layer_fwd(x, L, cos, sin, d_k, n_q, n_kv, lowmem=False):
    """一层前向：h = x + attn(rmsnorm(x)); y = h + moe(rmsnorm(h))。"""
    h = x + V.attention_orv(V.rmsnorm_orv(x, L["ln1"]),
                            L["wq"], L["wk"], L["wv"], L["wo"],
                            cos, sin, d_k, n_q, n_kv, lowmem=lowmem)
    hn3 = V.rmsnorm_orv(h, L["ln2"])
    hn = hn3.reshape(int(h.shape[1]), int(h.shape[2]))
    eids, g, _s = MOE.route_top1(hn, L["router"])
    mo = MOE.moe_dispatch_orv(hn, eids, g, L["W1"], L["W2"], L["W3"])
    return h + mo.reshape(1, int(h.shape[1]), int(h.shape[2]))


def leaf(arr):
    return T.tensor(np.ascontiguousarray(arr, np.float32), requires_grad=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--n_q", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lowmem", action="store_true",
                    help="注意力走不物化 S² 的内核（需 dh==64）")
    a = ap.parse_args()

    T.set_default_device("cuda")
    base = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.splitlines()[0])
    cfg = dict(d=a.d, d_ff=4 * a.d, E=a.E, layers=a.layers,
               n_q=a.n_q, n_kv=2, vocab=4096, seq=a.seq, steps=a.steps)
    rng = np.random.default_rng(0)
    P = build(cfg, rng)
    d_k = a.d // a.n_q
    cos, sin = V.rope_cos_sin(a.seq, d_k, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    opt = T.optim.AdamW(params_of(P), lr=a.lr)

    stop = threading.Event(); prof = {}
    th = threading.Thread(target=vram_peak, args=(stop, base, prof), daemon=True); th.start()

    ids_all = rng.integers(0, cfg["vocab"], (a.steps + 1, a.seq)).astype(np.int64)
    losses, times = [], []
    for step in range(a.steps):
        ids = ids_all[step]
        tgt = np.roll(ids, -1)
        t0 = time.time()
        opt.zero_grad()

        # ---- 前向：只留边界（no_grad）----
        with T.no_grad():
            x = P["emb"][ids].reshape(1, a.seq, a.d)
            bounds = [np.asarray(x.to_numpy(), np.float32)]
            for L in P["layers"]:
                x = layer_fwd(x, L, cos, sin, d_k, a.n_q, cfg["n_kv"], lowmem=a.lowmem)
                bounds.append(np.asarray(x.to_numpy(), np.float32))

        # ---- 头 + 损失（接图）----
        last = leaf(bounds[-1])
        logits = V.rmsnorm_orv(last, P["ln_f"]).reshape(a.seq, a.d) @ P["head"].transpose(0, 1)
        loss = T.functional.cross_entropy(logits, tgt)
        lv = loss.to_numpy() if hasattr(loss, "to_numpy") else loss
        lv = float(lv) if isinstance(lv, (int, float)) else float(np.asarray(lv).reshape(-1)[0])
        loss.backward()
        g_up = np.asarray(last.grad.to_numpy(), np.float32)

        # ---- 反向：逐层重算 ----
        for i in range(len(P["layers"]) - 1, -1, -1):
            xi = leaf(bounds[i])
            y = layer_fwd(xi, P["layers"][i], cos, sin, d_k, a.n_q, cfg["n_kv"],
                          lowmem=a.lowmem)
            y.backward(T.tensor(np.ascontiguousarray(g_up)))
            g_up = np.asarray(xi.grad.to_numpy(), np.float32)
            # ★ 每层反向完把池缓存归还驱动：否则不同尺寸的临时块会被池"缓存"住，
            #   `peak_bytes`（占用=活跃+缓存）会虚高，看起来像检查点更费显存
            T._cuda().empty_cache()

        opt.step()
        times.append(time.time() - t0); losses.append(lv)
        print("step %d loss=%.4f %.2fs" % (step, lv, times[-1]), flush=True)

    stop.set(); th.join(timeout=3)
    res = dict(cfg=cfg, losses=losses, ckpt=True,
               step_s=float(np.mean(times)), tok_s=a.seq / float(np.mean(times)),
               peak_mib=prof.get("peak"), base_mib=base,
               pool=T._cuda().pool_stats())
    print("RESULT", json.dumps({k: v for k, v in res.items() if k not in ("losses", "pool")},
                               ensure_ascii=False))
    pool = res["pool"]
    print("pool peak_live_bytes=%d  peak_bytes=%d  (MiB: %.0f / %.0f)"
          % (pool["peak_live_bytes"], pool["peak_bytes"],
             pool["peak_live_bytes"] / 2**20, pool["peak_bytes"] / 2**20))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
