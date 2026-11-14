# -*- coding: utf-8 -*-
"""检查点 v2：边界张量留在设备侧，逐层重算后**立即释放**并归还池缓存。

与 v1 的差别（v1 实测 +38%，已被判定为实现 bug）：

* v1 把每层边界转成 numpy 再转回，且循环里同时持有 `xi`/`y`/`bounds` 多份引用；
* v2 只保留边界张量本身，每层 backward 后 `del` 并 `empty_cache()`。

验收：1 层 seq2048 E=8 的 peak_live 应从 3,891 MiB 降到接近「只前向 1,142 + 参数梯度」。
"""
import argparse
import gc
import json
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402
from tgai_go.train import v3_ops as V, v3_moe as MOE  # noqa: E402
from v3_orv_train import build, params_of  # noqa: E402


TOP_K = [1]        # 由 --top_k 覆盖；默认 2 = 真实 V3（model_v3.py:145）
ATTN = ["flash"]   # "flash" | "materialized"，由 --attn 覆盖


def layer_fwd(x, L, cos, sin, d_k, n_q, n_kv):
    h = x + V.attention_orv(V.rmsnorm_orv(x, L["ln1"]), L["wq"], L["wk"], L["wv"],
                            L["wo"], cos, sin, d_k, n_q, n_kv,
                            flash=(ATTN[0] == "flash"),
                            lowmem=(ATTN[0] == "lowmem"))
    Tlen = int(h.shape[1]); d = int(h.shape[2])
    hn = V.rmsnorm_orv(h, L["ln2"]).reshape(Tlen, d)
    if int(TOP_K[0]) == 1:
        eids, g, _s = MOE.route_top1(hn, L["router"])
        mo = MOE.moe_dispatch_orv(hn, eids, g, L["W1"], L["W2"], L["W3"])
    else:
        eids, gates, _s = MOE.route_topk(hn, L["router"], int(TOP_K[0]))
        mo = MOE.moe_topk_orv(hn, eids, gates, L["W1"], L["W2"], L["W3"])
    if "S1" in L:                                  # 共享专家（与 PyT 的 n_shared_experts=1 对齐）
        mo = mo + V.situ_glu_orv(hn, L["S1"], L["S2"], L["S3"])
    return h + mo.reshape(1, Tlen, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--n_q", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=2,
                    help="每 token 激活专家数。真实 V3 是 2（model_v3.py:145）")
    ap.add_argument("--policy", default="standard",
                    choices=["standard", "speed", "extreme", "memory"],
                    help="执行策略：显存与速度的取舍，见 T.set_policy")
    ap.add_argument("--attn", default="policy",
                    choices=["policy", "flash", "materialized", "lowmem"],
                    help="注意力实现。lowmem = 引擎自带融合核（要求 d_k=64，即 n_q=8）")
    ap.add_argument("--checkpoint", default="policy", choices=["policy", "on", "off"],
                    help="梯度检查点。默认取策略值（policy），可显式覆盖")
    a = ap.parse_args()
    TOP_K[0] = int(a.top_k)
    T.set_default_device("cuda")
    T.set_policy(a.policy)
    # 策略即默认值，命令行只在显式给出时覆盖
    ATTN[0] = T.policy("attn", "flash") if a.attn == "policy" else a.attn
    ck = bool(T.policy("checkpoint", True)) if a.checkpoint == "policy" else (a.checkpoint == "on")
    a.checkpoint = ck
    rt = T._cuda()
    cfg = dict(d=a.d, d_ff=4 * a.d, E=a.E, layers=a.layers, n_q=a.n_q, n_kv=2,
               vocab=4096, seq=a.seq, steps=a.steps)
    rng = np.random.default_rng(0)
    P = build(cfg, rng)
    d_k = a.d // a.n_q
    cos, sin = V.rope_cos_sin(a.seq, d_k, 10000.0)
    cos, sin = T.tensor(cos), T.tensor(sin)
    opt = T.optim.AdamW(params_of(P), lr=3e-4)
    ids_all = rng.integers(0, 4096, (a.steps + 1, a.seq)).astype(np.int64)

    times, losses = [], []
    for step in range(a.steps):
        ids = ids_all[step]
        tgt = np.roll(ids, -1)
        t0 = time.time()
        opt.zero_grad()

        if not a.checkpoint:
            # 不做检查点：直接建图前向 + 一次反向（最快、最费显存）
            x = P["emb"][ids].reshape(1, a.seq, a.d)
            for L in P["layers"]:
                x = layer_fwd(x, L, cos, sin, d_k, a.n_q, cfg["n_kv"])
            logits = V.rmsnorm_orv(x, P["ln_f"]).reshape(a.seq, a.d) @ P["head"].transpose(0, 1)
            loss = T.functional.cross_entropy(logits, tgt)
            lv = loss.to_numpy() if hasattr(loss, "to_numpy") else loss
            lv = float(lv) if isinstance(lv, (int, float)) else float(np.asarray(lv).reshape(-1)[0])
            loss.backward()
            opt.step()
            times.append(time.time() - t0)
            losses.append(lv)
            del x, logits, loss
            print("step %d loss=%.4f %.2fs" % (step, lv, times[-1]), flush=True)
            continue

        with T.no_grad():
            x = P["emb"][ids].reshape(1, a.seq, a.d)
            bounds = [x]
            for L in P["layers"]:
                x = layer_fwd(x, L, cos, sin, d_k, a.n_q, cfg["n_kv"])
                bounds.append(x)

        last = T.tensor(np.asarray(bounds[-1].to_numpy(), np.float32), requires_grad=True)
        print("  前向(no_grad)完成 live=%.0f MiB" % (rt.pool_stats()["live_bytes"] / 2**20),
              flush=True)
        logits = V.rmsnorm_orv(last, P["ln_f"]).reshape(a.seq, a.d) @ P["head"].transpose(0, 1)
        loss = T.functional.cross_entropy(logits, tgt)
        lv = loss.to_numpy() if hasattr(loss, "to_numpy") else loss
        lv = float(lv) if isinstance(lv, (int, float)) else float(np.asarray(lv).reshape(-1)[0])
        loss.backward()
        g_up = np.asarray(last.grad.to_numpy(), np.float32)
        print("  头部反传完成 live=%.0f MiB" % (rt.pool_stats()["live_bytes"] / 2**20),
              flush=True)
        del last, logits, loss
        rt.empty_cache()

        for i in range(len(P["layers"]) - 1, -1, -1):
            xi = T.tensor(np.asarray(bounds[i].to_numpy(), np.float32), requires_grad=True)
            y = layer_fwd(xi, P["layers"][i], cos, sin, d_k, a.n_q, cfg["n_kv"])
            if step == 0 and i in (7, 4, 0):
                print("      layer %d 建图后 live=%.0f MiB"
                      % (i, rt.pool_stats()["live_bytes"] / 2**20), flush=True)
            y.backward(T.tensor(np.ascontiguousarray(g_up)))
            g_up = np.asarray(xi.grad.to_numpy(), np.float32)
            del xi, y
            # ★ 若图里有引用环（tensor → _backward 闭包 → tensor），`del` 不会立即释放，
            #   必须 gc.collect()。实测每层残留 ≈98 MB ≈ 6 个 (N, d_ff)，正是 MoE 的中间量。
            gc.collect()
            rt.empty_cache()
            print("    layer %d done, live=%.0f MiB"
                  % (i, rt.pool_stats()["live_bytes"] / 2**20), flush=True)

        opt.step()
        times.append(time.time() - t0); losses.append(lv)
        print("step %d loss=%.4f %.2fs" % (step, lv, times[-1]), flush=True)

    st = rt.pool_stats()
    print("RESULT ckpt_v2 layers=%d seq=%d step_s=%.2f peak_live=%.0f MiB peak=%.0f MiB"
          % (a.layers, a.seq, float(np.mean(times)),
             st["peak_live_bytes"] / 2**20, st["peak_bytes"] / 2**20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
