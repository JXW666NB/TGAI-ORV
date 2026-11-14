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
import os
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


def layer_fwd_moe_ckpt(x, L, cos, sin, d_k, n_q, n_kv):
    """**只对 MoE** 做梯度检查点，注意力保持 eager。

    取舍依据（实测）：注意力物化时每层留 134 MiB，MoE 每层只留约 200 MiB 但**重算很便宜**
    （分派换设备 gather 后，MoE 反向已从 0.375 s 降下来）。
    所以"重算 MoE、不重算注意力"能同时压住显存和速度。
    """
    h = x + V.attention_orv(V.rmsnorm_orv(x, L["ln1"]), L["wq"], L["wk"], L["wv"],
                            L["wo"], cos, sin, d_k, n_q, n_kv,
                            flash=(ATTN[0] == "flash"),
                            lowmem=(ATTN[0] == "lowmem"))
    Tlen = int(h.shape[1]); d = int(h.shape[2])

    def moe_block(hh):
        hn = V.rmsnorm_orv(hh, L["ln2"]).reshape(Tlen, d)
        if int(TOP_K[0]) == 1:
            eids, g, _s = MOE.route_top1(hn, L["router"])
            mo = MOE.moe_dispatch_orv(hn, eids, g, L["W1"], L["W2"], L["W3"])
        else:
            eids, gates, _s = MOE.route_topk(hn, L["router"], int(TOP_K[0]))
            mo = MOE.moe_topk_orv(hn, eids, gates, L["W1"], L["W2"], L["W3"])
        if "S1" in L:
            mo = mo + V.situ_glu_orv(hn, L["S1"], L["S2"], L["S3"])
        return mo

    mo = T.checkpoint(moe_block, h)
    return h + mo.reshape(1, Tlen, d)


def layer_fwd_attn_ckpt(x, L, cos, sin, d_k, n_q, n_kv):
    """**只对注意力**做梯度检查点，MoE 与共享专家保持 eager（不重算）。

    依据实测：整步 74% 花在反向，而检查点会把**整层**前向重算一遍；
    MoE 的激活很便宜（每层约 30 MiB），注意力的最贵（物化时 134 MiB/层）。
    所以只重算注意力，省下 MoE 那部分重算时间。
    """
    def attn_block(h_in):
        return V.attention_orv(V.rmsnorm_orv(h_in, L["ln1"]), L["wq"], L["wk"],
                               L["wv"], L["wo"], cos, sin, d_k, n_q, n_kv,
                               flash=(ATTN[0] == "flash"),
                               lowmem=(ATTN[0] == "lowmem"))

    h = x + T.checkpoint(attn_block, x)
    Tlen = int(h.shape[1]); d = int(h.shape[2])
    hn = V.rmsnorm_orv(h, L["ln2"]).reshape(Tlen, d)
    if int(TOP_K[0]) == 1:
        eids, g, _s = MOE.route_top1(hn, L["router"])
        mo = MOE.moe_dispatch_orv(hn, eids, g, L["W1"], L["W2"], L["W3"])
    else:
        eids, gates, _s = MOE.route_topk(hn, L["router"], int(TOP_K[0]))
        mo = MOE.moe_topk_orv(hn, eids, gates, L["W1"], L["W2"], L["W3"])
    if "S1" in L:
        mo = mo + V.situ_glu_orv(hn, L["S1"], L["S2"], L["S3"])
    return h + mo.reshape(1, Tlen, d)


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
    ap.add_argument("--ckpt-mode", default="layer", choices=["layer", "attn", "moe"],
                    help="检查点粒度：layer=整层重算（省最多），attn=只重算注意力，moe=只重算 MoE")
    ap.add_argument("--ckpt-head", action="store_true",
                    help="对 head+CE 做检查点：省约 66 MiB 常驻 logits，代价是重算一次 GEMM")
    ap.add_argument("--tile", type=int, default=0,
                    help="块化注意力的块大小（0 = 用策略值）。块对数 ∝ (S/tile)²")
    ap.add_argument("--opt", default="adamw", choices=["adamw", "fp16mv", "bf16mv"],
                    help="优化器：adamw（fp32 动量）/ fp16mv / bf16mv（后者不会下溢，推荐）")
    a = ap.parse_args()
    TOP_K[0] = int(a.top_k)
    T.set_default_device("cuda")
    T.set_policy(a.policy)
    if int(a.tile) > 0:
        T.set_policy_value("attn_tile", int(a.tile))
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
    if a.opt in ("fp16mv", "bf16mv"):
        # fp16/bf16 动量：Adam 状态从 2 份 fp32 变 2 份 2 字节，显存减半。
        # ★ 必须用 bf16：fp16 存二阶动量会下溢（实测 6 步 loss 8.42→552.98 发散）。
        from tgai_go.train.opt_fp16mv import Fp16MomAdamW
        opt = Fp16MomAdamW(params_of(P), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-2,
                           state="bf16" if a.opt == "bf16mv" else "fp16")
    else:
        opt = T.optim.AdamW(params_of(P), lr=3e-4)
    ids_all = rng.integers(0, 4096, (a.steps + 1, a.seq)).astype(np.int64)

    times, losses = [], []
    for step in range(a.steps):
        ids = ids_all[step]
        tgt = np.roll(ids, -1)
        t0 = time.time()
        opt.zero_grad()

        if (not a.checkpoint) or a.ckpt_mode in ("attn", "moe"):
            # 不做整层检查点：
            #   --checkpoint off  → 全部 eager（最快、最费显存）
            #   --ckpt-mode attn  → 只对注意力做检查点，MoE 保持 eager（省重算）
            _lf = {"attn": layer_fwd_attn_ckpt,
                   "moe": layer_fwd_moe_ckpt}.get(a.ckpt_mode, layer_fwd)
            x = P["emb"][ids].reshape(1, a.seq, a.d)
            for L in P["layers"]:
                x = _lf(x, L, cos, sin, d_k, a.n_q, cfg["n_kv"])
            if a.ckpt_head:
                # 头部检查点：logits 是 (seq, vocab) = 2048×4096×4 ≈ 33 MiB，
                # 加上 CE 反向需要的那份，常驻约 66 MiB。用重算换掉它，
                # 代价是反向时重跑一次 head 的 GEMM（8.6 GFLOP，约 2 ms）。
                def _head(hh):
                    return (V.rmsnorm_orv(hh, P["ln_f"]).reshape(a.seq, a.d)
                            @ P["head"].transpose(0, 1))
                logits = T.checkpoint(_head, x)
            else:
                logits = (V.rmsnorm_orv(x, P["ln_f"]).reshape(a.seq, a.d)
                          @ P["head"].transpose(0, 1))
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

        # ★ 边界保持**设备**张量：`_from_gpu` 包装同一块显存成新的叶子（无拷贝、无 D2H）。
        #   原先每次 `to_numpy()` + `T.tensor()` 会带一次强制同步 + 两次 PCIe 往返，
        #   8 层 × 2 遍 = 每步十几次同步，是检查点路径的最大固定开销之一。
        last = T.Tensor._from_gpu(bounds[-1]._gpu, requires_grad=True)
        print("  前向(no_grad)完成 live=%.0f MiB" % (rt.pool_stats()["live_bytes"] / 2**20),
              flush=True)
        logits = V.rmsnorm_orv(last, P["ln_f"]).reshape(a.seq, a.d) @ P["head"].transpose(0, 1)
        loss = T.functional.cross_entropy(logits, tgt)
        lv = loss.to_numpy() if hasattr(loss, "to_numpy") else loss
        lv = float(lv) if isinstance(lv, (int, float)) else float(np.asarray(lv).reshape(-1)[0])
        loss.backward()
        g_up = last.grad                     # 设备张量，直接往下一层传，不做 PCIe 往返
        print("  头部反传完成 live=%.0f MiB" % (rt.pool_stats()["live_bytes"] / 2**20),
              flush=True)
        del last, logits, loss
        rt.empty_cache()

        for i in range(len(P["layers"]) - 1, -1, -1):
            xi = T.Tensor._from_gpu(bounds[i]._gpu, requires_grad=True)
            y = layer_fwd(xi, P["layers"][i], cos, sin, d_k, a.n_q, cfg["n_kv"])
            if step == 0 and i in (7, 4, 0):
                print("      layer %d 建图后 live=%.0f MiB"
                      % (i, rt.pool_stats()["live_bytes"] / 2**20), flush=True)
            y.backward(g_up)
            g_up = xi.grad                    # 同样是设备张量
            del xi, y
            # ★ 若图里有引用环（tensor → _backward 闭包 → tensor），`del` 不会立即释放，
            #   需要 gc.collect()。但**每层都 gc 很贵**（Python 大堆一次约 20–40 ms，
            #   8 层就 0.2–0.3 s/步）。改成每 4 层一次：残留最多累积 4 层，显存可控。
            if i % int(os.environ.get("TGAI_GC_EVERY", "4")) == 0:
                gc.collect()
            if i % int(os.environ.get("TGAI_EMPTY_EVERY", "1")) == 0:
                rt.empty_cache()
            if os.environ.get("TGAI_CKPT_VERBOSE"):
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
