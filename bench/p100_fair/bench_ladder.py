# -*- coding: utf-8 -*-
"""ORV 侧"旋钮阶梯"：在同一台 P100 上把 ORV 自己的显存/速度开关逐个试出来。

为什么单独一份：公平对照（bench_fair.py）回答"ORV vs torch 谁快"；
这一份回答"**ORV 在这张卡上最好能调到什么程度**" —— 也就是你上次报告里
显存 +26~37% 的差距，到底有多少是**配置没开**造成的。

每一行是一个组合，输出 s/step、tok/s、三种显存口径，最后按峰值显存排序打印。

显存旋钮（都真实存在，`T.help` 可查）：
    lowmem        SDPA 不物化（`lowmem=True`）—— S=2048 时这是最大的一项
    fp16          `TG_FP16=1`（GEMM 走 fp16；P100 支持 fp16 算术，1:2 速率）
    ckpt          `T.checkpoint` 每层重算（省激活、多耗算力）
    bf16_act      `T.bf16_act_store(True)`（激活存 bf16，省一半；⚠ 全有或全无）
    ce_chunk      `Fn.linear_cross_entropy(chunk=N)`（V=36864 时 logits 很大）

用法
    python bench_ladder.py --bs 4 --iters 4
    python bench_ladder.py --bs 2,4 --iters 5 --json ladder.json
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

V, D, S = 36864, 1152, 2048
L, NH, DH, FF = 21, 18, 64, 2560


def smi_used_mib():
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        vals = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
        return max(vals) if vals else None
    except Exception:                             # noqa: BLE001
        return None


def run_one(bs, lowmem, fp16, ckpt, bf16_act, ce_chunk, iters):
    import tgai_go.train as T
    from tgai_go.train import functional as Fn

    if fp16:
        os.environ["TG_FP16"] = "1"
    else:
        os.environ.pop("TG_FP16", None)
    if bf16_act and hasattr(T, "bf16_act_store"):
        T.bf16_act_store(True, 262144)
    elif hasattr(T, "bf16_act_store"):
        T.bf16_act_store(False)

    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)

    def dev(n, m):
        return T.Tensor._from_gpu(rt.h2d(rng.normal(0, 0.02, (n, m)).astype(np.float32)))

    emb = T.nn.Embedding(V, D)
    head = dev(V, D)
    layers = []
    for _ in range(L):
        layers.append(dict(q=dev(D, D), kv=dev(256, D), ku=dev(D, 256),
                           vu=dev(D, 256), o=dev(D, D),
                           w1=dev(FF, D), w2=dev(D, FF)))
    params = [emb.weight, head] + [v for l in layers for v in l.values()]
    for t in params:
        t.requires_grad = True
    opt = T.optim.AdamW(params, lr=1e-4)
    ids = T.tensor(rng.integers(0, V, (bs, S)).astype(np.int64))
    tgt = T.tensor(rng.integers(0, V, (bs, S)).astype(np.int64))

    def blk(x, l):
        z = x.reshape(-1, D)
        q = (z @ l["q"].T).reshape(bs, S, NH, DH).permute(0, 2, 1, 3)
        kv = z @ l["kv"].T
        k = (kv @ l["ku"].T).reshape(bs, S, NH, DH).permute(0, 2, 1, 3)
        v = (kv @ l["vu"].T).reshape(bs, S, NH, DH).permute(0, 2, 1, 3)
        y = Fn.scaled_dot_product_attention(q.reshape(bs * NH, S, DH),
                                            k.reshape(bs * NH, S, DH),
                                            v.reshape(bs * NH, S, DH),
                                            is_causal=True, lowmem=lowmem)
        y = y.reshape(bs, NH, S, DH).permute(0, 2, 1, 3).reshape(-1, D)
        x = x + (y @ l["o"].T).reshape(bs, S, D)
        f = x.reshape(-1, D)
        return x + (Fn.silu(f @ l["w1"].T) @ l["w2"].T).reshape(bs, S, D)

    def body():
        x = emb(ids.reshape(-1)).reshape(bs, S, D)
        for l in layers:
            x = T.checkpoint(blk, x, l) if ckpt else blk(x, l)
        if ce_chunk and hasattr(Fn, "linear_cross_entropy"):
            loss = Fn.linear_cross_entropy(x.reshape(-1, D), head, tgt.reshape(-1),
                                           chunk=ce_chunk)
        else:
            loss = Fn.cross_entropy(x.reshape(-1, D) @ head.T, tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    best_smi = 0
    for _ in range(2):
        body()
    rt.sync()
    t0 = time.time()
    for _ in range(iters):
        body()
        u = smi_used_mib()
        if u and u > best_smi:
            best_smi = u
    rt.sync()
    dt = (time.time() - t0) / iters
    st = rt.pool_stats() if hasattr(rt, "pool_stats") else {}
    return {"bs": bs, "lowmem": lowmem, "fp16": fp16, "ckpt": ckpt,
            "bf16_act": bf16_act, "ce_chunk": ce_chunk,
            "s_per_step": dt, "tok_s": bs * S / dt,
            "peak_live_gib": st.get("peak_live_bytes", 0) / 2 ** 30,
            "peak_reserved_gib": st.get("peak_bytes", 0) / 2 ** 30,
            "peak_smi_gib": best_smi / 1024.0 if best_smi else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description="ORV 显存/速度旋钮阶梯（P100）")
    ap.add_argument("--bs", default="4")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--ce-chunks", default="0,2048", help="逗号分隔，0=不分块")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    rows = []
    for bs in [int(x) for x in a.bs.split(",")]:
        for ce_chunk in [int(x) for x in a.ce_chunks.split(",")]:
            combos = list(itertools.product((True, False), (True, False),
                                            (True, False), (True, False)))
            for lowmem, fp16, ckpt, bf16_act in combos:
                try:
                    r = run_one(bs, lowmem, fp16, ckpt, bf16_act, ce_chunk, a.iters)
                except Exception as e:            # noqa: BLE001
                    print("  bs=%d lowmem=%-5s fp16=%-5s ckpt=%-5s bf16act=%-5s "
                          "ce=%d FAILED: %s"
                          % (bs, lowmem, fp16, ckpt, bf16_act, ce_chunk,
                             str(e)[:90]), flush=True)
                    continue
                rows.append(r)
                print("  bs=%d lowmem=%-5s fp16=%-5s ckpt=%-5s bf16act=%-5s ce=%-5d"
                      " | %6.2f s/step %6.0f tok/s | live %5.2f res %5.2f smi %s"
                      % (bs, lowmem, fp16, ckpt, bf16_act, ce_chunk,
                         r["s_per_step"], r["tok_s"], r["peak_live_gib"],
                         r["peak_reserved_gib"],
                         ("%5.2f" % r["peak_smi_gib"]) if r["peak_smi_gib"] else "  n/a"),
                      flush=True)

    if rows:
        print("\n== 按峰值显存（框架 live）排序 ==")
        for r in sorted(rows, key=lambda x: x["peak_live_gib"])[:10]:
            print("  live %5.2f GiB | %6.2f s/step | lowmem=%-5s fp16=%-5s ckpt=%-5s"
                  " bf16act=%-5s ce=%d"
                  % (r["peak_live_gib"], r["s_per_step"], r["lowmem"], r["fp16"],
                     r["ckpt"], r["bf16_act"], r["ce_chunk"]))
        print("\n== 按步时排序 ==")
        for r in sorted(rows, key=lambda x: x["s_per_step"])[:10]:
            print("  %6.2f s/step | live %5.2f GiB | lowmem=%-5s fp16=%-5s ckpt=%-5s"
                  " bf16act=%-5s ce=%d"
                  % (r["s_per_step"], r["peak_live_gib"], r["lowmem"], r["fp16"],
                     r["ckpt"], r["bf16_act"], r["ce_chunk"]))
        if a.json:
            with open(a.json, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, ensure_ascii=False, indent=2)
            print("\n结果已写入 %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
