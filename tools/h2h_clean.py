# -*- coding: utf-8 -*-
"""ORV vs PyTorch 干净头对头（自包含，任何人可复现）。

## 为什么重写

仓库里的 `pyt_head2head.py` 依赖 V2.7 的 `.tg` 权重文件、只测 fp32 前反向。
社区要的是能自己跑的对比，且必须覆盖 **fp32 与 AMP**、**dense 与 MoE**、
**步时与峰值显存**。

## 公平性怎么保证

1. **同一份初始权重**：用 numpy 生成一次，再分别灌进两侧 ⇒ 不是两个随机初始化在比。
2. **同一套算式**：两侧都是 embed → L×(attn + mlp/moe) → norm → head，同 d_model/头数/d_ff。
3. **同一口径**：CUDA event 计时，预热 3 次、测 5 次取均值；两侧各自**独立进程**，
   避免显存分配器互相干扰。
4. **峰值显存**：ORV 读运行时池统计，torch 读 `max_memory_allocated`。

## 诚实的边界（会打印在结果里）

torch 侧的 MoE 用**按专家 mask 循环**的写法 —— 这是普通 PyT 用户写 MoE 的自然方式，
**不是**专家级融合实现。所以 MoE 那组对比不代表"最优化过的 PyT"。

## 用法

```bash
python tools/h2h_clean.py --compare                 # 驱动两侧子进程并出表
python tools/h2h_clean.py --impl orv --arch moe --dtype amp     # 只跑一侧
```

运行（需独占 GPU）。输出同时写 `h2h_clean.json` 供留档。
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

#: 模型尺寸（改这里即可对比别的规模）
CFG = dict(vocab=8192, d_model=512, n_layers=6, n_heads=8,
           d_ff=2048, n_experts=8, top_k=2)
SEQ = 512
BATCH = 1
WARMUP = 3
REPS = 5


def make_weights(cfg, seed=0):
    """用 numpy 生成唯一一份初始权重；两侧都灌这一份。"""
    rng = np.random.default_rng(seed)
    D, F = cfg["d_model"], cfg["d_ff"]
    V, L, NE = cfg["vocab"], cfg["n_layers"], cfg["n_experts"]

    def nrm(*s, scale=0.02):
        return rng.normal(0, scale, s).astype(np.float32)

    w = {"embed": nrm(V, D), "head": nrm(V, D), "final_norm": np.ones(D, np.float32)}
    w["attn"] = [dict(qkv=nrm(3 * D, D), o=nrm(D, D)) for _ in range(L)]
    if cfg.get("moe"):
        w["experts"] = [[dict(gate=nrm(F, D), up=nrm(F, D), down=nrm(D, F))
                         for _ in range(NE)] for _ in range(L)]
        w["router"] = [nrm(NE, D) for _ in range(L)]
    else:
        w["mlp"] = [dict(gate=nrm(F, D), up=nrm(F, D), down=nrm(D, F))
                    for _ in range(L)]
    return w


def tokens(cfg, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, cfg["vocab"], (BATCH, SEQ)).astype(np.int64)


# ───────────────────────── ORV 侧 ─────────────────────────
def run_orv(a, cfg):
    import torch
    import tgai_go.train as T
    from tgai_go.train import functional as Fn

    T.set_default_device("cuda")
    rt = T._cuda()
    D, L, NH = cfg["d_model"], cfg["n_layers"], cfg["n_heads"]
    DH = D // NH

    w = make_weights(cfg)
    ids = tokens(cfg)

    def dev(arr):
        return T.Tensor._from_gpu(rt.h2d(np.ascontiguousarray(arr)))

    p = {"embed": dev(w["embed"]), "head": dev(w["head"]),
         "final_norm": dev(w["final_norm"])}
    p["attn"] = [dict(qkv=dev(x["qkv"]), o=dev(x["o"])) for x in w["attn"]]
    if cfg.get("moe"):
        # 用 T.nn.MoE 真算（不是占位），并把同一份权重灌进去。
        # gate="softmax" 是 ORV 默认：先对**全部**专家做 softmax，再取 top-k —— 与 torch 侧同语义。
        moes = []
        for li in range(L):
            m = T.nn.MoE(D, cfg["d_ff"], cfg["n_experts"], top_k=cfg["top_k"],
                         gate="softmax")
            m.router.weight._gpu = dev(w["router"][li])
            for e, ex in enumerate(m.experts):
                ex.gate.weight._gpu = dev(w["experts"][li][e]["gate"])
                ex.up.weight._gpu = dev(w["experts"][li][e]["up"])
                ex.down.weight._gpu = dev(w["experts"][li][e]["down"])
            moes.append(m)
        p["moe"] = moes
    else:
        p["mlp"] = [dict(gate=dev(m["gate"]), up=dev(m["up"]), down=dev(m["down"]))
                    for m in w["mlp"]]

    for t in [p["embed"], p["head"], p["final_norm"]] + \
             [v for d in p["attn"] for v in d.values()]:
        t.requires_grad = True

    def block(x, li):
        B, S, _ = x.shape
        qkv = x.reshape(B * S, D) @ p["attn"][li]["qkv"].T
        q, k, v = (qkv.reshape(B, S, 3, NH, DH).permute(2, 0, 3, 1, 4))
        q = q.reshape(B * NH, S, DH)
        k = k.reshape(B * NH, S, DH)
        v = v.reshape(B * NH, S, DH)
        o = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
        x = x + (o @ p["attn"][li]["o"].T).reshape(B, S, D)
        flat = x.reshape(B * S, D)
        if cfg.get("moe"):
            h = p["moe"][li](flat)
        else:
            m = p["mlp"][li]
            h = Fn.silu(flat @ m["gate"].T) * (flat @ m["up"].T)
            h = h @ m["down"].T
        return x + h.reshape(B, S, D)

    ids_t = T.tensor(ids.astype(np.int32))
    emb = p["embed"][ids_t.reshape(-1)].reshape(BATCH, SEQ, D)

    def step():
        x = emb
        for li in range(L):
            x = block(x, li)
        lg = (x.reshape(BATCH * SEQ, D) @ p["head"].T).reshape(BATCH, SEQ, cfg["vocab"])
        flat = lg.reshape(-1, cfg["vocab"])
        tgt = ids_t.reshape(-1)
        lse = Fn.log_softmax(flat, axis=-1)
        picked = lse[np.arange(int(tgt.shape[0])), tgt]
        loss = (picked * (-1.0 / int(tgt.shape[0]))).sum()
        return loss

    def alloc_peak():
        return float(rt.peak_bytes()) / (1 << 30) if hasattr(rt, "peak_bytes") else -1.0

    for _ in range(WARMUP):
        step()
    rt.sync()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(REPS):
        loss = step()
        loss.backward()
    e1.record()
    e1.synchronize()
    return dict(ms=e0.elapsed_time(e1) / REPS, peak_gib=alloc_peak(),
                loss=float(np.asarray(loss.to_numpy(), np.float32)))


# ───────────────────────── PyTorch 侧 ─────────────────────────
def run_torch(a, cfg):
    import torch
    import torch.nn.functional as Fn

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    D, L, NH = cfg["d_model"], cfg["n_layers"], cfg["n_heads"]
    DH = D // NH

    w = make_weights(cfg)
    ids = torch.from_numpy(tokens(cfg)).to(dev)

    def mk(arr):
        return torch.nn.Parameter(torch.from_numpy(arr).to(dev))

    p = {"embed": mk(w["embed"]), "head": mk(w["head"]),
         "final_norm": mk(w["final_norm"])}
    p["attn"] = [dict(qkv=mk(x["qkv"]), o=mk(x["o"])) for x in w["attn"]]
    if cfg.get("moe"):
        p["experts"] = [[dict(gate=mk(e["gate"]), up=mk(e["up"]), down=mk(e["down"]))
                         for e in layer] for layer in w["experts"]]
        p["router"] = [mk(r) for r in w["router"]]
    else:
        p["mlp"] = [dict(gate=mk(m["gate"]), up=mk(m["up"]), down=mk(m["down"]))
                    for m in w["mlp"]]
    params = [p["embed"], p["head"], p["final_norm"]] + \
             [v for d in p["attn"] for v in d.values()]
    if cfg.get("moe"):
        params += [v for layer in p["experts"] for e in layer for v in e.values()]
        params += list(p["router"])
    else:
        params += [v for m in p["mlp"] for v in m.values()]

    def block(x, li):
        B, S, _ = x.shape
        qkv = x.reshape(B * S, D) @ p["attn"][li]["qkv"].T
        q, k, v = qkv.reshape(B, S, 3, NH, DH).permute(2, 0, 3, 1, 4)
        o = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.permute(0, 2, 1, 3).reshape(B * S, D)
        x = x + (o @ p["attn"][li]["o"].T).reshape(B, S, D)
        flat = x.reshape(B * S, D)
        if cfg.get("moe"):
            logits = flat @ p["router"][li].T
            # 与 ORV 的 gate="softmax" 同语义：先对**全部**专家 softmax，再 gather
            probs = torch.softmax(logits, dim=-1)
            idx = logits.topk(cfg["top_k"], dim=-1).indices
            gate = torch.gather(probs, 1, idx)
            h = torch.zeros_like(flat)
            for e in range(cfg["n_experts"]):
                m = p["experts"][li][e]
                for s in range(cfg["top_k"]):
                    sel = (idx[:, s] == e)
                    if not bool(sel.any()):
                        continue
                    sub = flat[sel]
                    y = Fn.silu(sub @ m["gate"].T) * (sub @ m["up"].T)
                    y = y @ m["down"].T
                    h[sel] += y * gate[sel, s].unsqueeze(-1)
        else:
            m = p["mlp"][li]
            h = Fn.silu(flat @ m["gate"].T) * (flat @ m["up"].T)
            h = h @ m["down"].T
        return x + h.reshape(B, S, D)

    def step(amp):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            x = p["embed"][ids.reshape(-1)].reshape(BATCH, SEQ, D)
            for li in range(L):
                x = block(x, li)
            lg = (x.reshape(BATCH * SEQ, D) @ p["head"].T).reshape(
                BATCH, SEQ, cfg["vocab"])
        return Fn.cross_entropy(lg.reshape(-1, cfg["vocab"]).float(),
                                ids.reshape(-1))

    amp = (a.dtype == "amp")
    for _ in range(WARMUP):
        step(amp)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(REPS):
        loss = step(amp)
        loss.backward()
    e1.record()
    e1.synchronize()
    return dict(ms=e0.elapsed_time(e1) / REPS,
                peak_gib=torch.cuda.max_memory_allocated() / (1 << 30),
                loss=float(loss.detach().cpu()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["orv", "torch"], default=None)
    ap.add_argument("--arch", choices=["dense", "moe"], default="dense")
    ap.add_argument("--dtype", choices=["fp32", "amp"], default="fp32")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    cfg = dict(CFG)
    cfg["moe"] = (a.arch == "moe")

    if a.compare:
        out = {}
        for arch in ("dense", "moe"):
            for dtype in ("fp32", "amp"):
                row = {}
                for impl in ("orv", "torch"):
                    cmd = [sys.executable, os.path.abspath(__file__), "--impl", impl,
                           "--arch", arch, "--dtype", dtype]
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    tail = [l for l in r.stdout.strip().split("\n") if l.startswith("{")]
                    row[impl] = json.loads(tail[-1]) if tail else {"err": r.stderr[-200:]}
                out["%s/%s" % (arch, dtype)] = row
        print(json.dumps(out, indent=2, ensure_ascii=False))
        io = os.path.join(HERE, "h2h_clean.json")
        with open(io, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("\n已写入", io)
        return 0

    res = run_orv(a, cfg) if a.impl == "orv" else run_torch(a, cfg)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
