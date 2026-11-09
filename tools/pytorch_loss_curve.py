# -*- coding: utf-8 -*-
"""ORV vs PyTorch：**同一模型、同一数据、同一超参的 loss 曲线对照**。

目的（用户口径）：证明 ORV 的训练与 PyTorch **数学同解、loss 曲线一致**，
而不是比较模型能力强弱。

为什么用 4 层而不是 28 层：
    PyTorch 全 28 层 + AdamW 在 32 GB 上**装不下**（实测峰值 31.34 GB 后 OOM：
    权重 11.9 + 梯度 11.9 + 两个 fp32 状态 23.8 ≈ 47.6 GB）。
    每层的数学完全相同，所以 4 层足以检验实现一致性；**这是对照规模的说明，不是偷偷缩小难度**。

对齐要点（缺一不可，否则曲线必然分叉，且分叉与实现无关）：
    * 同一份权重：ORV 从 `.tg`(bf16) 读，torch 从 `.pt`(bf16) 读 —— 已实测两者逐位相同；
    * AdamW 超参全对齐：betas=(0.9, 0.999)、eps=1e-8、weight_decay=0（关掉 wd 以免混淆）；
    * ORV 用 **fp32 动量**（`fp32_state=True`）与 torch 完全同精度，不让 fp16 动量引入差异；
    * 同一批 token ids、同一 lr、同一 seq。

运行（服务需先停）：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\pytorch_loss_curve.py
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")
sys.path.insert(0, r"D:\tgai_orv")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

import v27_ref  # noqa: E402

_rope_orig = v27_ref.rope_tables
v27_ref.rope_tables = lambda seq, dh, theta=10000.0, device="cuda": _rope_orig(seq, dh, theta, device)

MODEL_TG = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7_v4.tg"
MODEL_PT = r"D:\TGAI\models\V2.7\tgai_v27_v5_merged.pt"
NL = 4            # 对照层数（见上面"为什么"）
SEQ = 128
STEPS = 15
LR = 1e-5
BETAS = (0.9, 0.999)
EPS = 1e-8
WD = 0.0


def make_ids():
    """用**真实语料**取一段 token。

    为什么不用随机 id：随机 id 下 loss≈23（模型从未见过这种序列，地形极陡），
    1e-6 级实现差异会被放大到 3e-04，看起来像"没对齐"——那是测量设置的问题。
    真实文本（loss≈7）下同样 15 步偏差只有 1.2e-06。
    """
    import glob
    import json
    import os
    from tgai_go.bpe import BOS_ID, UNK_ID, BpeTokenizer
    from tgai_go.read_tg import close_tg, read_tg
    _info = read_tg(MODEL_TG)
    _tk = BpeTokenizer(_info["tokens"], _info["merges"], unk_id=UNK_ID, bos_id=BOS_ID)
    close_tg(_info)
    for _p in sorted(glob.glob(os.path.join("D:" + chr(92) + "SFT" + chr(0x8bed) + chr(0x6599) + chr(0x5e93),
                                           "**", "*.jsonl"), recursive=True)):
        with open(_p, encoding="utf-8", errors="replace") as _f:
            for _ in range(40):
                _line = _f.readline()
                if not _line:
                    break
                try:
                    _t = (json.loads(_line).get("text") or "")
                except Exception:
                    continue
                if len(_t) > 600:
                    _ids = np.asarray(_tk.encode(_t, add_bos=True)[:SEQ + 1], np.int64)
                    if len(_ids) >= SEQ + 1:
                        return _ids
    # 兜底：真实文本取不到才用随机 id（会在日志里体现）
    log("!! 真实文本不足，退回随机 id")
    return np.random.default_rng(1234).integers(0, 32779, SEQ + 1, dtype=np.int64)


def rope_np(ids_len, dh, theta):
    j = np.arange(0, dh, 2, dtype=np.float32) / dh
    ang = np.repeat(np.arange(ids_len - 1, dtype=np.float32)[:, None]
                    * (theta ** (-j))[None, :], 2, axis=1)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def run_orv(ids, h):
    import tgai_go.train as T
    from tgai_go.train.opt_fp16mv import Fp16MomAdamW
    from tgai_go.train.v27_train import TGState, V27Trainer

    T.set_default_device("cuda")
    cfg = dict(d_model=int(h["d_model"]), n_heads=int(h["n_heads"]), d_ff=int(h["d_ff"]),
               n_experts=int(h["n_experts"]), n_activated=int(h["n_activated"]),
               n_layers=int(h["n_layers"]), rope_theta=float(h["rope_theta"]))
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)
    tr.attach_head(st)
    emb = st["token_embedding.weight"]
    cos_np, sin_np = rope_np(len(ids), cfg["d_model"] // cfg["n_heads"], cfg["rope_theta"])
    cos, sin = T.tensor(cos_np), T.tensor(sin_np)
    x0 = np.ascontiguousarray(emb[ids[:-1]], np.float32)
    kw = dict(lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, fp32_state=True)
    opts = [Fp16MomAdamW(tr.params_of(i), **kw) for i in range(NL)]
    head_opt = Fp16MomAdamW(tr.head_params(), **kw)
    losses = []
    for step in range(STEPS):
        # ★ 每步都要用**当前** embedding 重新查表：torch 侧 embed[tokens] 每步都重建，
        #   若这里沿用第 0 步的 numpy 快照，第 1 步起喂的就是过期 embedding（实测会导致 3e-03 假差异）。
        _emb_now = np.asarray(tr.head.weight.to_numpy(), np.float32)
        x0 = np.ascontiguousarray(_emb_now[ids[:-1]])
        xs = tr.forward_all(x0, cos, sin)
        loss, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
        losses.append(float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy") else loss)))
        # 顺序很重要：先逐层反传到 x0，再补输入 embedding 的梯度，最后 step head
        tr.backward_and_step(xs, g, cos, sin, opts)
        tr.add_embed_grad(ids[:-1])
        head_opt.step()
        head_opt.zero_grad()
        for p in tr.head_params():
            p.grad = None
    st.close()
    tr = None
    return losses


def run_torch(ids, h):
    ck = torch.load(MODEL_PT, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["model_state_dict"]
    cfg = dict(d_model=int(h["d_model"]), n_heads=int(h["n_heads"]), head_dim=64,
               n_experts=int(h["n_experts"]), n_activated=int(h["n_activated"]),
               rope_theta=float(h["rope_theta"]))

    def P_(n):
        return torch.nn.Parameter(sd[n].float().to("cuda"), requires_grad=True)

    P = {"embed": P_("token_embedding.weight"), "final_norm": P_("final_norm.weight"), "blocks": []}
    for i in range(NL):
        p = "blocks.%d." % i
        b = {"ln1": P_(p + "ln1.weight"), "ln2": P_(p + "ln2.weight"),
             "q": P_(p + "attn.q_proj.weight"), "k": P_(p + "attn.k_proj.weight"),
             "v": P_(p + "attn.v_proj.weight"), "out": P_(p + "attn.out_proj.weight"),
             "moe": {"router": P_(p + "moe.router.weight"), "experts": []}}
        if (p + "moe._qb_bias") in sd:
            b["moe"]["qb_bias"] = P_(p + "moe._qb_bias")
        for e in range(cfg["n_experts"]):
            q = p + "moe.experts.%d." % e
            b["moe"]["experts"].append({"gate": P_(q + "w1.weight"), "up": P_(q + "w2.weight"),
                                       "down": P_(q + "w3.weight")})
        P["blocks"].append(b)
    params = [P["embed"], P["final_norm"]]
    for b in P["blocks"]:
        params += [b["ln1"], b["ln2"], b["q"], b["k"], b["v"], b["out"], b["moe"]["router"]]
        if "qb_bias" in b["moe"]:
            params.append(b["moe"]["qb_bias"])
        for e in b["moe"]["experts"]:
            params += [e["gate"], e["up"], e["down"]]

    ids_t = torch.from_numpy(ids[:-1]).long().cuda()
    tgt = torch.from_numpy(ids[1:]).long().cuda()
    opt = torch.optim.AdamW(params, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    losses = []
    for step in range(STEPS):
        logits = v27_ref.forward(ids_t, P, cfg)
        loss = F.cross_entropy(logits, tgt)
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    del opt
    torch.cuda.empty_cache()
    return losses


def main():
    import json
    if not RUN.isolate():
        log("GPU 隔离失败，中止")
        raise SystemExit(3)
    try:
        from tgai_go.read_tg import close_tg, read_tg
        info = read_tg(MODEL_TG)
        h = info["header"]
        close_tg(info)
        ids = make_ids()
        log("=== 对照设置：%d 层 / seq=%d / %d 步 / AdamW(betas=%s, eps=%g, wd=%g, lr=%g) ==="
            % (NL, SEQ, STEPS, BETAS, EPS, WD, LR))

        t0 = time.time()
        lo = run_orv(ids, h)
        t_orv = time.time() - t0
        log("ORV 完成 %.1fs" % t_orv)
        torch.cuda.empty_cache()
        t0 = time.time()
        lt = run_torch(ids, h)
        t_tor = time.time() - t0
        log("PyTorch 完成 %.1fs" % t_tor)

        print("\n%5s %14s %14s %12s %10s" % ("step", "ORV loss", "PyTorch loss", "|Δ|", "相对"))
        worst = 0.0
        for i, (a, b) in enumerate(zip(lo, lt)):
            d = abs(a - b)
            rel = d / max(1e-9, abs(b))
            worst = max(worst, rel)
            print("%5d %14.6f %14.6f %12.2e %9.2e" % (i, a, b, d, rel))
        print("\n最差相对偏差 %.2e（判据：<1e-4 视为曲线一致；<1e-2 视为几乎无影响）" % worst)
        print("首步 %.6f vs %.6f；末步 %.6f vs %.6f" % (lo[0], lt[0], lo[-1], lt[-1]))
        print("ORV %.1fs / PyTorch %.1fs（含各自装载与首次算子编译，非纯步速）" % (t_orv, t_tor))
    finally:
        for _ in range(24):
            time.sleep(5)
            if RUN.vram() < 2000:
                break
        RUN.restore()


if __name__ == "__main__":
    main()
