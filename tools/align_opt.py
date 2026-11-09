# -*- coding: utf-8 -*-
"""对齐定位 ⑧：**优化器 vs 逐层反向**，谁还在制造曲线漂移？

现状：前向已对齐（4 层 loss 逐位相同）、除 tied-embed 外梯度都对（≤7e-6），
补上 embed 梯度后曲线最差仍有 2.83e-03（目标 1e-4）。

本脚本做两条 ORV 训练路径，都与同一份 torch 对照：
  P1 **图反向**：整模型建图 → loss.backward() → ORV AdamW 更新（梯度公式与 torch 同源）
  P2 **逐层反向**：forward_all + backward_and_step（真实训练用的路径）

若 P1 与 torch 一致而 P2 不一致 ⇒ 剩余误差来自逐层反算（重算前向 + 分段累加）；
若两者都偏 ⇒ 优化器实现（bias correction / eps / wd 顺序）有差。

运行：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\align_opt.py
"""
import sys

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
SEQ, NL, STEPS, LR = 128, 4, 6, 1e-5
BETAS, EPS, WD = (0.9, 0.999), 1e-8, 0.0


def main():
    import glob
    import json
    import os
    import tgai_go.train as T
    from tgai_go.bpe import BOS_ID, UNK_ID, BpeTokenizer
    from tgai_go.read_tg import close_tg, read_tg
    from tgai_go.train.opt_fp16mv import Fp16MomAdamW
    from tgai_go.train.v27_train import TGState, V27Trainer

    info = read_tg(MODEL_TG)
    h = info["header"]
    tk = BpeTokenizer(info["tokens"], info["merges"], unk_id=UNK_ID, bos_id=BOS_ID)
    close_tg(info)
    DM, NH = int(h["d_model"]), int(h["n_heads"])
    DH, THETA = DM // NH, float(h["rope_theta"])
    cfg_t = dict(d_model=DM, n_heads=NH, head_dim=DH, n_experts=int(h["n_experts"]),
                 n_activated=int(h["n_activated"]), rope_theta=THETA)
    cfg = dict(d_model=DM, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=cfg_t["n_experts"],
               n_activated=cfg_t["n_activated"], n_layers=int(h["n_layers"]), rope_theta=THETA)
    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)
    tr.attach_head(st)
    emb = st["token_embedding.weight"]

    ids = None
    for p in sorted(glob.glob(os.path.join("D:" + chr(92) + "SFT" + chr(0x8bed) + chr(0x6599) + chr(0x5e93),
                                          "**", "*.jsonl"), recursive=True)):
        with open(p, encoding="utf-8", errors="replace") as f:
            for _ in range(40):
                line = f.readline()
                if not line:
                    break
                try:
                    t = (json.loads(line).get("text") or "")
                except Exception:
                    continue
                if len(t) > 600:
                    ids = np.asarray(tk.encode(t, add_bos=True)[:SEQ + 1], np.int64)
                    break
        if ids is not None:
            break
    TL = len(ids) - 1
    x0 = np.ascontiguousarray(emb[ids[:-1]], np.float32)
    tgt = torch.from_numpy(ids[1:]).long().cuda()
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(TL, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos = T.tensor(np.cos(ang).astype(np.float32)); sin = T.tensor(np.sin(ang).astype(np.float32))

    kw = dict(lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, fp32_state=True)

    def fresh_opt():
        opts = [Fp16MomAdamW(tr.params_of(i), **kw) for i in range(NL)]
        head = Fp16MomAdamW(tr.head_params(), **kw)
        return opts, head

    # ---------- P1 图反向 ----------
    opts1, head1 = fresh_opt()
    loss_p1 = []
    for step in range(STEPS):
        x = T.tensor(np.ascontiguousarray(tr._embed_np(ids[:-1])), requires_grad=True) if hasattr(tr, "_embed_np") else None
        x = T.tensor(x0, requires_grad=True)
        for i in range(NL):
            x = tr.fwd(x, i, cos, sin)
        logits = tr.head(tr.final_norm(x))
        loss = T.functional.cross_entropy(logits, ids[1:])
        loss_p1.append(float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy") else loss)))
        loss.backward()
        for o in opts1:
            o.step(); o.zero_grad()
        head1.step(); head1.zero_grad()
        for p in tr.head_params():
            p.grad = None

    # 复位权重到初始（从 .tg 重新装载）
    tr.load_all(st)
    tr.attach_head(st)

    # ---------- P2 逐层反向（真实训练路径）----------
    opts2, head2 = fresh_opt()
    loss_p2 = []
    for step in range(STEPS):
        xs = tr.forward_all(x0, cos, sin)
        loss, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
        loss_p2.append(float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy") else loss)))
        tr.backward_and_step(xs, g, cos, sin, opts2)
        tr.add_embed_grad(ids[:-1])
        head2.step(); head2.zero_grad()
        for p in tr.head_params():
            p.grad = None

    tr.load_all(st)
    tr.attach_head(st)

    # ---------- torch ----------
    ck = torch.load(MODEL_PT, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["model_state_dict"]

    def P_(n):
        return torch.nn.Parameter(sd[n].float().to("cuda"), requires_grad=True)

    blk = []
    for i in range(NL):
        p0 = "blocks.%d." % i
        b = {"ln1": P_(p0 + "ln1.weight"), "ln2": P_(p0 + "ln2.weight"),
             "q": P_(p0 + "attn.q_proj.weight"), "k": P_(p0 + "attn.k_proj.weight"),
             "v": P_(p0 + "attn.v_proj.weight"), "out": P_(p0 + "attn.out_proj.weight"),
             "moe": {"router": P_(p0 + "moe.router.weight"), "experts": []}}
        if (p0 + "moe._qb_bias") in sd:
            b["moe"]["qb_bias"] = P_(p0 + "moe._qb_bias")
        for e in range(cfg["n_experts"]):
            q = p0 + "moe.experts.%d." % e
            b["moe"]["experts"].append({"gate": P_(q + "w1.weight"), "up": P_(q + "w2.weight"),
                                       "down": P_(q + "w3.weight")})
        blk.append(b)
    PT = {"embed": P_("token_embedding.weight"), "final_norm": P_("final_norm.weight"), "blocks": blk}
    params = [PT["embed"], PT["final_norm"]]
    for b in blk:
        params += [b["ln1"], b["ln2"], b["q"], b["k"], b["v"], b["out"], b["moe"]["router"]]
        if "qb_bias" in b["moe"]:
            params.append(b["moe"]["qb_bias"])
        for e in b["moe"]["experts"]:
            params += [e["gate"], e["up"], e["down"]]
    opt = torch.optim.AdamW(params, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    ids_t = torch.from_numpy(ids[:-1]).long().cuda()
    loss_t = []
    for step in range(STEPS):
        lg = v27_ref.forward(ids_t, PT, cfg_t)
        l = F.cross_entropy(lg, tgt)
        loss_t.append(float(l.detach()))
        l.backward()
        opt.step(); opt.zero_grad(set_to_none=True)

    print("\n%5s %14s %14s %14s %12s %12s" % ("step", "ORV 图反向", "ORV 逐层", "PyTorch", "P1相对", "P2相对"))
    for i in range(STEPS):
        r1 = abs(loss_p1[i] - loss_t[i]) / max(1e-9, abs(loss_t[i]))
        r2 = abs(loss_p2[i] - loss_t[i]) / max(1e-9, abs(loss_t[i]))
        print("%5d %14.6f %14.6f %14.6f %12.2e %12.2e" % (i, loss_p1[i], loss_p2[i], loss_t[i], r1, r2))
    print("\n最差：图反向 %.2e / 逐层 %.2e" %
          (max(abs(a - b) / max(1e-9, abs(b)) for a, b in zip(loss_p1, loss_t)),
           max(abs(a - b) / max(1e-9, abs(b)) for a, b in zip(loss_p2, loss_t))))
    st.close()


if __name__ == "__main__":
    main()
