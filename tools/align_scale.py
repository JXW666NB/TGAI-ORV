# -*- coding: utf-8 -*-
"""对齐收尾：量化**单步一致度**与**曲线偏差的放大关系**。

已知（实测）：前向逐位相同、梯度 ≤7e-6、AdamW ≤5e-8、ORV 自比噪声底 3e-7，
embedding 梯度已修好（5.85e-6）。15 步曲线最差 2.88e-04。

本脚本回答："2.88e-04 是系统性错误，还是单步 1e-6 级差异在陡峭曲面上被放大？"
  ① 单步参数一致度：走 1 步后，ORV 与 torch 的参数最大相对差
  ② lr 缩放：lr=1e-6 与 lr=1e-5 各跑 5 步，比较曲线偏差是否随 lr 等比例缩小
  ③ 步数缩放：同 lr 下 3 步 vs 5 步，偏差是否随步数增长

运行（需独占 GPU）：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\align_scale.py
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
SEQ, NL = 128, 4
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
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(TL, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    tgt = torch.from_numpy(ids[1:]).long().cuda()
    cos_np, sin_np = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    ck = torch.load(MODEL_PT, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["model_state_dict"]

    def build_torch():
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
                b["moe"]["qb_bias"].requires_grad_(False)
            for e in range(cfg["n_experts"]):
                q = p0 + "moe.experts.%d." % e
                b["moe"]["experts"].append({"gate": P_(q + "w1.weight"), "up": P_(q + "w2.weight"),
                                           "down": P_(q + "w3.weight")})
            blk.append(b)
        PT = {"embed": P_("token_embedding.weight"), "final_norm": P_("final_norm.weight"), "blocks": blk}
        ps = [PT["embed"], PT["final_norm"]]
        for b in blk:
            ps += [b["ln1"], b["ln2"], b["q"], b["k"], b["v"], b["out"], b["moe"]["router"]]
            for e in b["moe"]["experts"]:
                ps += [e["gate"], e["up"], e["down"]]
        return PT, ps

    def run(lr, steps):
        """返回 (orv_losses, torch_losses, orv_params, torch_params)"""
        st = TGState(MODEL_TG)
        tr = V27Trainer(cfg, n_layers=NL)
        tr.load_all(st)
        tr.attach_head(st)
        emb = st["token_embedding.weight"]
        kw = dict(lr=lr, betas=BETAS, eps=EPS, weight_decay=WD, fp32_state=True)
        opts = [Fp16MomAdamW(tr.params_of(i), **kw) for i in range(NL)]
        head_opt = Fp16MomAdamW(tr.head_params(), **kw)
        cos = T.tensor(cos_np); sin = T.tensor(sin_np)
        lo = []
        for _ in range(steps):
            _e = np.asarray(tr.head.weight.to_numpy(), np.float32)
            x0 = np.ascontiguousarray(_e[ids[:-1]])
            xs = tr.forward_all(x0, cos, sin)
            ls, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
            lo.append(float(np.asarray(ls.to_numpy() if hasattr(ls, "to_numpy") else ls)))
            tr.backward_and_step(xs, g, cos, sin, opts)
            tr.add_embed_grad(ids[:-1])
            head_opt.step(); head_opt.zero_grad()
            for p in tr.head_params():
                p.grad = None
        orv_p = {"ln1": np.asarray(tr.layers[0]["ln1"].weight.to_numpy(), np.float32).copy(),
                 "embed": np.asarray(tr.head.weight.to_numpy(), np.float32).copy()}
        st.close()

        PT, ps = build_torch()
        opt = torch.optim.AdamW(ps, lr=lr, betas=BETAS, eps=EPS, weight_decay=WD)
        ids_t = torch.from_numpy(ids[:-1]).long().cuda()
        lt = []
        for _ in range(steps):
            lg = v27_ref.forward(ids_t, PT, cfg_t)
            l = F.cross_entropy(lg, tgt)
            lt.append(float(l.detach()))
            l.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        t_p = {"ln1": PT["blocks"][0]["ln1"].detach().cpu().numpy().copy(),
               "embed": PT["embed"].detach().cpu().numpy().copy()}
        return lo, lt, orv_p, t_p

    def rr(a, b):
        return float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).max()) / \
            max(1e-9, float(np.abs(np.asarray(b, np.float32)).max()))

    log("=== ① 单步（lr=1e-5, 1 步）：参数一致度 ===")
    lo, lt, po, pt_ = run(1e-5, 1)
    log("  loss ORV=%.6f torch=%.6f 相对=%.2e" % (lo[0], lt[0], abs(lo[0] - lt[0]) / max(1e-9, abs(lt[0]))))
    log("  参数 ln1  相对=%.2e" % rr(po["ln1"], pt_["ln1"]))
    log("  参数 embed 相对=%.2e" % rr(po["embed"], pt_["embed"]))

    log("=== ② lr 缩放（5 步）===")
    for lr in (1e-6, 1e-5):
        lo, lt, _, _ = run(lr, 5)
        worst = max(abs(a - b) / max(1e-9, abs(b)) for a, b in zip(lo, lt))
        log("  lr=%-8g 5 步最差相对=%.2e   末步 ORV=%.6f torch=%.6f" % (lr, worst, lo[-1], lt[-1]))

    log("=== ③ 15 步（两种 lr）：偏差是否随 lr 缩小 = 混沌放大而非系统性错误 ===")
    for lr in (1e-6, 1e-5):
        lo, lt, _, _ = run(lr, 15)
        devs = [abs(a - b) / max(1e-9, abs(b)) for a, b in zip(lo, lt)]
        ok = sum(1 for d in devs if d < 1e-4)
        log("  lr=%-8g 15 步：最差=%.2e 中位=%.2e  <1e-4 的步数=%d/15"
            % (lr, max(devs), float(np.median(devs)), ok))


if __name__ == "__main__":
    main()
