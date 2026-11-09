# -*- coding: utf-8 -*-
"""全 28 层对照：前向 loss 是否一致 + 单步耗时/显存。

用法（分两个进程跑，避免权重叠加 OOM）：
    python full_cmp.py --side orv
    python full_cmp.py --side torch

ORV：载入 28 层 → 前向 loss → 3 步训练（逐层反向 + AdamW fp16 动量）→ 时间/峰值显存
torch：载入 28 层 → 前向 loss（no_grad）→ 2 步 fwd+bwd → 时间/峰值显存 → 试 AdamW（预期 OOM）
"""
import argparse
import json
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
SEQ = 128
BETAS, EPS, WD, LR = (0.9, 0.999), 1e-8, 0.0, 1e-6


def peak_mib():
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["orv", "torch"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import glob
    import os
    import time
    import tgai_go.train as T
    from tgai_go.bpe import BOS_ID, UNK_ID, BpeTokenizer
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(MODEL_TG)
    h = info["header"]
    tk = BpeTokenizer(info["tokens"], info["merges"], unk_id=UNK_ID, bos_id=BOS_ID)
    close_tg(info)
    DM, NH, NL = int(h["d_model"]), int(h["n_heads"]), int(h["n_layers"])
    DH, THETA = DM // NH, float(h["rope_theta"])
    cfg_t = dict(d_model=DM, n_heads=NH, head_dim=DH, n_experts=int(h["n_experts"]),
                 n_activated=int(h["n_activated"]), rope_theta=THETA)

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
    res = {"side": args.side, "layers": NL, "seq": TL}

    if args.side == "orv":
        from tgai_go.train.opt_fp16mv import Fp16MomAdamW
        from tgai_go.train.v27_train import TGState, V27Trainer
        T.set_default_device("cuda")
        st = TGState(MODEL_TG)
        tr = V27Trainer(cfg=dict(d_model=DM, n_heads=NH, d_ff=int(h["d_ff"]),
                                 n_experts=cfg_t["n_experts"], n_activated=cfg_t["n_activated"],
                                 n_layers=NL, rope_theta=THETA), n_layers=NL)
        t0 = time.time()
        tr.load_all(st)
        tr.attach_head(st)
        res["load_s"] = round(time.time() - t0, 1)
        res["vram_after_load"] = peak_mib()
        emb = st["token_embedding.weight"]
        x0 = np.ascontiguousarray(emb[ids[:-1]], np.float32)
        j = np.arange(0, DH, 2, dtype=np.float32) / DH
        ang = np.repeat(np.arange(TL, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
        cos = T.tensor(np.cos(ang).astype(np.float32)); sin = T.tensor(np.sin(ang).astype(np.float32))

        xs = tr.forward_all(x0, cos, sin)
        ls, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
        res["loss"] = float(np.asarray(ls.to_numpy() if hasattr(ls, "to_numpy") else ls))
        kw = dict(lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
        opts = [Fp16MomAdamW(tr.params_of(i), **kw) for i in range(NL)]
        head = Fp16MomAdamW(tr.head_params(), **kw)
        res["vram_after_opt"] = peak_mib()
        times = []
        for _ in range(3):
            t0 = time.time()
            _e = np.asarray(tr.head.weight.to_numpy(), np.float32)
            xs = tr.forward_all(np.ascontiguousarray(_e[ids[:-1]]), cos, sin)
            ls, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
            tr.backward_and_step(xs, g, cos, sin, opts)
            head.step(); head.zero_grad()
            for p in tr.head_params():
                p.grad = None
            times.append(time.time() - t0)
        res["step_s"] = round(float(np.median(times)), 3)
        res["vram_peak"] = peak_mib()
        st.close()
    else:
        ck = torch.load(MODEL_PT, map_location="cpu", mmap=True, weights_only=False)
        sd = ck["model_state_dict"]

        def P_(n, rg=True):
            return torch.nn.Parameter(sd[n].float().to("cuda"), requires_grad=rg)

        blk = []
        for i in range(NL):
            p0 = "blocks.%d." % i
            b = {"ln1": P_(p0 + "ln1.weight"), "ln2": P_(p0 + "ln2.weight"),
                 "q": P_(p0 + "attn.q_proj.weight"), "k": P_(p0 + "attn.k_proj.weight"),
                 "v": P_(p0 + "attn.v_proj.weight"), "out": P_(p0 + "attn.out_proj.weight"),
                 "moe": {"router": P_(p0 + "moe.router.weight"), "experts": []}}
            if (p0 + "moe._qb_bias") in sd:
                b["moe"]["qb_bias"] = P_(p0 + "moe._qb_bias", False)
            for e in range(cfg_t["n_experts"]):
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
        res["vram_after_load"] = peak_mib()
        ids_t = torch.from_numpy(ids[:-1]).long().cuda()
        tgt = torch.from_numpy(ids[1:]).long().cuda()
        with torch.no_grad():
            res["loss"] = float(F.cross_entropy(v27_ref.forward(ids_t, PT, cfg_t), tgt))
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(2):
            t0 = time.time()
            lg = v27_ref.forward(ids_t, PT, cfg_t)
            F.cross_entropy(lg, tgt).backward()
            times.append(time.time() - t0)
            for p in ps:
                p.grad = None
        res["step_s"] = round(float(np.median(times)), 3)
        res["step_kind"] = "fwd+bwd（不含优化器）"
        res["vram_peak"] = peak_mib()
        try:
            opt = torch.optim.AdamW(ps, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
            lg = v27_ref.forward(ids_t, PT, cfg_t)
            F.cross_entropy(lg, tgt).backward()
            opt.step()
            res["adamw"] = "成功"
        except Exception as e:
            res["adamw"] = "OOM/失败: %s" % type(e).__name__
        res["vram_peak_adamw"] = peak_mib()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log("%s: %s" % (args.side, json.dumps(res, ensure_ascii=False)))


if __name__ == "__main__":
    main()
