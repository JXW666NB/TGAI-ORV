# -*- coding: utf-8 -*-
"""对齐定位 ②：**单块内分支隔离**（RMSNorm / attention / MoE）。

方法（两侧都只用各自库的正式路径，不手抄前向）：
  A. RMSNorm：ORV 的 `L["ln1"](x)` vs torch 的 `v27_ref.rmsnorm(x, ln1)`
  B. attention 分支：把**所有专家权重置零**（两侧同时）⇒ MoE 输出恒 0，
     只剩 x + attention，比较 1 层 logits/loss
  C. MoE 分支：把 **q/k/v 权重置零**（两侧同时）⇒ attention 输出恒 0，
     只剩 x + MoE，比较 1 层 logits/loss

判读：哪一项先分叉，问题就在那个分支。全部一致 ⇒ 只剩浮点累加顺序（要量化噪声底）。

运行：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\align_block.py
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
SEQ = 64


def get_ids(tk, V):
    import glob
    import json
    import os
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
                    return np.asarray(tk.encode(t, add_bos=True)[:SEQ + 1], np.int64)
    return np.random.default_rng(2024).integers(0, V, SEQ + 1, dtype=np.int64)


def main():
    import tgai_go.train as T
    from tgai_go.bpe import BOS_ID, UNK_ID, BpeTokenizer
    from tgai_go.read_tg import close_tg, read_tg
    from tgai_go.train.v27_train import TGState, V27Trainer

    info = read_tg(MODEL_TG)
    h = info["header"]
    tk = BpeTokenizer(info["tokens"], info["merges"], unk_id=UNK_ID, bos_id=BOS_ID)
    close_tg(info)
    DM, NH = int(h["d_model"]), int(h["n_heads"])
    DH, THETA, V = DM // NH, float(h["rope_theta"]), int(h["vocab"])
    cfg_t = dict(d_model=DM, n_heads=NH, head_dim=DH, n_experts=int(h["n_experts"]),
                 n_activated=int(h["n_activated"]), rope_theta=THETA)
    cfg = dict(d_model=DM, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=cfg_t["n_experts"],
               n_activated=cfg_t["n_activated"], n_layers=int(h["n_layers"]), rope_theta=THETA)

    T.set_default_device("cuda")
    rt = T._cuda()
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=1)
    tr.load_all(st)
    tr.attach_head(st)
    emb = st["token_embedding.weight"]
    ids = get_ids(tk, V)
    log("输入 %d 个真实 token" % len(ids))

    ck = torch.load(MODEL_PT, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["model_state_dict"]

    def P_(n):
        return torch.nn.Parameter(sd[n].float().to("cuda"), requires_grad=False)

    p0 = "blocks.0."
    blk = {"ln1": P_(p0 + "ln1.weight"), "ln2": P_(p0 + "ln2.weight"),
           "q": P_(p0 + "attn.q_proj.weight"), "k": P_(p0 + "attn.k_proj.weight"),
           "v": P_(p0 + "attn.v_proj.weight"), "out": P_(p0 + "attn.out_proj.weight"),
           "moe": {"router": P_(p0 + "moe.router.weight"), "experts": []}}
    if (p0 + "moe._qb_bias") in sd:
        blk["moe"]["qb_bias"] = P_(p0 + "moe._qb_bias")
    for e in range(cfg["n_experts"]):
        q = p0 + "moe.experts.%d." % e
        blk["moe"]["experts"].append({"gate": P_(q + "w1.weight"), "up": P_(q + "w2.weight"),
                                     "down": P_(q + "w3.weight")})
    PT = {"embed": P_("token_embedding.weight"), "final_norm": P_("final_norm.weight"),
          "blocks": [blk]}

    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(len(ids) - 1, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos_np, sin_np = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    cos, sin = T.tensor(cos_np), T.tensor(sin_np)
    x0 = np.ascontiguousarray(emb[ids[:-1]], np.float32)
    ids_t = torch.from_numpy(ids[:-1]).long().cuda()
    tgt = torch.from_numpy(ids[1:]).long().cuda()

    def orv_loss():
        x = T.tensor(x0)
        for i in range(1):
            x = tr.fwd(x, i, cos, sin)
        lg = np.asarray(tr.head(tr.final_norm(x)).to_numpy(), np.float32)
        return float(F.cross_entropy(torch.from_numpy(lg).cuda(), tgt).item()), lg

    def torch_loss():
        lg = v27_ref.forward(ids_t, PT, cfg_t)
        return float(F.cross_entropy(lg, tgt).item()), lg.detach().cpu().numpy()

    # ---------- A. RMSNorm ----------
    xt = T.tensor(x0)
    a_o = np.asarray(tr.layers[0]["ln1"](xt).to_numpy(), np.float32)
    xt_t = torch.from_numpy(x0).cuda()
    a_t = v27_ref.rmsnorm(xt_t, blk["ln1"]).detach().cpu().numpy()
    d = float(np.abs(a_o - a_t).max()); r = d / max(1e-9, float(np.abs(a_t).max()))
    log("A RMSNorm: max|Δ|=%.3e 相对=%.2e  %s" % (d, r, "一致" if r < 1e-5 else "★分叉"))

    # ---------- 基线 ----------
    lo, lgo = orv_loss()
    lt, lgt = torch_loss()
    log("基线: ORV loss=%.6f torch=%.6f 相对=%.2e  logits max|Δ|=%.3e"
        % (lo, lt, abs(lo - lt) / max(1e-9, abs(lt)), float(np.abs(lgo - lgt).max())))

    # ---------- B. 专家置零（只留 attention）----------
    saved_o, saved_t = [], []
    for e in range(cfg["n_experts"]):
        for slot in ("gate", "up", "down"):
            po = getattr(tr.layers[0]["moe"].experts[e], slot).weight
            arr = np.array(po.to_numpy(), np.float32)
            saved_o.append((po, arr, po._gpu))
            po._gpu = rt.h2d(np.zeros_like(arr))
            pt = blk["moe"]["experts"][e][slot]
            saved_t.append((pt, pt.detach().clone()))
            with torch.no_grad():
                pt.zero_()
    lo, _ = orv_loss(); lt, _ = torch_loss()
    log("B 只留 attention（专家置零）: ORV=%.6f torch=%.6f 相对=%.2e  %s"
        % (lo, lt, abs(lo - lt) / max(1e-9, abs(lt)), "一致" if abs(lo - lt) / max(1e-9, abs(lt)) < 1e-5 else "★分叉"))
    for po, arr, orig in saved_o:
        po._gpu = orig
    for pt, orig in saved_t:
        with torch.no_grad():
            pt.copy_(orig)

    # ---------- C. q/k/v 置零（只留 MoE）----------
    saved_o, saved_t = [], []
    for name in ("q", "k", "v"):
        mods = tr.layers[0][name]
        mods = mods if isinstance(mods, list) else [mods]
        for m in mods:                      # ORV 的 q/k/v 是**按头拆开**的，必须全零
            po = m.weight
            arr = np.array(po.to_numpy(), np.float32)
            saved_o.append((po, arr, po._gpu))
            po._gpu = rt.h2d(np.zeros_like(arr))
        pt = blk[name]
        saved_t.append((pt, pt.detach().clone()))
        with torch.no_grad():
            pt.zero_()
    lo, _ = orv_loss(); lt, _ = torch_loss()
    log("C 只留 MoE（q/k/v 置零，仅第 0 头置零亦可看出趋势）: ORV=%.6f torch=%.6f 相对=%.2e  %s"
        % (lo, lt, abs(lo - lt) / max(1e-9, abs(lt)), "一致" if abs(lo - lt) / max(1e-9, abs(lt)) < 1e-5 else "★分叉"))
    for po, arr, orig in saved_o:
        po._gpu = orig
    for pt, orig in saved_t:
        with torch.no_grad():
            pt.copy_(orig)

    st.close()


if __name__ == "__main__":
    main()
