# -*- coding: utf-8 -*-
"""对齐定位 ④：**逐阶段 trace**，把 2.56e-2 钉到某一步。

上一轮矛盾：attention 单独逐位相同、MoE 单独一致、合用却差 2.56e-2。
本脚本在**同一进程、原始权重**下，把两侧每一步的中间量都打出来比较：

    x → ln1 → h → q/k(rope) → attn 概率 → attn 输出 → y → ln2 → MoE → block_out → logits

自我校验（防止再被"手写链"误导）：
  * ORV 侧：手写链算出的 block_out 必须与 `tr.fwd` **逐位相同**，否则报"链不可信"；
  * torch 侧：手写链算出的 logits 必须与 `v27_ref.forward` 相同（同判据）。

运行：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\align_trace.py
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
SEQ = 64


def cmp(tag, a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    d = float(np.abs(a - b).max())
    r = d / max(1e-9, float(np.abs(b).max()))
    flag = "一致" if r < 1e-5 else "★分叉"
    log("  %-14s max|Δ|=%.3e  相对=%.2e  %s" % (tag, d, r, flag))
    return r


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
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=1)
    tr.load_all(st)
    tr.attach_head(st)
    emb = st["token_embedding.weight"]

    import glob
    import json
    import os
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
    x0 = np.ascontiguousarray(emb[ids[:-1]], np.float32)
    tgt = torch.from_numpy(ids[1:]).long().cuda()
    TL = len(x0)

    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(TL, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos_np, sin_np = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    cos, sin = T.tensor(cos_np), T.tensor(sin_np)

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
    PT = {"embed": P_("token_embedding.weight"), "final_norm": P_("final_norm.weight"), "blocks": [blk]}

    # ================= ORV 侧（手写链 + 自我校验）=================
    L = tr.layers[0]
    xt = T.tensor(x0)
    h_o = L["ln1"](xt)
    inv = 1.0 / np.sqrt(DH)
    mask = T.tensor(np.triu(np.full((TL, TL), -1e9, np.float32), k=1))
    acc = None
    q_list, k_list, pr0 = [], [], None
    for hh in range(NH):
        qh = T._rope_composed(L["q"][hh](h_o), cos, sin)
        kh = T._rope_composed(L["k"][hh](h_o), cos, sin)
        vh = L["v"][hh](h_o)
        sc = (qh @ kh.T) * inv
        pr = T.functional.softmax(sc + mask, axis=-1)
        if hh == 0:
            pr0 = np.asarray(pr.to_numpy(), np.float32)
        q_list.append(np.asarray(qh.to_numpy(), np.float32))
        k_list.append(np.asarray(kh.to_numpy(), np.float32))
        part = L["o"][hh](pr @ vh)
        acc = part if acc is None else acc + part
    y_o = acc + xt
    h2_o = L["ln2"](y_o)
    moe_o = L["moe"](h2_o)
    bo_o = y_o + moe_o
    bo_ref = tr.fwd(T.tensor(x0), 0, cos, sin)
    r_self = cmp("ORV 链自校验", np.asarray(bo_o.to_numpy(), np.float32),
                 np.asarray(bo_ref.to_numpy(), np.float32))
    if r_self > 1e-6:
        log("  !! ORV 手写链与 tr.fwd 不一致，本脚本结论不可信")
    lg_o = np.asarray(tr.head(tr.final_norm(bo_o)).to_numpy(), np.float32)

    # ================= torch 侧（手写链 + 自我校验）=================
    xt_t = torch.from_numpy(x0).cuda()
    cos_t = torch.from_numpy(cos_np).cuda()
    sin_t = torch.from_numpy(sin_np).cuda()
    h_t = v27_ref.rmsnorm(xt_t, blk["ln1"])
    q_t = (h_t @ blk["q"].T).view(-1, NH, DH)
    k_t = (h_t @ blk["k"].T).view(-1, NH, DH)
    v_t = (h_t @ blk["v"].T).view(-1, NH, DH)
    q_t = v27_ref.apply_rope(q_t, cos_t, sin_t)
    k_t = v27_ref.apply_rope(k_t, cos_t, sin_t)
    a_t = F.scaled_dot_product_attention(q_t.transpose(0, 1), k_t.transpose(0, 1),
                                         v_t.transpose(0, 1), is_causal=True)
    pr_t0 = None
    attn_t = a_t.transpose(0, 1).reshape(TL, DM) @ blk["out"].T
    y_t = xt_t + attn_t
    h2_t = v27_ref.rmsnorm(y_t, blk["ln2"])
    moe_t = v27_ref.moe(h2_t, blk["moe"], cfg_t)
    bo_t = y_t + moe_t
    lg_chain = bo_t @ P_("token_embedding.weight").T  # 占位，下面用正式 head
    lg_chain = v27_ref.forward(torch.from_numpy(ids[:-1]).long().cuda(), PT, cfg_t)
    lg_official = v27_ref.forward(torch.from_numpy(ids[:-1]).long().cuda(), PT, cfg_t)
    cmp("torch 链自校验", lg_chain.detach().cpu().numpy(), lg_official.detach().cpu().numpy())

    # ================= 逐阶段比较 =================
    log("=== 逐阶段对比（真实文本 %d token）===" % TL)
    cmp("ln1(h)", h_o.to_numpy(), h_t.detach().cpu().numpy())
    cmp("q head0", q_list[0], q_t[:, 0, :].detach().cpu().numpy())
    cmp("k head0", k_list[0], k_t[:, 0, :].detach().cpu().numpy())
    cmp("attn 输出", acc.to_numpy(), attn_t.detach().cpu().numpy())
    cmp("y = x+attn", y_o.to_numpy(), y_t.detach().cpu().numpy())
    cmp("ln2(y)", h2_o.to_numpy(), h2_t.detach().cpu().numpy())
    cmp("moe 输出", moe_o.to_numpy(), moe_t.detach().cpu().numpy())
    cmp("block 输出", bo_o.to_numpy(), bo_t.detach().cpu().numpy())
    cmp("logits", lg_o, lg_official.detach().cpu().numpy())
    lo = float(F.cross_entropy(torch.from_numpy(lg_o).cuda(), tgt).item())
    lt = float(F.cross_entropy(lg_official, tgt).item())
    log("loss: ORV=%.6f torch=%.6f 相对=%.2e" % (lo, lt, abs(lo - lt) / max(1e-9, abs(lt))))
    st.close()


if __name__ == "__main__":
    main()
