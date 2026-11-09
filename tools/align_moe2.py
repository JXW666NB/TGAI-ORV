# -*- coding: utf-8 -*-
"""对齐定位 ⑤：MoE 分叉的定性 —— **同一输入对照 + 真实链路的路由翻转统计**。

上一轮 trace 已把误差钉在 MoE 这一步：
    ln2(y) 两侧只差 9.5e-07（正常 fp32 噪声），但 MoE 输出差 **55%**。

本脚本用**真实链路的输入** ln2(y)（不是上轮错用的 ln2(x0)）做三件事：
  N1 同一输入喂两侧 MoE：若一致 ⇒ MoE 实现无问题，差异来自路由翻转；
  N2 真实链路输入下，统计 ORV 与 torch 各自路由 top-2 的**翻转 token 数**与**边际分布**；
  N3 若 N1 一致而 N2 有翻转 ⇒ 结论：**路由对 1e-7 级输入差异敏感**，需按噪声底口径评估。

运行：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\align_moe2.py
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
    TL = len(x0)

    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(TL, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos = T.tensor(np.cos(ang).astype(np.float32)); sin = T.tensor(np.sin(ang).astype(np.float32))

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

    L = tr.layers[0]
    xt = T.tensor(x0)
    h_o = L["ln1"](xt)
    inv = 1.0 / np.sqrt(DH)
    mask = T.tensor(np.triu(np.full((TL, TL), -1e9, np.float32), k=1))
    acc = None
    for hh in range(NH):
        qh = T._rope_composed(L["q"][hh](h_o), cos, sin)
        kh = T._rope_composed(L["k"][hh](h_o), cos, sin)
        vh = L["v"][hh](h_o)
        pr = T.functional.softmax((qh @ kh.T) * inv + mask, axis=-1)
        part = L["o"][hh](pr @ vh)
        acc = part if acc is None else acc + part
    y_o = acc + xt
    h2_o = np.asarray(L["ln2"](y_o).to_numpy(), np.float32)

    # torch 侧同样的 h2
    xt_t = torch.from_numpy(x0).cuda()
    h_t = v27_ref.rmsnorm(xt_t, blk["ln1"])
    qt = (h_t @ blk["q"].T).view(-1, NH, DH); kt = (h_t @ blk["k"].T).view(-1, NH, DH)
    vt = (h_t @ blk["v"].T).view(-1, NH, DH)
    cos_t = torch.from_numpy(np.cos(ang).astype(np.float32)).cuda()
    sin_t = torch.from_numpy(np.sin(ang).astype(np.float32)).cuda()
    qt = v27_ref.apply_rope(qt, cos_t, sin_t); kt = v27_ref.apply_rope(kt, cos_t, sin_t)
    at = F.scaled_dot_product_attention(qt.transpose(0, 1), kt.transpose(0, 1), vt.transpose(0, 1),
                                        is_causal=True).transpose(0, 1).reshape(TL, DM)
    y_t = xt_t + at @ blk["out"].T
    h2_t = np.asarray(v27_ref.rmsnorm(y_t, blk["ln2"]).detach().cpu().numpy(), np.float32)
    log("h2 输入差异 max|Δ|=%.3e" % float(np.abs(h2_o - h2_t).max()))

    moe_o = lambda Y: np.asarray(L["moe"](T.tensor(Y)).to_numpy(), np.float32)
    moe_t = lambda Y: v27_ref.moe(torch.from_numpy(Y).cuda(), blk["moe"], cfg_t).detach().cpu().numpy()

    # N1 同一输入
    a = moe_o(h2_o); b = moe_t(np.ascontiguousarray(h2_o))
    d = float(np.abs(a - b).max())
    log("N1 同一输入(h2_orv) MoE: max|Δ|=%.3e 相对=%.2e %s"
        % (d, d / max(1e-9, float(np.abs(b).max())),
           "一致" if d / max(1e-9, float(np.abs(b).max())) < 1e-5 else "★分叉"))
    # N1b 用 torch 的输入
    a2 = moe_o(h2_t); b2 = moe_t(h2_t)
    d2 = float(np.abs(a2 - b2).max())
    log("N1b 同一输入(h2_torch) MoE: max|Δ|=%.3e 相对=%.2e %s"
        % (d2, d2 / max(1e-9, float(np.abs(b2).max())),
           "一致" if d2 / max(1e-9, float(np.abs(b2).max())) < 1e-5 else "★分叉"))

    # N2 真实链路（各用各的输入）路由统计
    W = np.asarray(L["moe"].router.weight.to_numpy(), np.float32)
    bq = None
    try:
        bq = np.asarray(L["moe"].qb_bias.to_numpy(), np.float32)
    except Exception:
        pass

    def tops(Y, k=2):
        z = Y @ W.T + (bq if bq is not None else 0.0)
        idx = np.argsort(-z, axis=-1)[:, :k]
        srt = np.sort(z, axis=-1)
        return idx, (srt[:, -k] - srt[:, -k - 1])

    i_o, m_o = tops(h2_o)
    i_t, m_t = tops(h2_t)
    flip = int(np.sum([not (set(x) == set(y)) for x, y in zip(i_o, i_t)]))
    order = int(np.sum(np.all(i_o == i_t, axis=-1)))
    log("N2 路由: 集合一致=%d/%d 有序一致=%d/%d  最小边际=%.3e  中位边际=%.3e"
        % (len(i_o) - flip, len(i_o), order, len(i_o), float(np.min(m_o)), float(np.median(m_o))))
    # 差异 vs 边际：翻转 token 的边际应该很小
    small = int(np.sum(m_o < 1e-4))
    log("     边际 <1e-4 的 token 数=%d/%d（这些是易翻的位置）" % (small, len(m_o)))
    # N3 各用各的输入时的 MoE 输出差
    a3 = moe_o(h2_o); b3 = moe_t(h2_t)
    d3 = float(np.abs(a3 - b3).max())
    log("N3 真实链路 MoE 输出: max|Δ|=%.3e 相对=%.2e" % (d3, d3 / max(1e-9, float(np.abs(b3).max()))))
    st.close()


if __name__ == "__main__":
    main()
