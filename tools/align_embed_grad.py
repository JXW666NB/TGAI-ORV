# -*- coding: utf-8 -*-
"""复测 tied-embedding 的**总梯度**（输出侧 + 输入侧）与 torch 是否一致。

背景：1 层时测得 embed 梯度差 34%，原因是 ORV 训练喂的是 numpy 常量 x0，
输入查表那条梯度整体缺失。已加 `add_embed_grad`（用 rt.emb_bwd 散射），
但**修复后从未复测**总梯度。而曲线第 1 步就跳 2.4e-04，正好可能是这里。

做法（1 层、真实文本）：
  ORV：forward_all → loss_and_head_backward（拿到输出侧 head 梯度）
       → backward_and_step(opts=None)（拿到 d(loss)/d(x0)）
       → add_embed_grad(ids[:-1])（把输入侧散射进 head.weight.grad）
       → 读 head.weight.grad
  torch：v27_ref.forward（embed 在图上）→ CE → backward → 读 embed.grad
两组梯度逐元素比；同时单独看输入侧 dw 的量级，确认它真的被加进去了。

运行（需独占 GPU）：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\align_embed_grad.py
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


def rel(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def main():
    import glob
    import json
    import os
    import tgai_go.train as T
    from tgai_go.bpe import BOS_ID, UNK_ID, BpeTokenizer
    from tgai_go.read_tg import close_tg, read_tg
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
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]), rope_theta=THETA)
    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=1)
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
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(TL, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos = T.tensor(np.cos(ang).astype(np.float32)); sin = T.tensor(np.sin(ang).astype(np.float32))

    # ---------- ORV: 输出侧 + 输入侧 ----------
    xs = tr.forward_all(x0, cos, sin)
    loss, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
    g_out = None
    if tr.head.weight.grad is not None:
        g_out = np.asarray(tr.head.weight.grad.to_numpy(), np.float32).copy()
        log("输出侧梯度已取到：|g|max=%.6g  L2=%.6g"
            % (float(np.abs(g_out).max()), float(np.linalg.norm(g_out))))
    tr.backward_and_step(xs, g, cos, sin, None)     # opts=None：只算梯度，不更新
    gx0 = getattr(tr, "last_input_grad", None)
    log("d(loss)/d(x0)：%s" % ("无" if gx0 is None else "|g|max=%.6g" % float(np.abs(gx0).max())))
    ok = tr.add_embed_grad(ids[:-1])
    log("add_embed_grad 执行=%s" % ok)
    g_tot = np.asarray(tr.head.weight.grad.to_numpy(), np.float32)
    log("总梯度（输出+输入）：|g|max=%.6g  L2=%.6g"
        % (float(np.abs(g_tot).max()), float(np.linalg.norm(g_tot))))
    if g_out is not None:
        d_in = g_tot - g_out
        log("推出的输入侧贡献：|g|max=%.6g  L2=%.6g" % (float(np.abs(d_in).max()),
                                                    float(np.linalg.norm(d_in))))

    # ---------- torch ----------
    ck = torch.load(MODEL_PT, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["model_state_dict"]

    def P_(n):
        return torch.nn.Parameter(sd[n].float().to("cuda"), requires_grad=True)

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
    ids_t = torch.from_numpy(ids[:-1]).long().cuda()
    lg = v27_ref.forward(ids_t, PT, cfg_t)
    F.cross_entropy(lg, torch.from_numpy(ids[1:]).long().cuda()).backward()
    g_ref = PT["embed"].grad.detach().cpu().numpy()
    log("torch embed 梯度：|g|max=%.6g  L2=%.6g"
        % (float(np.abs(g_ref).max()), float(np.linalg.norm(g_ref))))

    log("=== 总梯度相对误差（ORV 输出+输入 vs torch）===")
    log("  相对=%.2e  %s" % (rel(g_tot, g_ref), "一致" if rel(g_tot, g_ref) < 1e-5 else "★分叉"))
    log("=== 逐元素差异最大的 5 行（按 token 行）===")
    diff = np.abs(g_tot - g_ref).max(axis=1)
    for r in np.argsort(-diff)[:5]:
        log("  row %6d  ORV|g|=%.3e  torch|g|=%.3e  |Δ|=%.3e"
            % (r, float(np.abs(g_tot[r]).max()), float(np.abs(g_ref[r]).max()), float(diff[r])))
    log("  这些行的 token 是否出现在序列里: %s"
        % [bool((ids[:-1] == int(r)).any()) for r in np.argsort(-diff)[:5]])
    st.close()


if __name__ == "__main__":
    main()
