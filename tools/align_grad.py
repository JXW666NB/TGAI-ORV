# -*- coding: utf-8 -*-
"""对齐定位 ⑦：**梯度对照**（同输入、同损失，按参数组比 d(loss)/d(param)）。

前向已对齐（4 层 loss 逐位相同），但 15 步曲线从第 1 步起差 4e-4。
⇒ 剩下的误差在**反向**。本脚本用"带图前向 + loss.backward()"取解析梯度，
与 torch 的同名参数梯度逐组比较，找出哪一组先爆。

运行：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\align_grad.py
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
NL = 1


def rel(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    d = float(np.abs(a - b).max())
    return d / max(1e-9, float(np.abs(b).max()))


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

    # ---------- ORV：带图前向 + backward ----------
    x = T.tensor(x0, requires_grad=True)
    for i in range(NL):
        x = tr.fwd(x, i, cos, sin)
    logits_o = tr.head(tr.final_norm(x))
    loss_o = F.cross_entropy(torch.from_numpy(np.asarray(logits_o.to_numpy(), np.float32)).cuda(), tgt)
    # 用 ORV 的 CE 更公平：这里直接对 logits 用 ORV 的 cross_entropy 才能拿到图上的梯度
    loss_o2 = T.functional.cross_entropy(logits_o, ids[1:])
    loss_o2.backward()
    lo = float(np.asarray(loss_o2.to_numpy() if hasattr(loss_o2, "to_numpy") else loss_o2))

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
    lg_t = v27_ref.forward(ids_t, PT, cfg_t)
    loss_t = F.cross_entropy(lg_t, tgt)
    loss_t.backward()
    lt = float(loss_t.item())
    log("loss: ORV=%.8f torch=%.8f 相对=%.2e" % (lo, lt, abs(lo - lt) / max(1e-9, abs(lt))))

    L = tr.layers[0]
    # (名字, ORV 参数, torch 参数, 梯度切片器)  —— 切片必须作用在**梯度**上，
    # 对 nn.Parameter 切片得到的临时张量 `.grad` 永远是 None（v1 踩过）。
    groups = []
    groups.append(("embed(head)", tr.head.weight, PT["embed"], None))
    groups.append(("final_norm", tr.final_norm.weight, PT["final_norm"], None))
    groups.append(("ln1", L["ln1"].weight, blk["ln1"], None))
    groups.append(("ln2", L["ln2"].weight, blk["ln2"], None))
    for hh in (0, NH - 1):
        sl = (hh * DH, (hh + 1) * DH)
        groups.append(("q head%d" % hh, L["q"][hh].weight, blk["q"], ("rows",) + sl))
        groups.append(("k head%d" % hh, L["k"][hh].weight, blk["k"], ("rows",) + sl))
        groups.append(("v head%d" % hh, L["v"][hh].weight, blk["v"], ("rows",) + sl))
        groups.append(("o head%d" % hh, L["o"][hh].weight, blk["out"], ("cols",) + sl))
    groups.append(("router", L["moe"].router.weight, blk["moe"]["router"], None))
    for e in (0, cfg["n_experts"] - 1):
        groups.append(("exp%d.gate" % e, L["moe"].experts[e].gate.weight,
                       blk["moe"]["experts"][e]["gate"], None))
        groups.append(("exp%d.up" % e, L["moe"].experts[e].up.weight,
                       blk["moe"]["experts"][e]["up"], None))
        groups.append(("exp%d.down" % e, L["moe"].experts[e].down.weight,
                       blk["moe"]["experts"][e]["down"], None))

    log("=== 梯度相对误差（ORV graph-backward vs torch autograd）===")
    worst = (0.0, "")
    for name, po, pt, sl in groups:
        go = getattr(po, "grad", None)
        gt = pt.grad
        if go is None or gt is None:
            log("  %-14s 缺梯度（orv=%s torch=%s）" % (name, go is not None, gt is not None))
            continue
        a = np.asarray(go.to_numpy() if hasattr(go, "to_numpy") else go, np.float32)
        b = np.asarray(gt.detach().cpu().numpy(), np.float32)
        if sl is not None:
            if sl[0] == "rows":
                b = b[sl[1]:sl[2], :]
            else:
                b = b[:, sl[1]:sl[2]]
        r = rel(a, b)
        if r > worst[0]:
            worst = (r, name)
        log("  %-14s 相对=%.2e  %s" % (name, r, "一致" if r < 1e-5 else "★分叉"))
    log("最差：%s 相对=%.2e" % (worst[1], worst[0]))
    st.close()


if __name__ == "__main__":
    main()
