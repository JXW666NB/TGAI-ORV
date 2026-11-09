# -*- coding: utf-8 -*-
"""对齐定位 ①：**depth bisect**（1 / 2 / 4 层），两侧都走库函数。

为什么要这样：上一版"手抄 torch 前向"导致 46% 的假分歧。这里改成：
  * ORV 侧：`tr.fwd(x, i, cos, sin)` 逐层串联（这就是 `forward_all` 的做法）→ `tr.final_norm` → `tr.head`
  * torch 侧：`v27_ref.forward(ids, P, cfg)`，把 `P["blocks"]` 截断到前 D 层
两侧都只用各自库里的正式实现，比较 **logits** 与 **loss**。

判读：
  * D=1 就不一致 ⇒ 单层语义（RoPE / attention / MoE）有差异，继续按分支隔离；
  * D=1 一致、D 增大才发散 ⇒ 多层串联/累加或数值累积问题。

运行（产线已停，GPU 独占）：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\align_depth.py
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
DEPTHS = (1, 2, 4)


def main():
    import tgai_go.train as T
    from tgai_go.read_tg import close_tg, read_tg
    from tgai_go.train.v27_train import TGState, V27Trainer

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    D_MODEL = int(h["d_model"])
    NH = int(h["n_heads"])
    DH = D_MODEL // NH
    THETA = float(h["rope_theta"])
    V = int(h["vocab"])
    cfg_t = dict(d_model=D_MODEL, n_heads=NH, head_dim=DH,
                 n_experts=int(h["n_experts"]), n_activated=int(h["n_activated"]),
                 rope_theta=THETA)

    T.set_default_device("cuda")
    cfg = dict(d_model=D_MODEL, n_heads=NH, d_ff=int(h["d_ff"]),
               n_experts=cfg_t["n_experts"], n_activated=cfg_t["n_activated"],
               n_layers=int(h["n_layers"]), rope_theta=THETA)
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=max(DEPTHS))
    t0 = time.time()
    tr.load_all(st)
    tr.attach_head(st)
    log("ORV 装载 %d 层 %.1fs" % (max(DEPTHS), time.time() - t0))
    emb_np = st["token_embedding.weight"]

    # ---- torch 参数（只取前 max(DEPTHS) 层）----
    ck = torch.load(MODEL_PT, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["model_state_dict"]

    def P_(n):
        return torch.nn.Parameter(sd[n].float().to("cuda"), requires_grad=False)

    PT = {"embed": P_("token_embedding.weight"), "final_norm": P_("final_norm.weight"), "blocks": []}
    for i in range(max(DEPTHS)):
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
        PT["blocks"].append(b)
    log("torch 参数就绪")

    # 用**真实文本**：随机 id 会把 loss 推到极端区（12~23），微小数值差被放大，无法判断对齐
    from tgai_go.bpe import BOS_ID, UNK_ID, BpeTokenizer
    info2 = read_tg(MODEL_TG)
    tk = BpeTokenizer(info2["tokens"], info2["merges"], unk_id=UNK_ID, bos_id=BOS_ID)
    close_tg(info2)
    import glob as _g
    import json as _j
    import os as _o
    ids = None
    for _p in sorted(_g.glob(_o.path.join("D:" + chr(92) + "SFT" + chr(0x8bed) + chr(0x6599) + chr(0x5e93),
                                       "**", "*.jsonl"), recursive=True)):
        with open(_p, encoding="utf-8", errors="replace") as _f:
            for _ in range(40):
                _line = _f.readline()
                if not _line:
                    break
                try:
                    _t = (_j.loads(_line).get("text") or "")
                except Exception:
                    continue
                if len(_t) > 600:
                    ids = np.asarray(tk.encode(_t, add_bos=True)[:SEQ + 1], np.int64)
                    break
        if ids is not None:
            break
    if ids is None or len(ids) < SEQ + 1:
        ids = np.random.default_rng(2024).integers(0, V, SEQ + 1, dtype=np.int64)
        log("!! 真实文本不足，用随机 id 兜底")
    log("输入：%d 个真实 token" % len(ids))
    tgt = torch.from_numpy(ids[1:]).long().cuda()

    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos, sin = T.tensor(np.cos(ang).astype(np.float32)), T.tensor(np.sin(ang).astype(np.float32))
    x0 = np.ascontiguousarray(emb_np[ids[:-1]], np.float32)

    ids_t = torch.from_numpy(ids[:-1]).long().cuda()
    print("\n%5s %14s %14s %12s %10s %12s" % ("depth", "ORV loss", "torch loss", "|Δloss|", "相对", "max|Δlogit|"))
    for D in DEPTHS:
        # ORV：逐层串联（与 forward_all 同一做法）
        x = T.tensor(x0)
        for i in range(D):
            x = tr.fwd(x, i, cos, sin)
        lg_o = tr.head(tr.final_norm(x))
        lg_o_np = np.asarray(lg_o.to_numpy(), np.float32)
        loss_o = float(F.cross_entropy(torch.from_numpy(lg_o_np).cuda(), tgt).item())
        # torch：库函数 + 截断层数
        P_D = dict(PT)
        P_D["blocks"] = PT["blocks"][:D]
        lg_t = v27_ref.forward(ids_t, P_D, cfg_t)
        lg_t_np = lg_t.detach().cpu().numpy()
        loss_t = float(F.cross_entropy(lg_t, tgt).item())
        dlogit = float(np.abs(lg_o_np - lg_t_np).max())
        dloss = abs(loss_o - loss_t)
        print("%5d %14.6f %14.6f %12.2e %9.2e %12.2e"
              % (D, loss_o, loss_t, dloss, dloss / max(1e-9, abs(loss_t)), dlogit))
        del lg_o, lg_t
        torch.cuda.empty_cache()

    # 额外：单层输入输出对比（把 ORV 的 x0 喂给 torch 的 0 层 vs ORV 的 0 层）
    log("=== 单层输入/输出幅值（自查用）===")
    xo = tr.fwd(T.tensor(x0), 0, cos, sin)
    yo = np.asarray(xo.to_numpy(), np.float32)
    log("  ORV 层0 输出 |max|=%.6g  |mean|=%.6g" % (float(np.abs(yo).max()), float(np.abs(yo).mean())))
    log("  ORV 输入    |max|=%.6g" % float(np.abs(x0).max()))
    del xo
    torch.cuda.empty_cache()
    st.close()


if __name__ == "__main__":
    main()
