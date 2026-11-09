# -*- coding: utf-8 -*-
"""对齐定位 ⑨：**纯优化器对照**（同一份梯度、同一初始参数，各走一步 AdamW）。

前向已逐位相同、梯度除 tied-embed 外 ≤7e-6、embed 侧已补齐，但曲线仍差 ~3e-03。
最后一个未验证环节就是 **ORV 的 AdamW vs torch 的 AdamW**。

做法（完全绕开反向，只测更新规则）：
  1. ORV 与 torch 用同一份初始参数（两侧都从 bf16 源载入，已实测逐位相同）；
  2. **torch 侧算一次梯度**，把这份梯度逐组塞进 ORV 的 `p.grad`；
  3. 两侧各执行一次 AdamW(lr, betas=(0.9,0.999), eps=1e-8, wd=0)；
  4. 比更新后的参数（等价于比 Δparam）。

判读：差 ≤1e-6 ⇒ 优化器没问题，剩余漂移要回到反向路径去找；差大 ⇒ 修 AdamW。

运行（需独占 GPU）：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\align_adamw.py
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
SEQ, NL, LR = 64, 1, 1e-5
BETAS, EPS, WD = (0.9, 0.999), 1e-8, 0.0


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
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]), rope_theta=THETA)
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
    tgt = torch.from_numpy(ids[1:]).long().cuda()

    # ---------- torch：建参数 + 算梯度 ----------
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
    params = [PT["embed"], PT["final_norm"], blk["ln1"], blk["ln2"], blk["q"], blk["k"], blk["v"],
              blk["out"], blk["moe"]["router"]]
    if "qb_bias" in blk["moe"]:
        params.append(blk["moe"]["qb_bias"])
        blk["moe"]["qb_bias"].requires_grad_(False)      # 产线里 bias 是冻结的
    for e in blk["moe"]["experts"]:
        params += [e["gate"], e["up"], e["down"]]

    ids_t = torch.from_numpy(ids[:-1]).long().cuda()
    lg = v27_ref.forward(ids_t, PT, cfg_t)
    F.cross_entropy(lg, tgt).backward()

    # ---------- 记录 torch 梯度与初始参数 ----------
    gmap = {}       # name -> torch grad (numpy)
    init = {}
    def key(t): return id(t)
    for name, pt in (("embed", PT["embed"]), ("final_norm", PT["final_norm"]),
                     ("ln1", blk["ln1"]), ("ln2", blk["ln2"]),
                     ("q", blk["q"]), ("k", blk["k"]), ("v", blk["v"]), ("out", blk["out"]),
                     ("router", blk["moe"]["router"])):
        gmap[name] = pt.grad.detach().cpu().numpy().copy()
        init[name] = pt.detach().cpu().numpy().copy()
    for e, ex in enumerate(blk["moe"]["experts"]):
        for nm in ("gate", "up", "down"):
            _g = ex[nm].grad
            if _g is None:                      # 未被选中的专家没有梯度，按 0 处理
                _g = torch.zeros_like(ex[nm]).detach()
            gmap["e%d.%s" % (e, nm)] = _g.detach().cpu().numpy().copy()
            init["e%d.%s" % (e, nm)] = ex[nm].detach().cpu().numpy().copy()

    # ---------- torch 更新 ----------
    opt = torch.optim.AdamW(params, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    opt.step()
    after_t = {}
    for name, pt in (("embed", PT["embed"]), ("final_norm", PT["final_norm"]),
                     ("ln1", blk["ln1"]), ("ln2", blk["ln2"]), ("q", blk["q"]),
                     ("k", blk["k"]), ("v", blk["v"]), ("out", blk["out"]),
                     ("router", blk["moe"]["router"])):
        after_t[name] = pt.detach().cpu().numpy().copy()
    for e, ex in enumerate(blk["moe"]["experts"]):
        for nm in ("gate", "up", "down"):
            after_t["e%d.%s" % (e, nm)] = ex[nm].detach().cpu().numpy().copy()

    # ---------- ORV：同一份梯度 + 同参数 ----------
    L = tr.layers[0]
    pairs = [("embed", tr.head.weight, None), ("final_norm", tr.final_norm.weight, None),
             ("ln1", L["ln1"].weight, None), ("ln2", L["ln2"].weight, None),
             ("router", L["moe"].router.weight, None)]
    for hh in range(NH):
        pairs += [("q", L["q"][hh].weight, ("rows", hh * DH, (hh + 1) * DH)),
                  ("k", L["k"][hh].weight, ("rows", hh * DH, (hh + 1) * DH)),
                  ("v", L["v"][hh].weight, ("rows", hh * DH, (hh + 1) * DH)),
                  ("o%d" % hh, L["o"][hh].weight, ("cols", hh * DH, (hh + 1) * DH))]
    # embed 梯度= 输出侧 + 输入侧；这里只对照"输出侧"（与 torch 的 embed grad 不同源，见末尾说明）
    for hh, slot in ((0, "q"), (0, "k"), (0, "v"), (0, "o0")):
        pass

    def set_grad(po, arr):
        po.grad = T.tensor(np.ascontiguousarray(arr, np.float32))

    # 逐组塞梯度（q/k/v 取 rows，out 取 cols，embed 用 torch 的整表梯度）
    for name, po, sl in pairs:
        src = "out" if name.startswith("o") else name
        g = gmap[src]
        if sl is None:
            set_grad(po, g)
        elif sl[0] == "rows":
            set_grad(po, g[sl[1]:sl[2], :])
        else:
            set_grad(po, g[:, sl[1]:sl[2]])
    for e, ex in enumerate(L["moe"].experts):
        for nm in ("gate", "up", "down"):
            set_grad(getattr(ex, nm).weight, gmap["e%d.%s" % (e, nm)])

    opt_o = Fp16MomAdamW(tr.params_of(0), lr=LR, betas=BETAS, eps=EPS,
                         weight_decay=WD, fp32_state=True)
    head_o = Fp16MomAdamW(tr.head_params(), lr=LR, betas=BETAS, eps=EPS,
                          weight_decay=WD, fp32_state=True)
    opt_o.step()
    head_o.step()

    log("=== 一次 AdamW 后的参数相对误差（ORV vs torch）===")
    worst = (0.0, "")
    checks = [("ln1", L["ln1"].weight, "ln1"), ("ln2", L["ln2"].weight, "ln2"),
              ("router", L["moe"].router.weight, "router"),
              ("final_norm", tr.final_norm.weight, "final_norm"),
              ("q head0", L["q"][0].weight, "q", ("rows", 0, DH)),
              ("out head0", L["o"][0].weight, "out", ("cols", 0, DH))]
    for e in (0, cfg["n_experts"] - 1):
        for nm in ("gate", "up", "down"):
            checks.append(("e%d.%s" % (e, nm), getattr(L["moe"].experts[e], nm).weight,
                           "e%d.%s" % (e, nm)))
    for item in checks:
        name, po, src = item[0], item[1], item[2]
        sl = item[3] if len(item) > 3 else None
        a = np.asarray(po.to_numpy(), np.float32)
        b = after_t[src].copy()
        if sl is not None:
            if sl[0] == "rows":
                b = b[sl[1]:sl[2], :]
            else:
                b = b[:, sl[1]:sl[2]]
        r = rel(a, b)
        if r > worst[0]:
            worst = (r, name)
        log("  %-12s 相对=%.2e  %s" % (name, r, "一致" if r < 1e-6 else "★分叉"))
    log("最差：%s 相对=%.2e" % (worst[1], worst[0]))
    st.close()


if __name__ == "__main__":
    main()
