# -*- coding: utf-8 -*-
"""对齐定位 ③：**MoE 同一输入对照 + 路由翻转敏感性**。

上一轮实测（很关键）：
    A RMSNorm 一致(7e-08)；B 只留 attention 逐位相同(0.00e+00)；
    C 只留 MoE 一致(2.4e-07)；**基线却差 2.56e-02**。
两条分支各自都对 ⇒ 差异出在"组合"处，最可能是 **MoE top-2 路由对 1e-7 级输入差异的放大**。

本脚本量化这件事：
  M1 同一份输入 Y 喂两侧 MoE：若一致 ⇒ MoE 实现无问题；
  M2 把 Y 加 1e-7 噪声后，统计 ORV 侧 top-2 **翻转的 token 数**；
  M3 翻转对 loss 的影响量级（同一 Y 与加噪 Y 的 loss 差）。

运行：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\align_moe.py
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
    tgt = torch.from_numpy(ids[1:]).long().cuda()

    ck = torch.load(MODEL_PT, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["model_state_dict"]

    def P_(n):
        return torch.nn.Parameter(sd[n].float().to("cuda"), requires_grad=False)

    p0 = "blocks.0."
    moe_t = {"router": P_(p0 + "moe.router.weight"), "experts": []}
    if (p0 + "moe._qb_bias") in sd:
        moe_t["qb_bias"] = P_(p0 + "moe._qb_bias")
    for e in range(cfg["n_experts"]):
        q = p0 + "moe.experts.%d." % e
        moe_t["experts"].append({"gate": P_(q + "w1.weight"), "up": P_(q + "w2.weight"),
                                 "down": P_(q + "w3.weight")})

    # 取一个"真实量级"的 MoE 输入：ln2(x0)（两侧 RMSNorm 已验证一致）
    ln2_o = np.asarray(tr.layers[0]["ln2"](T.tensor(x0)).to_numpy(), np.float32)
    ln2_t = v27_ref.rmsnorm(torch.from_numpy(x0).cuda(), P_(p0 + "ln2.weight")).detach().cpu().numpy()
    log("ln2(x0) 两侧 max|Δ|=%.3e" % float(np.abs(ln2_o - ln2_t).max()))

    W = np.asarray(tr.layers[0]["moe"].router.weight.to_numpy(), np.float32)
    bq = None
    try:
        bq = np.asarray(tr.layers[0]["moe"].qb_bias.to_numpy(), np.float32)
    except Exception:
        pass

    def tops(Y, k=2):
        z = Y @ W.T + (bq if bq is not None else 0.0)
        idx = np.argsort(-z, axis=-1)[:, :k]
        srt = np.sort(z, axis=-1)
        margin = srt[:, -k] - srt[:, -k - 1] if z.shape[-1] > k else np.zeros(len(z))
        return idx, margin, z

    def moe_orv(Y):
        return np.asarray(tr.layers[0]["moe"](T.tensor(Y)).to_numpy(), np.float32)

    def moe_torch(Y):
        return v27_ref.moe(torch.from_numpy(Y).cuda(), moe_t, cfg_t).detach().cpu().numpy()

    # ---- M1：同一输入，两侧 MoE ----
    Y = np.ascontiguousarray(ln2_o)
    a = moe_orv(Y)
    b = moe_torch(Y)
    d = float(np.abs(a - b).max())
    log("M1 同一输入 MoE 对照: max|Δ|=%.3e  相对=%.2e  %s"
        % (d, d / max(1e-9, float(np.abs(b).max())), "一致" if d / max(1e-9, float(np.abs(b).max())) < 1e-5 else "★分叉"))

    # ---- M2：输入加 1e-7 噪声 → 路由翻转计数 ----
    rng = np.random.default_rng(0)
    for scale in (1e-7, 1e-6, 1e-5):
        Yp = Y + (rng.standard_normal(Y.shape).astype(np.float32) * scale)
        i0, m0, _ = tops(Y)
        i1, m1, _ = tops(Yp)
        flip = int(np.sum([not (set(x) == set(y)) for x, y in zip(i0, i1)]))
        log("M2 噪声 %.0e: top-2 翻转 %d/%d 个 token（最小边际 %.3e）"
            % (scale, flip, len(i0), float(np.min(m0))))

    # ---- M3：翻转对输出的影响 ----
    Yp = Y + (rng.standard_normal(Y.shape).astype(np.float32) * 1e-7)
    a2 = moe_orv(Yp)
    log("M3 1e-7 输入扰动 → MoE 输出相对变化=%.2e"
        % (float(np.abs(a2 - a).max()) / max(1e-9, float(np.abs(a).max()))))
    st.close()


if __name__ == "__main__":
    main()
