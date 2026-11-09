# -*- coding: utf-8 -*-
"""ORV vs PyTorch 对齐差异**定位**（对应 ①②③）。

背景：4 层 loss 对照显示第 0 步前向就有 6.22e-02 差异（相对 2.7e-03），
但历史真实文本上曾测到 rel 4.8e-06 —— 说明差异**依赖输入**。

三个假设逐一验：
  ① router top-2 索引是否不同（并列/累加顺序）；
  ② 分歧从第几层开始（第 0 层就分叉 ⇒ 单层实现差异；后面才分叉 ⇒ 累积/路由）；
  ③ 换真实文本后差异是否显著变小（验证"随机 id 平票"假设）。

运行（服务需先停）：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\pytorch_align_debug.py
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

MODEL_TG = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7_v4.tg"
MODEL_PT = r"D:\TGAI\models\V2.7\tgai_v27_v5_merged.pt"
NL = 2
SEQ = 128
DH = 64


def rope_np(T_, theta):
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(T_, dtype=np.float32)[:, None] * (theta ** (-j))[None, :], 2, axis=1)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def torch_layers(x, blk, cfg, cos, sin, caps, router_dumps, li):
    """逐层跑 torch 参考，并把每层输出、router 输入落盘。"""
    D, NH = cfg["d_model"], cfg["n_heads"]
    h = v27_ref.rmsnorm(x, blk["ln1"])
    q = (h @ blk["q"].T).view(-1, NH, DH)
    k = (h @ blk["k"].T).view(-1, NH, DH)
    v = (h @ blk["v"].T).view(-1, NH, DH)
    q, k = v27_ref.apply_rope(q, cos, sin), v27_ref.apply_rope(k, cos, sin)
    a = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                                       is_causal=True).transpose(0, 1).reshape(-1, D)
    x = x + a @ blk["out"].T
    h2 = v27_ref.rmsnorm(x, blk["ln2"])
    z = h2 @ blk["moe"]["router"].T
    if blk["moe"].get("qb_bias") is not None:
        z = z + blk["moe"]["qb_bias"]
    router_dumps.append((h2.detach().cpu().numpy(), z.detach().cpu().numpy()))
    x = x + v27_ref.moe(h2, blk["moe"], cfg)
    caps.append(x.detach().cpu().numpy())
    return x


def main():
    import v27_ref
    from tgai_go.read_tg import close_tg, read_tg
    from tgai_go.train.v27_train import TGState, V27Trainer

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    cfg_t = dict(d_model=int(h["d_model"]), n_heads=int(h["n_heads"]), head_dim=DH,
                 n_experts=int(h["n_experts"]), n_activated=int(h["n_activated"]),
                 rope_theta=float(h["rope_theta"]))
    V = int(h["vocab"])

    if not RUN.isolate():
        log("GPU 隔离失败，中止")
        raise SystemExit(3)
    try:
        import tgai_go.train as T
        T.set_default_device("cuda")
        # ---- ORV 侧：只装 NL 层 ----
        cfg = dict(d_model=cfg_t["d_model"], n_heads=cfg_t["n_heads"], d_ff=int(h["d_ff"]),
                   n_experts=cfg_t["n_experts"], n_activated=cfg_t["n_activated"],
                   n_layers=int(h["n_layers"]), rope_theta=cfg_t["rope_theta"])
        st = TGState(MODEL_TG)
        tr = V27Trainer(cfg, n_layers=NL)
        tr.load_all(st)
        emb = st["token_embedding.weight"]
        # ---- torch 侧：NL 层 ----
        ck = torch.load(MODEL_PT, map_location="cpu", mmap=True, weights_only=False)
        sd = ck["model_state_dict"]

        def P_(n):
            return torch.nn.Parameter(sd[n].float().to("cuda"), requires_grad=False)

        PT = {"embed": P_("token_embedding.weight"), "blocks": []}
        for i in range(NL):
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

        # ---- 两种输入 ----
        cases = []
        cases.append(("随机 id", np.random.default_rng(1234).integers(0, V, SEQ, dtype=np.int64)))
        try:
            from tgai_go.bpe import BOS_ID, UNK_ID, BpeTokenizer
            info2 = read_tg(MODEL_TG)
            tk = BpeTokenizer(info2["tokens"], info2["merges"], unk_id=UNK_ID, bos_id=BOS_ID)
            close_tg(info2)
            import glob, json, os
            got = None
            for p in sorted(glob.glob(os.path.join("D:\\SFT\u8bed\u6599\u5e93", "**", "*.jsonl"), recursive=True)):
                with open(p, encoding="utf-8", errors="replace") as f:
                    for _ in range(40):
                        line = f.readline()
                        if not line:
                            break
                        try:
                            t = (json.loads(line).get("text") or "")
                        except Exception:
                            continue
                        if len(t) > 400:
                            got = tk.encode(t, add_bos=True)[:SEQ]
                            break
                if got:
                    break
            if got and len(got) >= 64:
                cases.append(("真实文本", np.asarray(got, np.int64)))
        except Exception as e:
            log("真实文本准备失败：%s" % type(e).__name__)

        for tag, ids in cases:
            log("================ %s（%d tok）================" % (tag, len(ids)))
            cos_np, sin_np = rope_np(len(ids) - 1, cfg["rope_theta"])
            cos_t = T.tensor(cos_np); sin_t = T.tensor(sin_np)
            x0 = np.ascontiguousarray(emb[ids[:-1]], np.float32)
            xs = tr.forward_all(x0, cos_t, sin_t)          # ORV 逐层输出
            # torch 逐层
            cos_tt = torch.from_numpy(cos_np).cuda()
            sin_tt = torch.from_numpy(sin_np).cuda()
            xt = PT["embed"][torch.from_numpy(ids[:-1]).long().cuda()]
            caps, rdumps = [], []
            for i in range(NL):
                xt = torch_layers(xt, PT["blocks"][i], cfg_t, cos_tt, sin_tt, caps, rdumps, i)
            for i in range(NL + 1):
                a = np.asarray(xs[i], np.float32)
                b = caps[i - 1] if i > 0 else np.ascontiguousarray(emb[ids[:-1]], np.float32)
                d = float(np.abs(a - b).max())
                rel = d / max(1e-9, float(np.abs(b).max()))
                log("  层%-2d  ORV|max|=%.4g  torch|max|=%.4g  max|Δ|=%.3e  相对=%.2e"
                    % (i, float(np.abs(a).max()), float(np.abs(b).max()), d, rel))
            # router 对比（第 0、1 层）
            for li in range(NL):
                h2_o = np.asarray(tr.layers[li]["ln2"](T.tensor(xs[li])).to_numpy(), np.float32)
                W = np.asarray(tr.layers[li]["moe"].router.weight.to_numpy(), np.float32)
                bq = None
                try:
                    bq = np.asarray(tr.layers[li]["moe"].qb_bias.to_numpy(), np.float32)
                except Exception:
                    pass
                z_o = h2_o @ W.T + (bq if bq is not None else 0.0)
                h2_t, z_t = rdumps[li]
                top_o = np.argsort(-z_o, axis=-1)[:, :2]
                top_t = np.argsort(-z_t, axis=-1)[:, :2]
                same_order = int(np.sum(np.all(top_o == top_t, axis=-1)))
                same_set = int(np.sum([set(a) == set(b) for a, b in zip(top_o, top_t)]))
                n = len(top_o)
                log("  层%d router: h2 max|Δ|=%.3e  z max|Δ|=%.3e  top2 有序一致=%d/%d  集合一致=%d/%d"
                    % (li, float(np.abs(h2_o - h2_t).max()), float(np.abs(z_o - z_t).max()),
                       same_order, n, same_set, n))
                # 平票观察：top2 与第3名的差距
                srt = np.sort(z_o, axis=-1)
                gap = srt[:, -2] - srt[:, -3]
                log("       第2/3名分差：中位 %.3e  最小 %.3e（越小越容易翻）"
                    % (float(np.median(gap)), float(np.min(gap))))
        st.close()
    finally:
        try:
            tr = None
            st.close()
            T._cuda().empty_cache()
        except Exception:
            pass
        for _ in range(24):
            time.sleep(5)
            if RUN.vram() < 2000:
                break
        RUN.restore()


if __name__ == "__main__":
    main()
