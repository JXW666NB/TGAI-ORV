# -*- coding: utf-8 -*-
"""faithful 模式端到端对拍：ORV（--faithful-v3 六项全开）vs PyTorch 真 V3。

与 `lab/e2e/e2e_parity.py` 的区别：那份把 6 项差异**中和掉**了，
这份**两边都全开**，验证「补齐后的 ORV 与真 V3 同构」。

用法（V100，需 torch + ORV 同进程）：
    C:\\Python311\\python.exe faithful_parity.py --d 128 --layers 2 --csa 2 --vocab 1024 --seq 32
    C:\\Python311\\python.exe faithful_parity.py --d 1024 --layers 2 --csa 4 --vocab 96000 --seq 512
"""
import argparse
import sys

sys.path.insert(0, r"D:\orv_pip_0962\Lib\site-packages")   # ORV
sys.path.insert(0, r"D:\TGAI-V3-1.5B\03_src")              # model_v3.py

import numpy as np
import torch

import model_v3 as M
from tgai_go.train import v3_csa as C
from tgai_go.train import v3_train_cli as CLI
import tgai_go.train as T

T.set_default_device("cuda")


def torch_cfg(a):
    return M.TGAIV3Config(
        vocab_size=a.vocab, d_model=a.d, n_layers=a.layers, n_heads=a.n_q,
        n_kv_heads=0, d_ff=a.d_ff or 4 * a.d, n_experts=a.E, n_activated=a.top_k,
        n_shared_experts=1, max_seq_len=a.seq,
        csa_compress=4, csa_conv_kernel=3, csa_local_window=0, csa_latent=0,
        attn_global_every=a.csa, rope_theta=500000.0,
        # ★ 真 V3 的六项默认值，全部保持 True/quantile/256
        attn_qk_norm=True, tie_embeddings=True, moe_post_norm=True,
        moe_balance="quantile", attn_res_enabled=True, attn_res_block=4,
        pad_token_id=256 if a.vocab > 256 else 0,
    )


def orv_cfg(a):
    cfg = dict(d=a.d, d_ff=a.d_ff or 4 * a.d, E=a.E, layers=a.layers, n_q=a.n_q,
               n_kv=2, vocab=a.vocab, seq=a.seq, attn_res=True, attn_res_block=4,
               top_k=a.top_k, csa=a.csa, n_kv_attn=a.n_q, rope_theta=500000.0,
               csa_compress=4, csa_conv_kernel=3, csa_local_window=0,
               tie=True, moe_post_norm=True, qk_norm=True,
               pad_id=256 if a.vocab > 256 else 0, moe_balance="quantile")
    return cfg


def map_weights(mod, cfg, kinds, orv_dtype=np.float32):
    """PyTorch state_dict -> ORV 参数 dict。名字逐一对照（读两边源码确认）。"""
    sd = {k: v.detach().cpu().numpy() for k, v in mod.named_parameters()}
    w = {}
    # --- 全局：embed / final_norm / head ---
    w["emb"] = sd["embed.weight"]
    w["ln_f"] = sd["final_norm.weight"]
    w["head"] = sd["embed.weight"] if cfg.get("tie") else sd["lm_head.weight"]
    for i, kind in enumerate(kinds):
        p = "blocks.%d." % i
        L = {}
        L["ln1"] = sd[p + "ln1.weight"]
        L["ln2"] = sd[p + "ln2.weight"]
        L["router"] = sd[p + "ffn.router.weight"]
        L["W1"] = [sd["%sffn.experts.%d.w1.weight" % (p, e)] for e in range(cfg["E"])]
        L["W2"] = [sd["%sffn.experts.%d.w2.weight" % (p, e)] for e in range(cfg["E"])]
        L["W3"] = [sd["%sffn.experts.%d.w3.weight" % (p, e)] for e in range(cfg["E"])]
        L["S1"] = sd[p + "ffn.shared_experts.0.w1.weight"]
        L["S2"] = sd[p + "ffn.shared_experts.0.w2.weight"]
        L["S3"] = sd[p + "ffn.shared_experts.0.w3.weight"]
        if cfg.get("moe_post_norm"):
            L["post_norm"] = sd[p + "ffn.post_norm.weight"]
        if cfg.get("attn_res"):
            L["res_w_attn"] = sd["attn_res_proj.%d.proj.weight" % i].reshape(-1)
            L["res_gn_attn"] = sd["attn_res_proj.%d.norm.weight" % i]
            L["res_w_mlp"] = sd["mlp_res_proj.%d.proj.weight" % i].reshape(-1)
            L["res_gn_mlp"] = sd["mlp_res_proj.%d.norm.weight" % i]
        if kind == "csa":
            sub = {k[len(p + "attn."):]: v for k, v in sd.items() if k.startswith(p + "attn.")}
            cw = C.weights_from_torch_state_dict(sub, cfg["csa_cfg"])
            L["csa"] = {k: (None if v is None else np.asarray(
                v.to_numpy() if hasattr(v, "to_numpy") else v, orv_dtype))
                for k, v in cw.items()}
        else:
            L["wq"] = sd[p + "attn.q_proj.weight"]
            L["wk"] = sd[p + "attn.k_proj.weight"]
            L["wv"] = sd[p + "attn.v_proj.weight"]
            L["wo"] = sd[p + "attn.o_proj.weight"]
            if cfg.get("qk_norm"):
                L["qn"] = sd[p + "attn.q_norm.weight"]
                L["kn"] = sd[p + "attn.k_norm.weight"]
        w.setdefault("layers", []).append(L)
    return w


def set_orv(P, w):
    """把 numpy 权重灌进 ORV 参数结构：递归重建 dict，值换成新张量。

    `build()` 返回的是普通 dict（值 = Tensor 或 list/dict），所以**直接替换**即可，
    不需要任何 in-place 赋值 API。
    """
    def rebuild(node, src):
        if isinstance(src, dict):
            out = {}
            for k, v in src.items():
                if k not in node:
                    raise SystemExit("ORV 没有键 %r（现有：%s）"
                                     % (k, sorted(node.keys())[:12]))
                out[k] = rebuild(node[k], v)
            # 保留 node 里未被覆盖的键（例如 CSA 的 None 槽位）
            for k, v in node.items():
                if k not in out:
                    out[k] = v
            return out
        if isinstance(src, list):
            return [rebuild(n, v) for n, v in zip(node, src)]
        if node is None:
            return None
        arr = np.asarray(src, np.float32)
        if tuple(node.shape) != arr.shape:
            raise SystemExit("形状不符：ORV %s vs torch %s" % (node.shape, arr.shape))
        return T.tensor(arr, requires_grad=True)

    return rebuild(P, w)


def _unused_assign():      # 保留占位，避免误用
    return None


def _old_assign(dst_obj, src_val):
    return None



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--E", type=int, default=2)
    ap.add_argument("--n_q", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--csa", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tcfg = torch_cfg(a)
    torch.manual_seed(a.seed)
    mod = M.TGAIV3Model(tcfg).cuda().eval()

    cfg = orv_cfg(a)
    rng = np.random.default_rng(a.seed)
    P = CLI.build(cfg, rng)

    kinds = CLI.layer_kinds(cfg)
    print("[cfg] ORV kinds:", kinds)
    print("[cfg] ORV 层数=%d csa=%d global=%d" % (len(kinds),
          sum(1 for k in kinds if k == "csa"), sum(1 for k in kinds if k == "global")))

    w = map_weights(mod, cfg, kinds)
    try:
        P = set_orv(P, w)
        print("[map] 权重已灌入 ORV")
    except Exception as e:
        print("[map] 灌权重失败：%r" % (e,))
        print("[map] 各层 ORV 键：", sorted(P["layers"][0].keys()))
        raise

    ids = np.random.default_rng(a.seed + 1).integers(0, a.vocab, a.seq).astype(np.int64)
    with torch.no_grad():
        y_t = mod(torch.tensor(ids, device="cuda")[None])
        y_t = y_t[0] if isinstance(y_t, tuple) else y_t
        lt = y_t.cpu().numpy()

    plan = None
    csac = cfg["csa_cfg"]
    plan = C.make_plan(csac, 1, a.seq)
    cos, sin = C.rope_tables(csac, a.seq)
    lo = CLI.forward_logits(P, cfg, kinds, plan, cos, sin, ids)
    lo = np.asarray(lo.to_numpy(), np.float32).reshape(a.seq, a.vocab)
    lt = lt.reshape(a.seq, a.vocab)
    d = np.abs(lo - lt)
    scale = max(float(np.abs(lt).max()), 1e-30)
    print("\n[result] logits max_abs=%.3e  max_rel=%.3e  |ref|max=%.4g"
          % (float(d.max()), float(d.max()) / scale, scale))
    print("[result] 判定：%s（判据 1e-5）"
          % ("PASS" if float(d.max()) <= 1e-5 else "FAIL"))


if __name__ == "__main__":
    main()
