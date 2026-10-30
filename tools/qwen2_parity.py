# -*- coding: utf-8 -*-
"""试金石 · Qwen2.5-0.5B 前向数值对齐（纯 CPU，不需要 GPU）

用 **tgai_go.train 的张量算子**实现 Qwen2 结构（RMSNorm + RoPE + GQA + SwiGLU + qkv bias），
加载**真实开源权重**（GGUF fp16），与 transformers 的 Qwen2 参考实现逐位比对 logits。

为什么重要：我们的模型是自己写的结构，测它只能证明"我们的模型能跑"。
换成 Qwen2 才知道**别人的结构**我们能不能表达 —— 这是"能否替代 PyTorch"的第一道门。

用法：
    python tools/qwen2_parity.py
（参考 logits 由 tools/qwen_ref.py 生成在 tools/_qwen_ref.npz）
"""
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for cand in (os.path.join(REPO, "python"), REPO):
    if os.path.isdir(os.path.join(cand, "tgai_go")) and cand not in sys.path:
        sys.path.insert(0, cand)

import tgai_go.train as T  # noqa: E402
from tgai_go.train import functional as F  # noqa: E402
from tgai_go.train import nn_ops as O  # noqa: E402

GGUF = r"D:\DSH-Workspace\_gguf_test\qwen2.5-0.5b-fp16.gguf"
REF = os.path.join(REPO, "tools", "_qwen_ref.npz")


def load_conv():
    spec = importlib.util.spec_from_file_location(
        "gguf_to_tg", os.path.join(REPO, "tools", "gguf_to_tg.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gguf_to_tg"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_weights(path, cfg):
    """按转换器同一套名字映射读 GGUF，返回 {tg_name: np.float32}。"""
    import gguf
    conv = load_conv()
    rd = gguf.GGUFReader(path)
    out = {}
    for t in rd.tensors:
        name = conv.map_weight_name(t.name, cfg)
        if name is None:
            continue
        out[name] = np.asarray(t.data, dtype=np.float32)
    return out


def t_of(w, name):
    a = w[name]
    return T.tensor(np.ascontiguousarray(a.astype(np.float32)))


def proj(x, w, name, out_features):
    """x @ W.T + b，W 为 [out, in]；朝向按 out_features 自动判定后交给 nn_ops.linear。"""
    """x @ W，其中 W 的存储朝向**不统一**（转换器对非方阵做了 dims 交换）。

    与其猜朝向，不如按「输出维必须等于 out_features」自动判断并转置。
    （这也是给转换器记的一笔账：同一个 .TG 里 [out,in] 与 [in,out] 混着出现。）
    """
    W = t_of(w, name)
    if W.shape[0] != out_features:
        assert W.shape[1] == out_features, (name, W.shape, out_features)
        W = W.T
    return O.linear(x, W)


def rmsnorm(x, w, eps):
    """RMSNorm 一律走已验证的 nn_ops（与 torch.nn.functional.rms_norm 对齐）。"""
    return O.rms_norm(x, w, eps)


def rope_tables(seq, head_dim, theta):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    pos = np.arange(seq, dtype=np.float64)
    ang = np.outer(pos, inv)                    # [S, head_dim/2]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def apply_rope(x, cos, sin):
    """x: [H, S, Dh]；cos/sin: [S, Dh/2] —— Qwen/LLaMA 的 rotate_half 形式。"""
    h, s, d = x.shape
    half = d // 2
    x1 = x[:, :, :half]
    x2 = x[:, :, half:]
    rot = T.cat([-x2, x1], dim=-1)              # rotate_half
    # ★ 必须是「前半 + 后半」重复（tile），不是逐元素重复（repeat）：
    #   rotate_half 的 cos 形如 [c0..c31, c0..c31]；
    #   若写成 repeat 会变成 [c0,c0,c1,c1,...] —— 位置 0 恰好 cos=1/sin=0 看不出来，
    #   一旦 pos>0 立刻全错（这正是本脚本最初的 bug）。
    c = T.tensor(np.concatenate([cos, cos], axis=-1)[None, :, :])   # [1,S,Dh]
    si = T.tensor(np.concatenate([sin, sin], axis=-1)[None, :, :])
    c = c.reshape(1, s, d)
    si = si.reshape(1, s, d)
    return x * c + rot * si


def main():
    global ids
    T.set_default_device("cpu")
    if not os.path.exists(REF):
        print("缺少参考 logits：先跑 python tools/qwen_ref.py")
        return 1
    ref = np.load(REF)
    ids_full, ref_logits_full = ref["ids"], ref["logits"]
    n = int(os.environ.get("QWEN_SEQ", "0")) or len(ids_full)
    ids = ids_full[:n]
    # 因果模型：前 n 个 token 的最后一个位置 logits 等于用前 n 个 token 跑出来的最后一个位置
    ref_logits = ref_logits_full[:, :n, :]
    print("序列长度 %d（QWEN_SEQ 可缩短做分段定位）" % n)

    conv = load_conv()
    cfg = conv.ARCH_CONFIG["qwen2"]
    w = load_weights(GGUF, cfg)
    print("权重张量 %d 个（已按转换器映射命名）" % len(w))

    # ---- 结构超参（来自 GGUF 元数据，实测值）----
    H, KV, D, DH = 14, 2, 896, 64
    NL, FF, EPS, THETA = 24, 4864, 1e-6, 1e6

    seq = len(ids)
    # 词嵌入：目前公开 API 没有 gather/embedding 算子，先用 numpy 查表（记为缺口）
    x = O.embedding(T.tensor(np.ascontiguousarray(w["token_embedding.weight"])),
                    np.asarray(ids, dtype=np.int64))

    cos, sin = rope_tables(seq, DH, THETA)
    # 因果掩码（加性大负数）
    mask = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)
    mask_t = T.tensor(mask)
    scale = float(DH) ** -0.5

    for i in range(NL):
        p = "blocks.%d." % i
        # ---- attention ----
        h = rmsnorm(x, t_of(w, p + "ln1.weight"), EPS)
        q = proj(h, w, p + "attn.q_proj.weight", H * DH) + t_of(w, p + "attn.q_proj.bias")
        k = proj(h, w, p + "attn.k_proj.weight", KV * DH) + t_of(w, p + "attn.k_proj.bias")
        v = proj(h, w, p + "attn.v_proj.weight", KV * DH) + t_of(w, p + "attn.v_proj.bias")
        # [S, H*Dh] -> [H, S, Dh]
        q = q.reshape(seq, H, DH).permute(1, 0, 2)
        k = k.reshape(seq, KV, DH).permute(1, 0, 2)
        v = v.reshape(seq, KV, DH).permute(1, 0, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # GQA：kv 头 2 -> 14
        k = k.repeat_interleave(H // KV, dim=0)
        v = v.repeat_interleave(H // KV, dim=0)
        scores = (q @ k.swap_last2()) * scale + mask_t     # [H,S,S]
        probs = F.softmax(scores, -1)
        o = probs @ v                                       # [H,S,Dh]
        o = o.permute(1, 0, 2).reshape(seq, H * DH)
        x = x + proj(o, w, p + "attn.out_proj.weight", D)

        # ---- MLP (SwiGLU) ----
        h2 = rmsnorm(x, t_of(w, p + "ln2.weight"), EPS)
        g = proj(h2, w, p + "ffn.w1.weight", FF)
        u = proj(h2, w, p + "ffn.w2.weight", FF)
        act = F.silu(g) * u
        x = x + proj(act, w, p + "ffn.w3.weight", D)

    x = rmsnorm(x, t_of(w, "final_norm.weight"), EPS)
    # lm_head：GGUF 有独立 output.weight（与 embedding 不同权重）
    import gguf
    rd = gguf.GGUFReader(GGUF)
    head = None
    for t in rd.tensors:
        if t.name == "output.weight":
            head = np.asarray(t.data, dtype=np.float32)
            break
    if head is None:
        raise SystemExit("GGUF 里没有 output.weight")
    logits = O.linear(x, T.tensor(np.ascontiguousarray(head)))
    got = np.asarray(logits.to_numpy(), dtype=np.float32)[None, :, :]

    print("我们 logits:", got.shape, "最后一位前 5：", got[0, -1, :5])

    d = np.abs(got.astype(np.float64) - ref_logits.astype(np.float64))
    max_abs = float(d.max())
    denom = float(np.abs(ref_logits).max()) or 1.0
    rel = max_abs / denom
    # 逐位置相对误差（更严格：每个 token 单独看）
    per = []
    for t in range(ref_logits.shape[1]):
        den = float(np.abs(ref_logits[0, t]).max()) or 1.0
        per.append(float(d[0, t].max()) / den)

    print("")
    print("最大绝对误差 max|Δ| = %.3e" % max_abs)
    print("相对误差（对参考最大值）= %.3e" % rel)
    print("逐 token 相对误差：", ["%.2e" % v for v in per])
    ok = rel <= 1e-5
    print("")
    print("判据 ≤1e-5：%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
