# -*- coding: utf-8 -*-
"""Qwen3 前向的 CI 用例：与 PyTorch 孪生逐位比对（纯 CPU）。

Qwen3 与 Qwen2.5 的三处硬差异，本测试全部覆盖：
1. **head_dim != hidden / heads**（0.6B：128 vs 64）—— 这里用 DH=32 vs 96/4=24；
2. **q_norm / k_norm**：每个 head 对 head_dim 做 RMSNorm，且必须在 **RoPE 之前**；
3. **无 qkv bias**（Qwen2.5 有）。

另外校验 tie 语义与参数去重（tie 时 lm_head 与词嵌入是同一对象，
重复传给优化器会被更新两次 —— 实测一步 0.98 而非 0.99）。
"""
import importlib.util
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: D=96, NH=4 ⇒ hidden/heads = 24，但 DH=32：**故意不等**，正是 Qwen3-0.6B 的情形
CFG3 = dict(V=128, D=96, L=2, NH=4, KV=2, DH=32, FF=160,
            THETA=1e6, EPS=1e-6, QK_NORM=True, QKV_BIAS=False, TIE=True)


def _load():
    import sys
    sys.path.insert(0, os.path.join(REPO, "python"))
    path = os.path.join(REPO, "examples", "qwen25_sft.py")
    spec = importlib.util.spec_from_file_location("qwen25_sft", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _torch_twin_qwen3(w, ids, cfg):
    """逐行照抄 Qwen3：reshape → q_norm/k_norm → rope → GQA 展开 → sdpa。"""
    import torch.nn.functional as TF
    D, L, NH, KV = cfg["D"], cfg["L"], cfg["NH"], cfg["KV"]
    DH = cfg["DH"]
    B, S = ids.shape
    p = {k: torch.from_numpy(v.copy()) for k, v in w.items()}
    if "lm_head.weight" not in p:                       # tie
        p["lm_head.weight"] = p["token_embedding.weight"]

    inv = 1.0 / (cfg["THETA"] ** (np.arange(0, DH, 2, dtype=np.float64) / DH))
    ang = np.outer(np.arange(S, dtype=np.float64), inv)
    cos = torch.from_numpy(np.concatenate([np.cos(ang)] * 2, -1).astype(np.float32))[None]
    sin = torch.from_numpy(np.concatenate([np.sin(ang)] * 2, -1).astype(np.float32))[None]

    def rope(t):
        h = DH // 2
        rot = torch.cat([-t[..., h:], t[..., :h]], dim=-1)
        return t * cos + rot * sin

    def rms(t, wgt):                                    # 只归一化最后一维（=head_dim）
        v = t.pow(2).mean(-1, keepdim=True)
        return t * torch.rsqrt(v + cfg["EPS"]) * wgt

    x = p["token_embedding.weight"][torch.from_numpy(ids)].reshape(B, S, D)
    for i in range(L):
        b = "blocks.%d." % i
        h = rms(x, p[b + "ln1.weight"]).reshape(B * S, D)
        q = (h @ p[b + "attn.q_proj.weight"].T).reshape(B, S, NH, DH).permute(0, 2, 1, 3)
        k = (h @ p[b + "attn.k_proj.weight"].T).reshape(B, S, KV, DH).permute(0, 2, 1, 3)
        v = (h @ p[b + "attn.v_proj.weight"].T).reshape(B, S, KV, DH).permute(0, 2, 1, 3)
        if cfg.get("QK_NORM"):
            q = rms(q, p[b + "attn.q_norm.weight"])
            k = rms(k, p[b + "attn.k_norm.weight"])
        q = rope(q.reshape(B * NH, S, DH))
        k = rope(k.reshape(B * KV, S, DH))
        k = k.reshape(B, KV, S, DH).repeat_interleave(NH // KV, dim=1).reshape(B * NH, S, DH)
        v = v.reshape(B, KV, S, DH).repeat_interleave(NH // KV, dim=1).reshape(B * NH, S, DH)
        o = TF.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, NH * DH)
        x = x + (o @ p[b + "attn.out_proj.weight"].T).reshape(B, S, D)
        h2 = rms(x, p[b + "ln2.weight"]).reshape(B * S, D)
        act = TF.silu(h2 @ p[b + "ffn.w1.weight"].T) * (h2 @ p[b + "ffn.w2.weight"].T)
        x = x + (act @ p[b + "ffn.w3.weight"].T).reshape(B, S, D)
    z = rms(x, p["final_norm.weight"]).reshape(B * S, D)
    return (z @ p["lm_head.weight"].T).reshape(B, S, cfg["V"]).detach().numpy()


def test_qwen3_forward_matches_torch_twin():
    q = _load()
    import tgai_go.train as T
    T.set_default_device("cpu")

    w = q.make_tiny_weights(CFG3, seed=3)
    model = q.Qwen25(CFG3, w)
    ids = np.random.default_rng(0).integers(0, CFG3["V"], size=(2, 12)).astype(np.int64)

    ours = np.asarray(model.forward(ids).to_numpy(), dtype=np.float64)
    ref = _torch_twin_qwen3(w, ids, CFG3).astype(np.float64)
    rel = np.abs(ours - ref).max() / (np.abs(ref).max() or 1.0)
    assert rel <= 1e-5, rel


def test_qwen3_head_dim_is_not_hidden_over_heads():
    """Qwen3-0.6B 的真实情形：DH=32 而 D//NH=24。若实现里用了 D//NH，这里会崩。"""
    q = _load()
    import tgai_go.train as T
    T.set_default_device("cpu")
    assert CFG3["DH"] != CFG3["D"] // CFG3["NH"]
    m = q.Qwen25(CFG3, q.make_tiny_weights(CFG3))
    out = m.forward(np.zeros((1, 8), dtype=np.int64))
    assert tuple(out.shape) == (1, 8, CFG3["V"])


def test_qwen3_qk_norm_is_wired_and_ordered():
    """接线验证要用**强非均匀**的 q_norm/k_norm 权重。

    踩过的坑：q_norm 权重接近 1 时，它对每个 head 近似是**常数缩放**，
    而这个常数会被 softmax 基本抵消 —— 实测开/关差异只有 1.8e-7，测不出接线错误。
    真实 Qwen3 的 q_norm 最大 11.8、k_norm 最大 96.5，是强逐通道整形，不可省。
    """
    q = _load()
    import tgai_go.train as T
    T.set_default_device("cpu")
    w = q.make_tiny_weights(CFG3)
    strong = np.linspace(0.2, 3.0, CFG3["DH"], dtype=np.float32)
    for i in range(CFG3["L"]):
        w["blocks.%d.attn.q_norm.weight" % i] = strong.copy()
        w["blocks.%d.attn.k_norm.weight" % i] = strong[::-1].copy()
    # ★ 必须用**互不相同**的 token：全 0 时所有位置相同 ⇒ 注意力权重恒为均匀，
    #   q_norm 怎么变都不影响输出（实测差异 2.9e-7，测不出接线错误）。
    ids = np.random.default_rng(7).integers(0, CFG3["V"], size=(1, 10)).astype(np.int64)
    a = np.asarray(q.Qwen25(dict(CFG3), w).forward(ids).to_numpy())
    b = np.asarray(q.Qwen25(dict(CFG3, QK_NORM=False), w).forward(ids).to_numpy())
    rel = np.abs(a - b).max() / (np.abs(a).max() or 1.0)
    assert rel > 1e-3, rel        # 关掉 qk_norm 必须明显改变输出


def test_qwen3_tie_shares_one_parameter_and_dedups():
    q = _load()
    import tgai_go.train as T
    T.set_default_device("cpu")
    w = q.make_tiny_weights(CFG3)                       # TIE=True ⇒ 不生成独立 lm_head
    m = q.Qwen25(CFG3, w)
    assert m.p["lm_head.weight"] is m.p["token_embedding.weight"]
    ids = [id(x) for x in m.params()]
    assert len(ids) == len(set(ids)), "params() 必须去重，否则 tie 参数被更新两次"


def test_qwen3_tie_with_duplicate_lm_head_is_dropped():
    """官方 Qwen3 safetensors 带 lm_head.weight（与词嵌入相同），必须被吸收而不是各训一份。"""
    q = _load()
    import tgai_go.train as T
    T.set_default_device("cpu")
    w = q.make_tiny_weights(dict(CFG3, TIE=False))
    w["lm_head.weight"] = w["token_embedding.weight"].copy()
    m = q.Qwen25(dict(CFG3, TIE=True), w)
    assert m.p["lm_head.weight"] is m.p["token_embedding.weight"]


def test_qwen3_tie_mismatch_raises():
    q = _load()
    import tgai_go.train as T
    T.set_default_device("cpu")
    w = q.make_tiny_weights(dict(CFG3, TIE=False))      # lm_head 是独立随机值
    with pytest.raises(ValueError) as ei:
        q.Qwen25(dict(CFG3, TIE=True), w)
    assert "TIE" in str(ei.value) and "修法" in str(ei.value)
