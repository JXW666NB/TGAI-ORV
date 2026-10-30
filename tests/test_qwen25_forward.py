# -*- coding: utf-8 -*-
"""Qwen2.5 前向的 CI 用例：与 PyTorch 孪生逐位比对（纯 CPU）。

覆盖 Qwen2.5 与我们自有模型的四个差异：
  1. GQA（NH 个 Q 头 : KV 个 KV 头）
  2. qkv **bias**（我们没有）
  3. **独立 lm_head**（不绑词嵌入）
  4. RoPE 标准 rotate_half + 大 base（1e6）
真权重对齐（GGUF/safetensors）在 examples/qwen25_sft.py 里跑，这里用合成小权重保证 CI 可跑。
"""
import importlib.util
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = dict(V=128, D=64, L=2, NH=4, KV=2, FF=160, THETA=1e6, EPS=1e-6)


def _load():
    import sys
    sys.path.insert(0, os.path.join(REPO, "python"))
    path = os.path.join(REPO, "examples", "qwen25_sft.py")
    spec = importlib.util.spec_from_file_location("qwen25_sft", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _weights(cfg, seed=3):
    rng = np.random.default_rng(seed)
    D, L, NH, KV, FF, V = cfg["D"], cfg["L"], cfg["NH"], cfg["KV"], cfg["FF"], cfg["V"]
    DH = D // NH
    k = 1.0 / np.sqrt(D)
    w = {"token_embedding.weight": rng.normal(0, 0.02, (V, D)).astype(np.float32),
         "lm_head.weight": rng.normal(0, 0.02, (V, D)).astype(np.float32),
         "final_norm.weight": np.ones(D, np.float32)}
    for i in range(L):
        b = "blocks.%d." % i
        w[b + "ln1.weight"] = np.ones(D, np.float32)
        w[b + "ln2.weight"] = np.ones(D, np.float32)
        w[b + "attn.q_proj.weight"] = rng.normal(0, k, (NH * DH, D)).astype(np.float32)
        w[b + "attn.k_proj.weight"] = rng.normal(0, k, (KV * DH, D)).astype(np.float32)
        w[b + "attn.v_proj.weight"] = rng.normal(0, k, (KV * DH, D)).astype(np.float32)
        w[b + "attn.out_proj.weight"] = rng.normal(0, k, (D, NH * DH)).astype(np.float32)
        w[b + "attn.q_proj.bias"] = rng.normal(0, 0.01, (NH * DH,)).astype(np.float32)
        w[b + "attn.k_proj.bias"] = rng.normal(0, 0.01, (KV * DH,)).astype(np.float32)
        w[b + "attn.v_proj.bias"] = rng.normal(0, 0.01, (KV * DH,)).astype(np.float32)
        w[b + "ffn.w1.weight"] = rng.normal(0, k, (FF, D)).astype(np.float32)
        w[b + "ffn.w2.weight"] = rng.normal(0, k, (FF, D)).astype(np.float32)
        w[b + "ffn.w3.weight"] = (rng.normal(0, k, (D, FF)) / np.sqrt(FF)).astype(np.float32)
    return w


def _torch_forward(w, ids, cfg):
    """PyTorch 孪生：同样的权重、同样的结构。"""
    import torch.nn.functional as TF
    D, L, NH, KV, FF = cfg["D"], cfg["L"], cfg["NH"], cfg["KV"], cfg["FF"]
    DH = D // NH
    B, S = ids.shape
    p = {k: torch.from_numpy(v.copy()) for k, v in w.items()}

    inv = 1.0 / (cfg["THETA"] ** (np.arange(0, DH, 2, dtype=np.float64) / DH))
    ang = np.outer(np.arange(S, dtype=np.float64), inv)
    c = np.concatenate([np.cos(ang)] * 2, axis=-1).astype(np.float32)
    s = np.concatenate([np.sin(ang)] * 2, axis=-1).astype(np.float32)
    cos = torch.from_numpy(c)[None, :, :]
    sin = torch.from_numpy(s)[None, :, :]

    def rope(t):
        half = DH // 2
        rot = torch.cat([-t[..., half:], t[..., :half]], dim=-1)
        return t * cos + rot * sin

    x = p["token_embedding.weight"][torch.from_numpy(ids)].reshape(B, S, D)
    for i in range(L):
        b = "blocks.%d." % i
        h = TF.rms_norm(x, (D,), p[b + "ln1.weight"], cfg["EPS"]).reshape(B * S, D)
        q = (h @ p[b + "attn.q_proj.weight"].T + p[b + "attn.q_proj.bias"])
        k = (h @ p[b + "attn.k_proj.weight"].T + p[b + "attn.k_proj.bias"])
        v = (h @ p[b + "attn.v_proj.weight"].T + p[b + "attn.v_proj.bias"])
        q = q.reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        k = k.reshape(B, S, KV, DH).permute(0, 2, 1, 3).reshape(B * KV, S, DH)
        v = v.reshape(B, S, KV, DH).permute(0, 2, 1, 3).reshape(B * KV, S, DH)
        q, k = rope(q), rope(k)
        k = k.reshape(B, KV, S, DH).repeat_interleave(NH // KV, dim=1).reshape(B * NH, S, DH)
        v = v.reshape(B, KV, S, DH).repeat_interleave(NH // KV, dim=1).reshape(B * NH, S, DH)
        o = TF.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
        x = x + (o @ p[b + "attn.out_proj.weight"].T).reshape(B, S, D)
        h2 = TF.rms_norm(x, (D,), p[b + "ln2.weight"], cfg["EPS"]).reshape(B * S, D)
        act = TF.silu(h2 @ p[b + "ffn.w1.weight"].T) * (h2 @ p[b + "ffn.w2.weight"].T)
        x = x + (act @ p[b + "ffn.w3.weight"].T).reshape(B, S, D)
    z = TF.rms_norm(x, (D,), p["final_norm.weight"], cfg["EPS"]).reshape(B * S, D)
    return (z @ p["lm_head.weight"].T).reshape(B, S, cfg["V"]).detach().numpy()


def test_qwen25_forward_matches_torch():
    q = _load()
    import tgai_go.train as T
    T.set_default_device("cpu")

    w = _weights(CFG)
    model = q.Qwen25(CFG, w)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, CFG["V"], size=(2, 12)).astype(np.int64)

    ours = np.asarray(model.forward(ids).to_numpy(), dtype=np.float32)
    ref = _torch_forward(w, ids, CFG)

    d = np.abs(ours.astype(np.float64) - ref.astype(np.float64)).max()
    rel = d / (float(np.abs(ref).max()) or 1.0)
    assert rel <= 1e-5, rel


def test_qwen25_gqa_shape_sanity():
    """GQA：k/v 只有 KV 个头，展开后必须与 Q 头数一致（否则 sdpa 会形状错）。"""
    q = _load()
    import tgai_go.train as T
    T.set_default_device("cpu")
    w = _weights(CFG)
    model = q.Qwen25(CFG, w)
    ids = np.zeros((1, 8), dtype=np.int64)
    logits = model.forward(ids)
    assert tuple(logits.shape) == (1, 8, CFG["V"])
