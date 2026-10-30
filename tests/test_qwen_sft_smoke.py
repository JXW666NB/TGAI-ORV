# -*- coding: utf-8 -*-
"""SFT 链路冒烟测试（随机权重的迷你 Qwen，纯 CPU，CI 可跑）。

为什么用「固定探针序列」而不是训练 loss：
    训练样本轮转 ⇒ 每步 loss 不可比（不同样本难度不同）。我第一版就是用训练
    loss 判定，结果误判为 FAIL（其实链路是通的）。
"""
import importlib.util
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    import sys
    sys.path.insert(0, os.path.join(REPO, "python"))
    path = os.path.join(REPO, "examples", "qwen25_sft.py")
    spec = importlib.util.spec_from_file_location("qwen25_sft", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tiny_sft_probe_loss_decreases():
    q = _load()
    import tgai_go.train as T
    T.set_default_device("cpu")

    cfg = dict(q.TINY_CFG)
    cfg.update(D=32, L=1, NH=2, KV=1, FF=64, V=128)
    model = q.Qwen25(cfg, q.make_tiny_weights(cfg))
    seqs = q.tiny_sequences(cfg, n=3, seq_len=24)

    # 探针现在是「训练前基线 + 定期评估」，所以条数不再等于步数
    losses, probes = q.sft(model, seqs, 8, 3e-3, 24, probe=seqs[0], probe_every=4)
    assert len(probes) >= 2, probes
    assert all(np.isfinite(p) for p in probes)
    assert probes[-1] < probes[0], (probes[0], probes[-1])


def test_probe_loss_requires_no_grad_side_effects():
    """探针评估不能污染参数梯度（否则训练会被评估悄悄影响）。"""
    q = _load()
    import tgai_go.train as T
    T.set_default_device("cpu")
    cfg = dict(q.TINY_CFG)
    cfg.update(D=32, L=1, NH=2, KV=1, FF=64, V=128)
    model = q.Qwen25(cfg, q.make_tiny_weights(cfg))
    seq = q.tiny_sequences(cfg, n=1, seq_len=16)[0]
    for p in model.params():
        p.zero_grad()
    before = q._probe_loss(model, seq, 16)
    after = q._probe_loss(model, seq, 16)
    assert abs(before - after) < 1e-6
    assert all(getattr(p, "grad", None) is None for p in model.params())


def test_hf_name_mapping():
    q = _load()
    cases = {
        "model.embed_tokens.weight": "token_embedding.weight",
        "lm_head.weight": "lm_head.weight",
        "model.norm.weight": "final_norm.weight",
        "model.layers.0.input_layernorm.weight": "blocks.0.ln1.weight",
        "model.layers.3.post_attention_layernorm.weight": "blocks.3.ln2.weight",
        "model.layers.2.self_attn.q_proj.weight": "blocks.2.attn.q_proj.weight",
        "model.layers.2.self_attn.k_proj.bias": "blocks.2.attn.k_proj.bias",
        "model.layers.1.mlp.gate_proj.weight": "blocks.1.ffn.w1.weight",
        "model.layers.1.mlp.up_proj.weight": "blocks.1.ffn.w2.weight",
        "model.layers.1.mlp.down_proj.weight": "blocks.1.ffn.w3.weight",
    }
    for hf, ours in cases.items():
        assert q.hf_name_to_ours(hf) == ours, hf
    assert q.hf_name_to_ours("model.layers.0.self_attn.rotary_emb.inv_freq") is None
