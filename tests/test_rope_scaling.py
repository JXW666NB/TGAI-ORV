# -*- coding: utf-8 -*-
"""RoPE 缩放（llama3 / linear）与官方 transformers 对拍 + 未支持类型必须报错。

背景：Llama-3.x、Mistral-v0.3 等模型的 config.json 里都有 `rope_scaling`。
我们原来**只读 rope_theta、静默忽略 scaling** —— 短上下文看着差不多，长上下文直接错，
而且不报错。这类"静默算错"比崩溃危险得多。

这条测试用**别人的实现**（`transformers.LlamaForCausalLM`）做参照，
把我方权重灌进官方模型，比较 logits。
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
# 本文件用 transformers 的 LlamaForCausalLM 当参照物；没装就跳过（CI 现在装了，所以这条路是真跑的）。
#
# 注意：**不能**只用 `pytest.importorskip("transformers")`。它只捕获 ImportError，
# 而 transformers 懒加载 scipy 时，在本机（Windows + scipy 的 .libs）会抛
#   FileNotFoundError: [WinError 206] 文件名或扩展名太长: '...\\scipy.libs'
# 这种"环境坏了"的情况会让整条测试**失败**而不是跳过，把真正要测的 RoPE 逻辑淹没掉。
# 所以这里显式接住任何异常并 skip，同时把原因写进 skip 消息（不是静默吞掉）。
try:
    import transformers  # noqa: F401
except Exception as _exc:  # pragma: no cover - 取决于机器环境
    pytest.skip("transformers 不可用（环境问题，非本库缺陷）：%s: %s"
                % (type(_exc).__name__, str(_exc)[:160]), allow_module_level=True)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))


def _load_example():
    path = os.path.join(REPO, "examples", "qwen25_sft.py")
    spec = importlib.util.spec_from_file_location("qwen25_sft_rope", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen25_sft_rope"] = mod
    spec.loader.exec_module(mod)
    return mod


# Llama 形状：DH = D/NH（Llama 没有单独的 head_dim）、无 qk_norm、无 qkv bias、不 tie
CFG = dict(V=64, D=32, L=2, NH=4, KV=2, DH=8, FF=64, THETA=500000.0, EPS=1e-5,
           QK_NORM=False, QKV_BIAS=False, TIE=False,
           ROPE_SCALING={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                         "high_freq_factor": 4.0,
                         "original_max_position_embeddings": 128})

OURS_TO_HF_LAYER = [
    ("ln1.weight", "input_layernorm.weight"),
    ("ln2.weight", "post_attention_layernorm.weight"),
    ("attn.q_proj.weight", "self_attn.q_proj.weight"),
    ("attn.k_proj.weight", "self_attn.k_proj.weight"),
    ("attn.v_proj.weight", "self_attn.v_proj.weight"),
    ("attn.out_proj.weight", "self_attn.o_proj.weight"),
    ("ffn.w1.weight", "mlp.gate_proj.weight"),
    ("ffn.w2.weight", "mlp.up_proj.weight"),
    ("ffn.w3.weight", "mlp.down_proj.weight"),
]


def _build_hf(w, cfg):
    # 这里是 transformers 真正加载模型的地方，也是本机 scipy `.libs` 那个
    # FileNotFoundError(WinError 206) 的触发点。环境问题应**跳过并说明原因**，
    # 而不是让 RoPE 对照测试失败（那会把"环境坏了"误报成"实现错了"）。
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except Exception as exc:  # pragma: no cover - 取决于机器环境
        pytest.skip("transformers/scipy 在本机不可用（环境问题，非本库缺陷）：%s: %s"
                    % (type(exc).__name__, str(exc)[:160]))
    c = LlamaConfig(
        vocab_size=cfg["V"], hidden_size=cfg["D"], num_hidden_layers=cfg["L"],
        num_attention_heads=cfg["NH"], num_key_value_heads=cfg["KV"],
        intermediate_size=cfg["FF"], max_position_embeddings=128,
        rms_norm_eps=cfg["EPS"], rope_theta=cfg["THETA"],
        rope_scaling=dict(cfg["ROPE_SCALING"]), tie_word_embeddings=False,
        attention_bias=False)
    model = LlamaForCausalLM(c)
    sd = model.state_dict()
    mapping = {"token_embedding.weight": "model.embed_tokens.weight",
               "final_norm.weight": "model.norm.weight",
               "lm_head.weight": "lm_head.weight"}
    for i in range(cfg["L"]):
        for ours, hf in OURS_TO_HF_LAYER:
            mapping["blocks.%d.%s" % (i, ours)] = "model.layers.%d.%s" % (i, hf)
    new = {}
    for k, arr in w.items():
        hf = mapping.get(k)
        if hf and hf in sd and tuple(sd[hf].shape) == tuple(arr.shape):
            new[hf] = torch.from_numpy(np.ascontiguousarray(arr.copy()))
    model.load_state_dict(new, strict=False)
    model.eval()
    return model


def test_llama3_rope_matches_transformers():
    q = _load_example()
    import tgai_go.train as T
    T.set_default_device("cpu")

    w = q.make_tiny_weights(dict(CFG), seed=11)
    ids = np.random.default_rng(5).integers(0, CFG["V"], size=(1, 10)).astype(np.int64)

    hf = _build_hf(w, CFG)
    with torch.no_grad():
        ref = hf(torch.from_numpy(ids)).logits.float().numpy()

    ours_m = q.Qwen25(dict(CFG), dict(w))
    got = np.asarray(ours_m.forward(ids).to_numpy(), dtype=np.float32)

    scale = max(float(np.abs(ref).max()), 1e-12)
    rel = float(np.abs(got - ref).max()) / scale
    assert rel < 1e-5, "llama3 rope 缩放没对齐（rel=%.2e）—— 检查分段缩放公式" % rel


def test_linear_rope_matches_transformers():
    q = _load_example()
    import tgai_go.train as T
    T.set_default_device("cpu")

    cfg = dict(CFG)
    cfg["ROPE_SCALING"] = {"rope_type": "linear", "factor": 4.0}
    w = q.make_tiny_weights(dict(cfg), seed=12)
    ids = np.random.default_rng(6).integers(0, cfg["V"], size=(1, 10)).astype(np.int64)

    hf = _build_hf(w, cfg)
    with torch.no_grad():
        ref = hf(torch.from_numpy(ids)).logits.float().numpy()
    ours_m = q.Qwen25(dict(cfg), dict(w))
    got = np.asarray(ours_m.forward(ids).to_numpy(), dtype=np.float32)
    rel = float(np.abs(got - ref).max()) / max(float(np.abs(ref).max()), 1e-12)
    assert rel < 1e-5, "linear rope 缩放没对齐（rel=%.2e）" % rel


def test_no_scaling_still_matches_standard_rope():
    q = _load_example()
    a = q.rope_inv_freq(16, 10000.0, None)
    b = q.rope_inv_freq(16, 10000.0, {"rope_type": "default"})
    assert np.allclose(a, b)


def test_unsupported_rope_type_raises_with_fix():
    q = _load_example()
    for stype in ("yarn", "longrope", "dynamic", "mrope"):
        with pytest.raises(ValueError) as ei:
            q.rope_inv_freq(16, 10000.0, {"rope_type": stype, "factor": 4.0})
        msg = str(ei.value)
        assert stype in msg and "修法" in msg, msg
        assert "静默算错" in msg, "错误信息要说明为什么不能忽略"
