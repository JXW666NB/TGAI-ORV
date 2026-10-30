# -*- coding: utf-8 -*-
"""GGUF→.TG 转换器的命名映射测试（纯函数，不需要模型文件）。

这条映射是「转换后引擎能不能按名字取到权重」的唯一依据：
  `.TG` 权重表存的是 FNV-1a 哈希，取回时也用同一套名字算哈希 ⇒ 映射错一个字符就取不到。
（2026-10-30 我在写校验脚本时第一版就是自己另写了一套映射，结果 291 个张量全部取不到。）
"""
import importlib.util
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def conv():
    path = os.path.join(REPO, "tools", "gguf_to_tg.py")
    spec = importlib.util.spec_from_file_location("gguf_to_tg", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gguf_to_tg"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("arch", ["llama", "qwen2", "qwen3", "mistral", "gemma2", "phi3"])
def test_arch_known(conv, arch):
    assert arch in conv.ARCH_CONFIG


def test_embed_and_final_norm(conv):
    cfg = conv.ARCH_CONFIG["qwen2"]
    assert conv.map_weight_name("token_embd.weight", cfg) == "token_embedding.weight"
    assert conv.map_weight_name("output_norm.weight", cfg) == "final_norm.weight"


def test_output_weight_is_intentionally_skipped(conv):
    """output.weight 与 token_embedding 权重绑定，转换器有意不写，返回 None。"""
    cfg = conv.ARCH_CONFIG["qwen2"]
    assert conv.map_weight_name("output.weight", cfg) is None


def test_block_layer_names(conv):
    cfg = conv.ARCH_CONFIG["qwen2"]
    cases = {
        "blk.0.attn_q.weight": "blocks.0.attn.q_proj.weight",
        "blk.7.attn_k.weight": "blocks.7.attn.k_proj.weight",
        "blk.23.attn_v.weight": "blocks.23.attn.v_proj.weight",
        "blk.3.attn_output.weight": "blocks.3.attn.out_proj.weight",
        "blk.1.ffn_gate.weight": "blocks.1.ffn.w1.weight",
        "blk.1.ffn_up.weight": "blocks.1.ffn.w2.weight",
        "blk.1.ffn_down.weight": "blocks.1.ffn.w3.weight",
        "blk.2.attn_norm.weight": "blocks.2.ln1.weight",
        "blk.2.ffn_norm.weight": "blocks.2.ln2.weight",
    }
    for gguf_name, want in cases.items():
        got = conv.map_weight_name(gguf_name, cfg)
        assert got == want, "%s -> %s（期望 %s）" % (gguf_name, got, want)


def test_qkv_bias_mapping(conv):
    """Qwen2.5 有 qkv bias，必须能正确映射（这是它与 LLaMA 的关键差异之一）。"""
    cfg = conv.ARCH_CONFIG["qwen2"]
    for i in (0, 23):
        for part, proj in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")):
            got = conv.map_weight_name("blk.%d.attn_%s.bias" % (i, part), cfg)
            assert got == "blocks.%d.attn.%s.bias" % (i, proj), got


def test_hash_is_stable(conv):
    """.TG 表里存的是这个哈希；同一名字必须永远算出同一个值。"""
    h1 = conv.fnv1a_64("blocks.0.attn.q_proj.weight")
    h2 = conv.fnv1a_64("blocks.0.attn.q_proj.weight")
    assert h1 == h2 and h1 != conv.fnv1a_64("blocks.0.attn.k_proj.weight")
    # FNV-1a 64 的已知向量
    assert conv.fnv1a_64("") == 0xcbf29ce484222325
