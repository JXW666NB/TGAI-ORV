# -*- coding: utf-8 -*-
"""检查点读写测试：GGUF / safetensors 名字映射 / npz 往返。

设计约束：**不需要外部模型文件**（CI 里没有），用 gguf.GGUFWriter 现场造一个迷你 GGUF。
"""
import numpy as np
import pytest

from tgai_go.train import checkpoints as C

gguf = pytest.importorskip("gguf")


# ---------------------------------------------------------------------------
# 名字映射
# ---------------------------------------------------------------------------

def test_map_gguf_name_llama():
    assert C.map_gguf_name("token_embd.weight") == "token_embedding.weight"
    assert C.map_gguf_name("output_norm.weight") == "final_norm.weight"
    assert C.map_gguf_name("blk.3.attn_q.weight") == "blocks.3.attn.q_proj.weight"
    assert C.map_gguf_name("blk.0.ffn_gate.weight") == "blocks.0.ffn.w1.weight"
    assert C.map_gguf_name("blk.0.attn_q.bias") == "blocks.0.attn.q_proj.bias"
    assert C.map_gguf_name("blk.0.unknown_thing.weight") is None
    assert C.map_gguf_name("nonsense") is None


def test_map_gguf_name_qwen3_has_qk_norm():
    """Qwen3 的 per-head RMSNorm 必须映射，否则静默丢权重 → 输出全错。"""
    assert C.map_gguf_name("blk.1.attn_q_norm.weight") == "blocks.1.attn.q_norm.weight"
    assert C.map_gguf_name("blk.1.attn_k_norm.weight") == "blocks.1.attn.k_norm.weight"


def test_hf_name_to_ours():
    cases = {
        "model.embed_tokens.weight": "token_embedding.weight",
        "lm_head.weight": "lm_head.weight",
        "model.norm.weight": "final_norm.weight",
        "model.layers.0.input_layernorm.weight": "blocks.0.ln1.weight",
        "model.layers.2.self_attn.q_proj.bias": "blocks.2.attn.q_proj.bias",
        "model.layers.2.self_attn.q_norm.weight": "blocks.2.attn.q_norm.weight",
        "model.layers.1.mlp.down_proj.weight": "blocks.1.ffn.w3.weight",
    }
    for hf, ours in cases.items():
        assert C.hf_name_to_ours(hf) == ours, hf
    assert C.hf_name_to_ours("model.rotary_emb.inv_freq") is None


# ---------------------------------------------------------------------------
# 造一个迷你 GGUF 并读回来
# ---------------------------------------------------------------------------

def _write_tiny_gguf(path, arch="llama", n_layer=1, D=8, NH=2, KV=1, FF=16, V=32,
                     tie=True, drop=()):
    w = gguf.GGUFWriter(path, arch)
    w.add_uint32("%s.block_count" % arch, n_layer)
    w.add_uint32("%s.embedding_length" % arch, D)
    w.add_uint32("%s.attention.head_count" % arch, NH)
    w.add_uint32("%s.attention.head_count_kv" % arch, KV)
    w.add_uint32("%s.feed_forward_length" % arch, FF)
    w.add_float32("%s.attention.layer_norm_rms_epsilon" % arch, 1e-6)
    w.add_float32("%s.rope.freq_base" % arch, 1e6)

    def add(name, arr):
        if name in drop:
            return
        w.add_tensor(name, np.asarray(arr, dtype=np.float32),
                     raw_dtype=gguf.GGMLQuantizationType.F32)

    add("token_embd.weight", np.random.rand(V, D))
    add("output_norm.weight", np.ones(D))
    if not tie:
        add("output.weight", np.random.rand(V, D))
    DH = D // NH
    for i in range(n_layer):
        add("blk.%d.attn_norm.weight" % i, np.ones(D))
        add("blk.%d.ffn_norm.weight" % i, np.ones(D))
        add("blk.%d.attn_q.weight" % i, np.random.rand(NH * DH, D))
        add("blk.%d.attn_k.weight" % i, np.random.rand(KV * DH, D))
        add("blk.%d.attn_v.weight" % i, np.random.rand(KV * DH, D))
        add("blk.%d.attn_output.weight" % i, np.random.rand(D, NH * DH))
        add("blk.%d.ffn_gate.weight" % i, np.random.rand(FF, D))
        add("blk.%d.ffn_up.weight" % i, np.random.rand(FF, D))
        add("blk.%d.ffn_down.weight" % i, np.random.rand(D, FF))
        if arch == "qwen3":
            add("blk.%d.attn_q_norm.weight" % i, np.ones(DH))
            add("blk.%d.attn_k_norm.weight" % i, np.ones(DH))
    # 这个版本的 gguf 需要三步：文件头 → KV → 张量（写入顺序有状态机校验）
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    return path


def test_load_gguf_roundtrip(tmp_path):
    p = str(tmp_path / "tiny.gguf")
    _write_tiny_gguf(p, arch="llama")
    ck = C.load_gguf(p)

    assert ck.config["arch"] == "llama"
    assert ck.config["n_layer"] == 1
    assert ck.config["n_embd"] == 8
    assert ck.config["n_head"] == 2
    assert ck.config["n_head_kv"] == 1
    assert ck.config["vocab_size"] == 32
    assert ck.config["tie_embeddings"] is True
    assert abs(ck.config["rope_theta"] - 1e6) < 1.0

    assert "token_embedding.weight" in ck.weights
    assert ck.weights["token_embedding.weight"].shape == (32, 8)
    assert ck.weights["blocks.0.ffn.w3.weight"].shape == (8, 16)
    assert ck.weights["token_embedding.weight"].dtype == np.float32
    # 绑定权重时没有独立 lm_head
    assert "lm_head.weight" not in ck.weights


def test_load_gguf_untied_keeps_lm_head(tmp_path):
    p = str(tmp_path / "untied.gguf")
    _write_tiny_gguf(p, arch="llama", tie=False)
    ck = C.load_gguf(p)
    assert ck.config["tie_embeddings"] is False
    assert "lm_head.weight" in ck.weights


def test_load_gguf_qwen3_keeps_qk_norm(tmp_path):
    p = str(tmp_path / "q3.gguf")
    _write_tiny_gguf(p, arch="qwen3")
    ck = C.load_gguf(p)
    assert ck.config["arch"] == "qwen3"
    assert "blocks.0.attn.q_norm.weight" in ck.weights
    assert "blocks.0.attn.k_norm.weight" in ck.weights


def test_load_gguf_missing_layer_weight_errors(tmp_path):
    """缺 ffn_down 必须报错并列出缺的是哪个，而不是静默继续。"""
    p = str(tmp_path / "broken.gguf")
    _write_tiny_gguf(p, arch="llama", drop=("blk.0.ffn_down.weight",))
    with pytest.raises(C.CheckpointError) as ei:
        C.load_gguf(p)
    msg = str(ei.value)
    assert "ffn_down" in msg and "修法" in msg


def test_unsupported_arch_errors(tmp_path):
    p = str(tmp_path / "weird.gguf")
    _write_tiny_gguf(p, arch="gemma9")
    with pytest.raises(C.CheckpointError) as ei:
        C.load_gguf(p)
    msg = str(ei.value)
    assert "gemma9" in msg and "修法" in msg and "qwen2" in msg


def test_missing_file_errors():
    with pytest.raises(C.CheckpointError) as ei:
        C.load_gguf("D:/definitely/not/here.gguf")
    assert "修法" in str(ei.value)


def test_quantized_tensor_errors_clearly():
    """量化张量要给出「怎么转成 F16」的修法，而不是抛看不懂的异常。"""

    class _FakeType:
        name = "Q8_0"

    class _FakeTensor:
        tensor_type = _FakeType()
        data = b""

    with pytest.raises(C.CheckpointError) as ei:
        C._tensor_to_numpy(_FakeTensor(), "blk.0.attn_q.weight")
    msg = str(ei.value)
    assert "Q8_0" in msg and "F16" in msg and "修法" in msg


# ---------------------------------------------------------------------------
# npz 往返
# ---------------------------------------------------------------------------

def test_npz_roundtrip(tmp_path):
    p = str(tmp_path / "ck.npz")
    w = {"token_embedding.weight": np.random.rand(8, 4).astype(np.float32),
         "final_norm.weight": np.ones(4, np.float32)}
    cfg = {"arch": "qwen2", "n_layer": 0, "n_embd": 4, "vocab_size": 8}
    C.save_npz(w, p, config=cfg)
    ck = C.load_npz(p)
    assert ck.config["arch"] == "qwen2"
    assert set(ck.weights) == set(w)
    for k in w:
        assert np.allclose(ck.weights[k], w[k])


def test_npz_missing_file_errors():
    with pytest.raises(C.CheckpointError) as ei:
        C.load_npz("D:/nope/ck.npz")
    assert "修法" in str(ei.value)
