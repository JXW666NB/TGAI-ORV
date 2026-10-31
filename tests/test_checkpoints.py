# -*- coding: utf-8 -*-
"""检查点读写测试：GGUF / safetensors 名字映射 / npz 往返。

设计约束：**不需要外部模型文件**（CI 里没有），用 gguf.GGUFWriter 现场造一个迷你 GGUF。
"""
import os

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
                     tie=True, drop=(), equal_head=False):
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

    emb = np.random.rand(V, D).astype(np.float32)
    add("token_embd.weight", emb)
    add("output_norm.weight", np.ones(D))
    if not tie:
        # equal_head=True：写出与词嵌入**逐位相同**的 output（模拟 Qwen2.5 GGUF 的真实情况）
        add("output.weight", emb if equal_head else np.random.rand(V, D))
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


# ---------------------------------------------------------------------------
# config.json 语义（Qwen3 的 head_dim / tie 只能从这里读）
# ---------------------------------------------------------------------------

def _write_hf_dir(dirpath, D=96, NH=4, KV=2, DH=32, L=1, V=128, tie=True,
                  lm_head_equal=True, FF=160):
    """造一个迷你 HF 权重目录（config.json + model.safetensors）。"""
    from safetensors.numpy import save_file
    rng = np.random.default_rng(0)
    w = {"model.embed_tokens.weight": rng.normal(0, 0.02, (V, D)).astype(np.float32),
         "model.norm.weight": np.ones(D, np.float32)}
    if tie:
        head = (w["model.embed_tokens.weight"] if lm_head_equal
                else rng.normal(0, 0.02, (V, D)).astype(np.float32))
        w["lm_head.weight"] = head.astype(np.float32)
    else:
        w["lm_head.weight"] = rng.normal(0, 0.02, (V, D)).astype(np.float32)
    for i in range(L):
        b = "model.layers.%d." % i
        w[b + "input_layernorm.weight"] = np.ones(D, np.float32)
        w[b + "post_attention_layernorm.weight"] = np.ones(D, np.float32)
        w[b + "self_attn.q_proj.weight"] = rng.normal(0, 0.02, (NH * DH, D)).astype(np.float32)
        w[b + "self_attn.k_proj.weight"] = rng.normal(0, 0.02, (KV * DH, D)).astype(np.float32)
        w[b + "self_attn.v_proj.weight"] = rng.normal(0, 0.02, (KV * DH, D)).astype(np.float32)
        w[b + "self_attn.o_proj.weight"] = rng.normal(0, 0.02, (D, NH * DH)).astype(np.float32)
        w[b + "mlp.gate_proj.weight"] = rng.normal(0, 0.02, (FF, D)).astype(np.float32)
        w[b + "mlp.up_proj.weight"] = rng.normal(0, 0.02, (FF, D)).astype(np.float32)
        w[b + "mlp.down_proj.weight"] = rng.normal(0, 0.02, (D, FF)).astype(np.float32)
        if DH != D // NH:                      # Qwen3 特征：per-head RMSNorm
            w[b + "self_attn.q_norm.weight"] = np.ones(DH, np.float32)
            w[b + "self_attn.k_norm.weight"] = np.ones(DH, np.float32)
    save_file(w, str(dirpath / "model.safetensors"))
    import json
    cfg = {"hidden_size": D, "num_attention_heads": NH, "num_key_value_heads": KV,
           "num_hidden_layers": L, "vocab_size": V, "intermediate_size": FF,
           "rope_theta": 1e6, "rms_norm_eps": 1e-6,
           "tie_word_embeddings": tie}
    if DH != D // NH:
        cfg["head_dim"] = DH
    (dirpath / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return dirpath


def test_safetensors_reads_head_dim_from_config(tmp_path):
    """Qwen3-0.6B 的 head_dim=128 而 hidden/heads=64：只猜形状会算错。"""
    d = _write_hf_dir(tmp_path, D=96, NH=4, DH=32)
    ck = C.load_safetensors_dir(str(d))
    assert ck.config["head_dim"] == 32
    assert ck.config["head_dim"] != ck.config["n_embd"] // ck.config["n_head"]
    assert "blocks.0.attn.q_norm.weight" in ck.weights
    assert ck.config["arch"] == "qwen3"


def test_safetensors_tie_drops_lm_head(tmp_path):
    """官方 Qwen3 safetensors 带 lm_head，但 tie=true ⇒ 必须共享而不是两份。"""
    d = _write_hf_dir(tmp_path, tie=True, lm_head_equal=True)
    ck = C.load_safetensors_dir(str(d))
    assert ck.config["tie_embeddings"] is True
    assert "lm_head.weight" not in ck.weights


def test_safetensors_tie_mismatch_raises(tmp_path):
    d = _write_hf_dir(tmp_path, tie=True, lm_head_equal=False)
    with pytest.raises(C.CheckpointError) as ei:
        C.load_safetensors_dir(str(d))
    assert "tie" in str(ei.value) and "修法" in str(ei.value)


def test_safetensors_untied_keeps_lm_head(tmp_path):
    d = _write_hf_dir(tmp_path, tie=False)
    ck = C.load_safetensors_dir(str(d))
    assert ck.config["tie_embeddings"] is False
    assert "lm_head.weight" in ck.weights


def test_gguf_tied_equal_output_is_dropped(tmp_path):
    """Qwen2.5 的 GGUF：output 与 token_embd 逐位相同 ⇒ 应判 tie 并丢掉 lm_head。"""
    p = str(tmp_path / "tied.gguf")
    _write_tiny_gguf(p, arch="qwen2", tie=False, equal_head=True)
    ck = C.load_gguf(p)
    assert ck.config["tie_embeddings"] is True
    assert "lm_head.weight" not in ck.weights


def test_gguf_untied_keeps_lm_head(tmp_path):
    p = str(tmp_path / "untied2.gguf")
    _write_tiny_gguf(p, arch="qwen2", tie=False, equal_head=False)
    ck = C.load_gguf(p)
    assert ck.config["tie_embeddings"] is False
    assert "lm_head.weight" in ck.weights


# ---------------------------------------------------------------------------
# .TG 交错布局的反置换
# ---------------------------------------------------------------------------

def test_tg_depermute_inverts_converter_permutation():
    """`.TG` 的通道被转换器按交错 RoPE 置换过：`tg = hf[:, perm]`。

    反置换要用 `argsort(perm)`（**不是对合**，perm 应用两次不等于恒等 —— 我第一版
    测试就错设了这个前提）。语义验证方式：造一个 HF 布局的张量，按转换器的方式置换成
    .TG 布局，再反置换，必须回到原样。
    """
    dh, nh = 8, 2
    perm = C._tg_perm(nh, dh)
    hf = np.arange(nh * dh, dtype=np.float32).reshape(nh, dh)

    tg = hf[:, perm]                          # 转换器做的事
    back = C._tg_depermute(tg.reshape(-1), nh, dh).reshape(nh, dh)
    assert np.allclose(back, hf), (back, hf)
    assert not np.allclose(tg, hf)             # 置换确实改变了顺序


def test_tg_depermute_2d_inverts_converter_permutation():
    dh, nh, cols = 4, 2, 3
    perm = C._tg_perm(nh, dh)
    hf = np.arange(nh * dh * cols, dtype=np.float32).reshape(nh, dh, cols)
    tg = hf[:, perm, :]
    back = C._tg_depermute(tg.reshape(nh * dh, cols), nh, dh).reshape(nh, dh, cols)
    assert np.allclose(back, hf)


# ---------------------------------------------------------------------------
# cfg 适配（两套命名：checkpoints 用 HF 风格，模型类用 D/L/V）
# ---------------------------------------------------------------------------

def test_cfg_from_checkpoint_maps_keys(tmp_path):
    """模型类要的键（D/L/V/...）必须从 Checkpoint 的 HF 风格键正确映射。

    这条防的是"main() 里 cfg 写死成 Qwen2.5 的 896，拿 Qwen3（1024）就 reshape 崩"
    —— 实测崩溃信息：reshape (1,127,1024) -> (1,127,896)。
    """
    import importlib.util
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "python"))
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "examples", "qwen25_sft.py")
    spec = importlib.util.spec_from_file_location("qwen25_sft_cfg", path)
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)

    ck = C.load_safetensors_dir(str(_write_hf_dir(tmp_path, D=96, NH=4, DH=32, L=1, V=128)))
    cfg = q.cfg_from_checkpoint(ck)
    assert cfg["D"] == 96 and cfg["L"] == 1 and cfg["V"] == 128
    assert cfg["NH"] == 4 and cfg["KV"] == 2
    assert cfg["DH"] == 32                      # ★ head_dim 来自 config.json
    assert cfg["DH"] != cfg["D"] // cfg["NH"]   # 32 != 24
    assert cfg["QK_NORM"] is True
    assert cfg["TIE"] is True
