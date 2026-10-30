# -*- coding: utf-8 -*-
"""参考侧：从 GGUF 读真实 Qwen2.5-0.5B 权重，用 transformers 构建 Qwen2 并产出参考 logits。

产物：tools/_qwen_ref.npz（含 ids / logits / 权重名到 HF 名的映射清单）
只跑 CPU。
"""
import json
import os
import sys

import numpy as np
import torch

GGUF = r"D:\DSH-Workspace\_gguf_test\qwen2.5-0.5b-fp16.gguf"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_qwen_ref.npz")
IDS = [151643, 9707, 1879, 264, 4326, 13, 151645]   # 任意固定序列，只用于对齐


def load_gguf(path):
    """读出所有张量到 numpy（F16/F32）。GGML 形状是 [ne0, ne1...]，ne0 为最快变化维。"""
    import gguf
    rd = gguf.GGUFReader(path)
    meta = {}
    for k in rd.fields:
        try:
            v = rd.fields[k].contents()
            if isinstance(v, (bytes, bytearray)):
                v = bytes(v).decode("utf-8", "replace")
            meta[k] = v
        except Exception:  # noqa: BLE001
            pass
    tensors = {}
    for t in rd.tensors:
        arr = np.asarray(t.data)
        # t.data 可能已是 numpy；F16 直接是 float16
        if arr.dtype == np.uint8:
            # 未转换类型（如量化）——本文件是 fp16，不该出现
            raise SystemExit("张量 %s 不是 F16/F32（dtype=%s），本脚本只处理 fp16/fp32"
                             % (t.name, arr.dtype))
        tensors[t.name] = arr
    return meta, tensors


def to_hf_state(tensors):
    """GGUF(llama.cpp) 命名 → HF Qwen2 命名。

    实测：gguf 的 t.data 直接就是 [out, in]（与 HF Linear 一致），**不需要转置**。
    （t.shape 打印出来的 dims 是反的，别被骗 —— 我第一版就转错了。）
    """
    st = {}

    def put(hf_name, arr, transpose=False):
        a = np.asarray(arr, dtype=np.float32)
        if transpose and a.ndim == 2:
            a = a.T
        st[hf_name] = torch.from_numpy(np.ascontiguousarray(a))

    put("model.embed_tokens.weight", tensors["token_embd.weight"])
    put("lm_head.weight", tensors["output.weight"])
    put("model.norm.weight", tensors["output_norm.weight"], transpose=False)
    for i in range(24):
        p = "blk.%d." % i
        h = "model.layers.%d." % i
        put(h + "input_layernorm.weight", tensors[p + "attn_norm.weight"], transpose=False)
        put(h + "post_attention_layernorm.weight", tensors[p + "ffn_norm.weight"], transpose=False)
        put(h + "self_attn.q_proj.weight", tensors[p + "attn_q.weight"])
        put(h + "self_attn.k_proj.weight", tensors[p + "attn_k.weight"])
        put(h + "self_attn.v_proj.weight", tensors[p + "attn_v.weight"])
        put(h + "self_attn.o_proj.weight", tensors[p + "attn_output.weight"])
        for nm in ("q", "k", "v"):
            key = p + "attn_%s.bias" % nm
            if key in tensors:
                st[h + "self_attn.%s_proj.bias" % nm] = torch.from_numpy(
                    np.asarray(tensors[key], dtype=np.float32))
        put(h + "mlp.gate_proj.weight", tensors[p + "ffn_gate.weight"])
        put(h + "mlp.up_proj.weight", tensors[p + "ffn_up.weight"])
        put(h + "mlp.down_proj.weight", tensors[p + "ffn_down.weight"])
    return st


def main():
    print("读取 GGUF：", GGUF)
    meta, tensors = load_gguf(GGUF)
    cfg_kv = {k: v for k, v in meta.items() if str(k).startswith("qwen2.")}
    print("关键配置：", {k: cfg_kv[k] for k in cfg_kv
                        if any(s in k for s in ("block_count", "embedding_length",
                                                "head_count", "freq_base",
                                                "layer_norm_rms_epsilon", "context_length"))})
    print("张量数：", len(tensors))

    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(
        vocab_size=int(cfg_kv["qwen2.vocab_size"]) if "qwen2.vocab_size" in cfg_kv else 151936,
        hidden_size=int(cfg_kv["qwen2.embedding_length"]),
        intermediate_size=int(cfg_kv["qwen2.feed_forward_length"]) if "qwen2.feed_forward_length" in cfg_kv else 4864,
        num_hidden_layers=int(cfg_kv["qwen2.block_count"]),
        num_attention_heads=int(cfg_kv["qwen2.attention.head_count"]),
        num_key_value_heads=int(cfg_kv["qwen2.attention.head_count_kv"]),
        max_position_embeddings=int(cfg_kv["qwen2.context_length"]),
        rms_norm_eps=float(cfg_kv["qwen2.attention.layer_norm_rms_epsilon"]),
        tie_word_embeddings=False,
        hidden_act="silu",
    )
    # transformers 5.x：rope 配置在 rope_parameters 里（旧版是 rope_theta）
    theta = float(cfg_kv["qwen2.rope.freq_base"])
    cfg.rope_parameters = {"rope_theta": theta, "rope_type": "default"}
    cfg.rope_scaling = dict(cfg.rope_parameters)
    print("config：hidden=%d layers=%d heads=%d kv=%d ffn=%d theta=%g eps=%g"
          % (cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads,
             cfg.num_key_value_heads, cfg.intermediate_size, theta, cfg.rms_norm_eps))

    model = Qwen2ForCausalLM(cfg)
    st = to_hf_state(tensors)
    missing, unexpected = model.load_state_dict(st, strict=False)
    print("load_state_dict: missing=%d unexpected=%d" % (len(missing), len(unexpected)))
    if missing:
        print("  missing 前 5：", list(missing)[:5])
    if unexpected:
        print("  unexpected 前 5：", list(unexpected)[:5])
    model.eval()

    ids = torch.tensor([IDS], dtype=torch.long)
    with torch.no_grad():
        out = model(ids)
    logits = out.logits.float().numpy()
    print("参考 logits：", logits.shape, "最后一位前 5 个：", logits[0, -1, :5])
    np.savez(OUT, ids=np.array(IDS, np.int64), logits=logits)
    print("已保存", OUT)


if __name__ == "__main__":
    sys.exit(main())
