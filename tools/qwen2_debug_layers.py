# -*- coding: utf-8 -*-
"""分段定位：逐层对比 Qwen2.5 的中间激活，找出第一处发散点。

对 seq=1 跑 HF 与我们的实现，在每一层打印 max|Δ| / 相对误差。
"""
import importlib.util
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for cand in (os.path.join(REPO, "python"), REPO):
    if os.path.isdir(os.path.join(cand, "tgai_go")) and cand not in sys.path:
        sys.path.insert(0, cand)

import tgai_go.train as T  # noqa: E402
from tgai_go.train import functional as F  # noqa: E402

GGUF = r"D:\DSH-Workspace\_gguf_test\qwen2.5-0.5b-fp16.gguf"


def load_conv():
    spec = importlib.util.spec_from_file_location(
        "gguf_to_tg", os.path.join(REPO, "tools", "gguf_to_tg.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gguf_to_tg"] = mod
    spec.loader.exec_module(mod)
    return mod


def report(tag, ours, ref):
    a = np.asarray(ours, dtype=np.float64).reshape(-1)
    b = np.asarray(ref, dtype=np.float64).reshape(-1)
    d = float(np.abs(a - b).max())
    rel = d / (float(np.abs(b).max()) or 1.0)
    print("  %-30s max|Δ|=%.4e rel=%.3e  ours[:3]=%s ref[:3]=%s"
          % (tag, d, rel, np.round(a[:3], 5), np.round(b[:3], 5)))
    return rel


def main():
    T.set_default_device("cpu")
    conv = load_conv()
    cfg = conv.ARCH_CONFIG["qwen2"]
    import gguf
    rd = gguf.GGUFReader(GGUF)
    w = {}
    for t in rd.tensors:
        nm = conv.map_weight_name(t.name, cfg)
        if nm:
            w[nm] = np.asarray(t.data, dtype=np.float32)
    head = next(np.asarray(t.data, dtype=np.float32) for t in rd.tensors
                if t.name == "output.weight")

    ids = [151643]
    D, H, KV, DH, NL, FF, EPS = 896, 14, 2, 64, 24, 4864, 1e-6

    # ---------- 参考：HF ----------
    from transformers import Qwen2Config, Qwen2ForCausalLM
    c = Qwen2Config(vocab_size=151936, hidden_size=D, intermediate_size=FF,
                    num_hidden_layers=NL, num_attention_heads=H, num_key_value_heads=KV,
                    max_position_embeddings=8192, rms_norm_eps=EPS,
                    tie_word_embeddings=False, hidden_act="silu")
    c.rope_parameters = {"rope_theta": 1e6, "rope_type": "default"}
    c.rope_scaling = dict(c.rope_parameters)
    ref_model = Qwen2ForCausalLM(c)

    def put(hf, key, transpose=False):
        a = w[key].T if transpose else w[key]
        ref_model.state_dict()[hf].copy_(torch.from_numpy(np.ascontiguousarray(a)))

    put("model.embed_tokens.weight", "token_embedding.weight")
    put("lm_head.weight", "output.weight", transpose=True)
    put("model.norm.weight", "final_norm.weight")
    for i in range(NL):
        p, h = "blocks.%d." % i, "model.layers.%d." % i
        put(h + "input_layernorm.weight", p + "ln1.weight")
        put(h + "post_attention_layernorm.weight", p + "ln2.weight")
        for gg, hf in (("attn.q_proj", "self_attn.q_proj"), ("attn.k_proj", "self_attn.k_proj"),
                       ("attn.v_proj", "self_attn.v_proj"), ("attn.out_proj", "self_attn.o_proj")):
            # HF Linear 是 [out,in]；我们 .TG 侧朝向不统一，统一按 out 维判断
            W = w[p + gg + ".weight"]
            out_dim = {"q_proj": H * DH, "k_proj": KV * DH, "v_proj": KV * DH, "o_proj": D}[gg.split(".")[-1]]
            put(h + hf + ".weight", p + gg + ".weight", transpose=(W.shape[0] != out_dim))
            bias = p + gg + ".bias"
            if bias in w:
                ref_model.state_dict()[h + hf + ".bias"].copy_(torch.from_numpy(w[bias]))
        for gg, hf in (("ffn.w1", "mlp.gate_proj"), ("ffn.w2", "mlp.up_proj"),
                       ("ffn.w3", "mlp.down_proj")):
            W = w[p + gg + ".weight"]
            out_dim = FF if gg != "ffn.w3" else D
            put(h + hf + ".weight", p + gg + ".weight", transpose=(W.shape[0] != out_dim))
    ref_model.eval()

    acts = {}

    def hook(name):
        def fn(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            acts[name] = o.detach().float().numpy()
        return fn

    ref_model.model.embed_tokens.register_forward_hook(hook("emb"))
    ref_model.model.layers[0].input_layernorm.register_forward_hook(hook("ln1"))
    ref_model.model.layers[0].self_attn.register_forward_hook(hook("attn"))
    ref_model.model.layers[0].post_attention_layernorm.register_forward_hook(hook("ln2"))
    ref_model.model.layers[0].mlp.register_forward_hook(hook("mlp"))
    ref_model.model.norm.register_forward_hook(hook("final_norm"))
    with torch.no_grad():
        out = ref_model(torch.tensor([ids], dtype=torch.long))
    ref_logits = out.logits.float().numpy()

    # ---------- 我们的实现 ----------
    print("== 逐层对比（seq=1）==")
    emb = w["token_embedding.weight"][np.array(ids, dtype=np.int64)]
    report("embedding", emb, acts["emb"])

    x = T.tensor(np.ascontiguousarray(emb.astype(np.float32)))

    def rmsnorm(xx, ww, eps):
        s = (xx * xx).mean(-1, keepdim=True)
        d = (s + eps).sqrt()
        return (xx / d) * ww

    def proj(xx, name, out_features):
        W = T.tensor(np.ascontiguousarray(w[name]))
        if W.shape[1] != out_features:
            W = W.T
        return xx @ W

    h = rmsnorm(x, T.tensor(np.ascontiguousarray(w["blocks.0.ln1.weight"])), EPS)
    report("layer0.ln1", h.to_numpy(), acts["ln1"])

    seq = 1
    q = proj(h, "blocks.0.attn.q_proj.weight", H * DH) + T.tensor(w["blocks.0.attn.q_proj.bias"])
    k = proj(h, "blocks.0.attn.k_proj.weight", KV * DH) + T.tensor(w["blocks.0.attn.k_proj.bias"])
    v = proj(h, "blocks.0.attn.v_proj.weight", KV * DH) + T.tensor(w["blocks.0.attn.v_proj.bias"])
    q = q.reshape(seq, H, DH).permute(1, 0, 2)
    k = k.reshape(seq, KV, DH).permute(1, 0, 2)
    v = v.reshape(seq, KV, DH).permute(1, 0, 2)
    k = k.repeat_interleave(H // KV, dim=0)
    v = v.repeat_interleave(H // KV, dim=0)
    s = (q @ k.swap_last2()) * (DH ** -0.5)
    o = F.softmax(s, -1) @ v
    o = o.permute(1, 0, 2).reshape(seq, H * DH)
    attn_out = proj(o, "blocks.0.attn.out_proj.weight", D)
    report("layer0.attn(seq1)", attn_out.to_numpy(), acts["attn"])

    x2 = x + attn_out
    h2 = rmsnorm(x2, T.tensor(np.ascontiguousarray(w["blocks.0.ln2.weight"])), EPS)
    report("layer0.ln2", h2.to_numpy(), acts["ln2"])

    g = proj(h2, "blocks.0.ffn.w1.weight", FF)
    u = proj(h2, "blocks.0.ffn.w2.weight", FF)
    act = F.silu(g) * u
    mlp_out = proj(act, "blocks.0.ffn.w3.weight", D)
    report("layer0.mlp", mlp_out.to_numpy(), acts["mlp"])

    # 直接对比 silu 本身（用一个已知输入）
    z = np.array([[-3.0, -1.0, 0.0, 1.0, 3.0]], dtype=np.float32)
    ours_silu = F.silu(T.tensor(z)).to_numpy()
    ref_silu = torch.nn.functional.silu(torch.from_numpy(z)).numpy()
    report("silu(z)", ours_silu, ref_silu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
