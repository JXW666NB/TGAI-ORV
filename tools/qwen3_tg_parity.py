# -*- coding: utf-8 -*-
"""Qwen3 真权重对拍（CPU）：`.TG` 切片 vs 官方 `Qwen3ForCausalLM`。

为什么用"切片"：完整 Qwen3-0.6B 在 CPU 上跑 28 层太慢，而**前 2 层 + 词表前 1024 行**
足以暴露所有结构性错误（head_dim、q_norm 位置、tie、RoPE 布局），且秒级完成。

为什么用官方 HF 类：它由**别人**实现。我们自己的 torch 孪生只能验证"两边写法一致"，
验证不了"我们对 Qwen3 的理解对不对"。

判据：max|Δ| / max|ref| ≤ 1e-5。
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for cand in (os.path.join(REPO, "python"), REPO):
    if os.path.isdir(os.path.join(cand, "tgai_go")) and cand not in sys.path:
        sys.path.insert(0, cand)

from tgai_go.train import checkpoints as C  # noqa: E402


def load_example():
    import importlib.util
    path = os.path.join(REPO, "examples", "qwen25_sft.py")
    spec = importlib.util.spec_from_file_location("qwen25_sft", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen25_sft"] = mod
    spec.loader.exec_module(mod)
    return mod


OURS_TO_HF = [
    ("token_embedding.weight", "model.embed_tokens.weight"),
    ("lm_head.weight", "lm_head.weight"),
    ("final_norm.weight", "model.norm.weight"),
]
OURS_TO_HF_LAYER = [
    ("ln1.weight", "input_layernorm.weight"),
    ("ln2.weight", "post_attention_layernorm.weight"),
    ("attn.q_proj.weight", "self_attn.q_proj.weight"),
    ("attn.k_proj.weight", "self_attn.k_proj.weight"),
    ("attn.v_proj.weight", "self_attn.v_proj.weight"),
    ("attn.out_proj.weight", "self_attn.o_proj.weight"),
    ("attn.q_norm.weight", "self_attn.q_norm.weight"),
    ("attn.k_norm.weight", "self_attn.k_norm.weight"),
    ("ffn.w1.weight", "mlp.gate_proj.weight"),
    ("ffn.w2.weight", "mlp.up_proj.weight"),
    ("ffn.w3.weight", "mlp.down_proj.weight"),
]


def build_hf(model_cls, cfg_cls, ck):
    """用我们的权重造一个官方 Qwen3 模型（同名结构，只有层数和词表是切片的）。"""
    import torch
    c = ck.config
    cfg = cfg_cls(vocab_size=c["vocab_size"], hidden_size=c["n_embd"],
                  num_hidden_layers=c["n_layer"], num_attention_heads=c["n_head"],
                  num_key_value_heads=c["n_head_kv"], head_dim=c["head_dim"],
                  intermediate_size=c["n_ff"], rope_theta=c["rope_theta"],
                  rms_norm_eps=c["eps"], tie_word_embeddings=c["tie_embeddings"])
    model = model_cls(cfg)
    sd = model.state_dict()
    mapping = dict(OURS_TO_HF)
    for k in ck.weights:
        if k.startswith("blocks."):
            _, idx, rest = k.split(".", 2)
            for ours, hf in OURS_TO_HF_LAYER:
                if rest == ours:
                    mapping[k] = "model.layers.%s.%s" % (idx, hf)
                    break
    loaded, skipped = 0, []
    new_sd = {}
    for ours, arr in ck.weights.items():
        hf = mapping.get(ours)
        if hf is None or hf not in sd:
            skipped.append(ours)
            continue
        if tuple(sd[hf].shape) != tuple(arr.shape):
            skipped.append("%s(形状 %s vs %s)" % (ours, arr.shape, tuple(sd[hf].shape)))
            continue
        new_sd[hf] = torch.from_numpy(arr.copy())
        loaded += 1
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    return model, loaded, skipped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tg", default=r"D:\DSH-Workspace\_gguf_test\Qwen3-fix.TG")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args(argv)

    try:
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
    except ImportError as ex:
        print("需要 torch + transformers>=4.51（Qwen3）：%s" % (ex,))
        return 2

    q = load_example()
    ck = C.load_tg(a.tg, max_layers=a.layers, vocab_limit=a.vocab)
    print("切片检查点：%s" % (ck,))

    model, loaded, skipped = build_hf(Qwen3ForCausalLM, Qwen3Config, ck)
    print("灌入 HF 的参数 %d 个%s"
          % (loaded, ("；跳过 %d 个（前 3：%s）" % (len(skipped), skipped[:3]))
             if skipped else ""))

    rng = np.random.default_rng(0)
    ids = rng.integers(0, a.vocab, size=(1, 8)).astype(np.int64)

    with torch.no_grad():
        ref = model(torch.from_numpy(ids)).logits.float().numpy()

    import tgai_go.train as T
    T.set_default_device("cpu")
    ours_model = q.Qwen25(q.cfg_from_checkpoint(ck), dict(ck.weights))
    got = np.asarray(ours_model.forward(ids).to_numpy(), dtype=np.float32)

    d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    rel = float(d.max()) / (float(np.abs(ref).max()) or 1.0)
    print("")
    print("参考 [0,-1,:3] =", np.round(ref[0, -1, :3], 6))
    print("我们 [0,-1,:3] =", np.round(got[0, -1, :3], 6))
    print("max|Δ| = %.3e   相对误差 = %.3e   判据 ≤%.0e：%s"
          % (float(d.max()), rel, a.tol, "PASS" if rel <= a.tol else "FAIL"))
    return 0 if rel <= a.tol else 1


if __name__ == "__main__":
    raise SystemExit(main())
