# -*- coding: utf-8 -*-
"""Qwen2.5 在 TGAI ORV 上微调（真实开源权重，引擎原生实现）。

## 阶段

1. `--forward-only`：加载 GGUF 里的真实 Qwen2.5-0.5B 权重跑前向，与 HF 参考 logits 对齐
   （判据：相对误差 ≤1e-5）。**这是能不能微调的前提。**
2. `--steps N`：在语料上做 SFT。

## 与我们的模型（exam2）的差异

| 项 | exam2（我们的模型） | Qwen2.5 |
| --- | --- | --- |
| 注意力 | MHA | **GQA**（14 Q : 2 KV） |
| qkv | 无 bias | **有 bias** |
| 词嵌入/lm_head | 绑定 | **不绑定**（独立 output.weight） |
| RoPE | 引擎交错式（fused_rope） | 标准 rotate_half（θ=1e6） |
| MLP | MoE | 稠密 SwiGLU |

RoPE 先用**组合式** rotate_half（正确优先）；融合核版本需要把 Q/K 权重按
`tools/gguf_to_tg.maybe_permute_rope` 预置换，留作性能优化。

用法：
    python examples/qwen25_sft.py --forward-only \
        --gguf D:/DSH-Workspace/_gguf_test/qwen2.5-0.5b-fp16.gguf \
        --ref tools/_qwen_ref.npz
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for cand in (os.path.join(REPO, "python"), REPO):
    if os.path.isdir(os.path.join(cand, "tgai_go")) and cand not in sys.path:
        sys.path.insert(0, cand)

import tgai_go.train as T  # noqa: E402
from tgai_go.train import functional as F  # noqa: E402

# Qwen2.5-0.5B-instruct 结构（来自 GGUF 元数据实测）
CFG = dict(V=151936, D=896, L=24, NH=14, KV=2, FF=4864, THETA=1e6, EPS=1e-6)


def load_converter():
    spec = importlib.util.spec_from_file_location(
        "gguf_to_tg", os.path.join(REPO, "tools", "gguf_to_tg.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gguf_to_tg"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_gguf_weights(path, cfg):
    """读 GGUF → {转换器命名: np.float32}。名字映射与 .TG 转换完全一致。"""
    import gguf
    conv = load_converter()
    arch = conv.ARCH_CONFIG["qwen2"]
    rd = gguf.GGUFReader(path)
    out = {}
    for t in rd.tensors:
        nm = conv.map_weight_name(t.name, arch)
        if nm is None:
            continue
        out[nm] = np.asarray(t.data, dtype=np.float32)
    # 转换器会跳过 output.weight（.TG 里靠权重绑定），但 Qwen2.5 的 lm_head 是独立的
    for t in rd.tensors:
        if t.name == "output.weight":
            out["lm_head.weight"] = np.asarray(t.data, dtype=np.float32)
    return out


def rope_tables(seq, dh, theta):
    """标准 rotate_half 的 cos/sin：[S, dh]（前半与后半相同）。"""
    inv = 1.0 / (theta ** (np.arange(0, dh, 2, dtype=np.float64) / dh))
    ang = np.outer(np.arange(seq, dtype=np.float64), inv)
    c = np.cos(ang).astype(np.float32)
    s = np.sin(ang).astype(np.float32)
    return np.concatenate([c, c], axis=-1), np.concatenate([s, s], axis=-1)


class Qwen25:
    """引擎原生 Qwen2.5。参数用 T.nn.Parameter，融合核用 T.nn.RMSNorm。"""

    def __init__(self, cfg, w):
        self.cfg = cfg
        P = T.nn.Parameter
        self.p = {k: P(T.tensor(np.ascontiguousarray(v.astype(np.float32))))
                  for k, v in w.items()}
        self.emb = T.nn.Embedding(cfg["V"], cfg["D"])
        self.emb.weight = self.p["token_embedding.weight"]
        self.ln1, self.ln2 = [], []
        for i in range(cfg["L"]):
            b = "blocks.%d." % i
            m1 = T.nn.RMSNorm(cfg["D"], eps=cfg["EPS"])
            m1.weight = self.p[b + "ln1.weight"]
            m2 = T.nn.RMSNorm(cfg["D"], eps=cfg["EPS"])
            m2.weight = self.p[b + "ln2.weight"]
            self.ln1.append(m1)
            self.ln2.append(m2)
        self.final = T.nn.RMSNorm(cfg["D"], eps=cfg["EPS"])
        self.final.weight = self.p["final_norm.weight"]
        self._rope = {}

    def params(self):
        return list(self.p.values())

    def _cos_sin(self, S, DH):
        key = (S, DH)
        cs = self._rope.get(key)
        if cs is None:
            cos, sin = rope_tables(S, DH, self.cfg["THETA"])
            cs = (T.tensor(cos[None, :, :]), T.tensor(sin[None, :, :]))
            self._rope[key] = cs
        return cs

    @staticmethod
    def _rope_apply(x, cos_t, sin_t):
        dh = int(x.shape[-1])
        h = dh // 2
        rot = T.cat([-x[:, :, h:], x[:, :, :h]], dim=-1)
        return x * cos_t + rot * sin_t

    def forward(self, ids):
        """ids: numpy int64 [B, S] → logits [B, S, V]"""
        cfg = self.cfg
        D, L, NH, KV, FF = cfg["D"], cfg["L"], cfg["NH"], cfg["KV"], cfg["FF"]
        DH = D // NH
        B, S = int(ids.shape[0]), int(ids.shape[1])
        p = self.p
        cos_t, sin_t = self._cos_sin(S, DH)

        x = self.emb(T.tensor(ids) if T.__name__ == "torch" else ids).reshape(B, S, D)

        for i in range(L):
            b = "blocks.%d." % i
            h = self.ln1[i](x).reshape(B * S, D)
            q = (h @ p[b + "attn.q_proj.weight"].T + p[b + "attn.q_proj.bias"])
            k = (h @ p[b + "attn.k_proj.weight"].T + p[b + "attn.k_proj.bias"])
            v = (h @ p[b + "attn.v_proj.weight"].T + p[b + "attn.v_proj.bias"])
            q = q.reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
            k = k.reshape(B, S, KV, DH).permute(0, 2, 1, 3).reshape(B * KV, S, DH)
            v = v.reshape(B, S, KV, DH).permute(0, 2, 1, 3).reshape(B * KV, S, DH)
            q = self._rope_apply(q, cos_t, sin_t)
            k = self._rope_apply(k, cos_t, sin_t)
            # GQA：KV 头复制到 Q 头数（在 [B,KV,S,DH] 上按第 1 维展开，避免跨 batch 打乱）
            k = k.reshape(B, KV, S, DH).repeat_interleave(NH // KV, dim=1).reshape(B * NH, S, DH)
            v = v.reshape(B, KV, S, DH).repeat_interleave(NH // KV, dim=1).reshape(B * NH, S, DH)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            o = o.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
            x = x + (o @ p[b + "attn.out_proj.weight"].T).reshape(B, S, D)

            h2 = self.ln2[i](x).reshape(B * S, D)
            g = h2 @ p[b + "ffn.w1.weight"].T
            u = h2 @ p[b + "ffn.w2.weight"].T
            act = F.silu(g) * u
            x = x + (act @ p[b + "ffn.w3.weight"].T).reshape(B, S, D)

        z = self.final(x).reshape(B * S, D)
        return (z @ p["lm_head.weight"].T).reshape(B, S, cfg["V"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=r"D:\DSH-Workspace\_gguf_test\qwen2.5-0.5b-fp16.gguf")
    ap.add_argument("--ref", default=os.path.join(REPO, "tools", "_qwen_ref.npz"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--forward-only", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args(argv)

    T.set_default_device(a.device)
    print("device=%s" % (T.get_device(),))
    cfg = CFG
    print("读取 GGUF：%s" % a.gguf)
    w = load_gguf_weights(a.gguf, cfg)
    print("权重 %d 个（含独立 lm_head）" % len(w))

    model = Qwen25(cfg, w)
    print("参数已上卡：%d 个张量" % len(model.params()))
    del w

    if a.forward_only:
        ref = np.load(a.ref)
        ids, ref_logits = ref["ids"], ref["logits"]
        ids2 = np.asarray(ids, dtype=np.int64).reshape(1, -1)
        print("参考序列长度 %d" % ids2.shape[1])
        logits = model.forward(ids2)
        got = np.asarray(logits.to_numpy(), dtype=np.float32)
        d = np.abs(got.astype(np.float64) - ref_logits.astype(np.float64))
        denom = float(np.abs(ref_logits).max()) or 1.0
        rel = float(d.max()) / denom
        print("")
        print("我们 logits[0,-1,:5] =", got[0, -1, :5])
        print("参考 logits[0,-1,:5] =", ref_logits[0, -1, :5])
        print("max|Δ| = %.3e   相对误差 = %.3e" % (float(d.max()), rel))
        ok = rel <= a.tol
        print("判据 ≤%.0e：%s" % (a.tol, "PASS" if ok else "FAIL"))
        return 0 if ok else 1

    print("SFT 训练路径尚未实现（先用 --forward-only 验证前向）")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
