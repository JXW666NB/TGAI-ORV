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

    def __init__(self, cfg, w, tie=None):
        self.cfg = cfg
        P = T.nn.Parameter
        self.p = {k: P(T.tensor(np.ascontiguousarray(v.astype(np.float32))))
                  for k, v in w.items()}
        # Qwen2.5-0.5B/1.5B 官方是**权重绑定**的（没有独立 lm_head）：
        # 这时必须让 lm_head 与词嵌入**共享同一个 Parameter 对象**，
        # 复制一份会让两条梯度各走各的，训练后两者就分家了。
        if tie is None:
            tie = "lm_head.weight" not in self.p
        self.tie = bool(tie)
        if self.tie:
            self.p["lm_head.weight"] = self.p["token_embedding.weight"]
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


# ---------------------------------------------------------------------------
# SFT（监督微调）
# ---------------------------------------------------------------------------

def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def build_sequences(tok, data_path, seq_len, limit):
    """把 jsonl 转成 token id 序列（Qwen 的 <|im_start|> 对话模板）。"""
    import json
    seqs = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            user = rec.get("user") or rec.get("q") or rec.get("question") or rec.get("input") or ""
            asst = (rec.get("assistant") or rec.get("a") or rec.get("answer")
                    or rec.get("output") or "")
            if rec.get("prompt") and rec.get("completion"):
                # 结构化 SFT：只对 **回答段** 计 loss（提示段用 ignore_index 屏蔽）
                pid = tok.encode(rec["prompt"]).ids
                cid = tok.encode(rec["completion"]).ids
                ids = (pid + cid)[:seq_len]
                mf = min(len(pid), len(ids))
                # 回答段至少要留 2 个 token，否则整条样本 loss 全被 mask 掉（无信号）
                if len(ids) >= 8 and len(cid) >= 2 and mf <= len(ids) - 2:
                    seqs.append({"ids": ids, "mask_from": mf})
                if len(seqs) >= limit:
                    break
                continue
            if user and asst:
                # 有问答字段 ⇒ 用 Qwen 的对话模板
                text = ("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s<|im_end|>"
                        % (user, asst))
            elif isinstance(rec.get("text"), str) and rec["text"].strip():
                # 只有整段文本（如 persona.jsonl 的 <用户>/<TGAI><回答>）⇒ 直接按语言模型目标训
                text = rec["text"].strip()
            else:
                continue
            ids = tok.encode(text).ids[:seq_len]
            if len(ids) >= 8:
                # 没有分段信息时整段都算 loss（mask_from=0）
                seqs.append({"ids": ids, "mask_from": 0})
            if len(seqs) >= limit:
                break
    return seqs


def _make_xy(item, seq_len):
    """把 {"ids","mask_from"} 变成 (xb, yb)，并把提示段的 yb 置 -100。"""
    if isinstance(item, dict):
        ids, mask_from = list(item["ids"]), int(item.get("mask_from", 0))
    else:
        ids, mask_from = list(item), 0
    pad = ids[-1]
    ids = (ids + [pad] * seq_len)[:seq_len]
    xb = np.asarray([ids[:-1]], dtype=np.int64)
    yb = np.asarray([ids[1:]], dtype=np.int64)
    cut = max(0, mask_from - 1)          # 预测 ids[i+1]，回答从 mask_from 开始
    if cut:
        yb[:, :cut] = -100
    return xb, yb


def _probe_multi(model, probes, seq_len):
    """留出集探针：取**多条平均**。单条噪声太大 —— 一条样本 ±1% 的波动
    就会被误读成"学到了"或"没学到"。"""
    # 注意：单条探针本身也可能是「token id 列表」，不能只看外层是不是 list，
    # 要看元素是不是 dict/list（多条）——否则会把 token id 当样本，报 int 不可迭代。
    if (isinstance(probes, (list, tuple)) and probes
            and isinstance(probes[0], (dict, list, tuple))):
        items = list(probes)
    else:
        items = [probes]
    vals = [_probe_loss(model, it, seq_len) for it in items]
    return float(np.mean(vals))


def _probe_loss(model, ids, seq_len):
    """在固定探针序列上算 loss（不建图）：训练过程中唯一可比的指标。"""
    xb, yb = _make_xy(ids, seq_len)
    with T.no_grad():
        logits = model.forward(xb)
        loss = F.cross_entropy(logits.reshape(xb.size, model.cfg["V"]), yb.reshape(-1))
    return float(loss.to_numpy())


def sft(model, seqs, steps, lr, seq_len, log_every=1, probe=None):
    """最小可用 SFT 循环：AdamW + 交叉熵（全 token）。

    注意：训练样本轮转 ⇒ **每步 loss 不可比**（不同样本难度不同）。
    所以这里同时报一条固定 `probe` 序列上的 loss，只看它是否下降。
    """
    import time
    cfg = model.cfg
    opt = T.optim.AdamW(model.params(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    losses, probes = [], []
    t0 = time.time()
    for step in range(steps):
        xb, yb = _make_xy(seqs[step % len(seqs)], seq_len)

        if hasattr(opt, "zero_grad"):
            opt.zero_grad()
        else:
            for prm in model.params():
                prm.zero_grad()

        logits = model.forward(xb)
        loss = F.cross_entropy(logits.reshape(xb.size, cfg["V"]), yb.reshape(-1))
        loss.backward()
        opt.step()

        train_loss = float(loss.to_numpy())
        losses.append(train_loss)
        pl = _probe_multi(model, probe, seq_len) if probe is not None else None
        if pl is not None:
            probes.append(pl)
        if (step + 1) % log_every == 0:
            extra = ("| probe %.6f " % probes[-1]) if probes else ""
            print("step %3d | train %6.4f %s| %.1f s/step"
                  % (step + 1, train_loss, extra, (time.time() - t0) / (step + 1)))
    return losses, probes


# ---------------------------------------------------------------------------
# safetensors 加载（HF 官方权重目录）
# ---------------------------------------------------------------------------

_HF_PATTERNS = [
    (r"^model\.embed_tokens\.weight$", "token_embedding.weight"),
    (r"^lm_head\.weight$", "lm_head.weight"),
    (r"^model\.norm\.weight$", "final_norm.weight"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.weight$", r"blocks.\1.ln1.weight"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$", r"blocks.\1.ln2.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$", r"blocks.\1.attn.q_proj.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$", r"blocks.\1.attn.k_proj.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$", r"blocks.\1.attn.v_proj.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$", r"blocks.\1.attn.out_proj.\2"),
    (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$", r"blocks.\1.ffn.w1.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$", r"blocks.\1.ffn.w2.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$", r"blocks.\1.ffn.w3.weight"),
]


def hf_name_to_ours(name):
    """HF 参数名 → 我们脚本里的命名（与 GGUF 转换器的命名一致）。"""
    import re as _re
    for pat, rep in _HF_PATTERNS:
        if _re.match(pat, name):
            return _re.sub(pat, rep, name)
    return None


def load_safetensors_weights(dirpath, cfg):
    """读 HF safetensors 目录 → {我们的名字: np.float32}。"""
    from safetensors.numpy import load_file
    path = os.path.join(dirpath, "model.safetensors")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "找不到 %s。修法：确认 --safetensors 指向含 model.safetensors 的目录" % (path,))
    try:
        raw = load_file(path)
    except TypeError as exc:
        # 官方权重常是 bfloat16，numpy 没有 bf16 类型 ⇒ 用 torch 转 fp32；
        # 没有 torch 就明确告知修法，而不是抛一个看不懂的 TypeError。
        if "bfloat16" not in str(exc):
            raise
        try:
            from safetensors.torch import load_file as tload
        except ImportError:
            raise RuntimeError(
                "权重是 bfloat16，numpy 无法直接读取。修法：pip install torch（用它转 fp32），"
                "或先把 safetensors 转成 fp16/fp32 再加载") from exc
        raw = {k: v.float().numpy() for k, v in tload(path).items()}
    out, unknown = {}, []
    for k, v in raw.items():
        ours = hf_name_to_ours(k)
        if ours is None:
            unknown.append(k)
            continue
        out[ours] = np.asarray(v, dtype=np.float32)
    if unknown:
        print("  注意：%d 个 HF 参数未映射（前 3 个：%s）" % (len(unknown), unknown[:3]))
    need = ["token_embedding.weight", "final_norm.weight"]
    lack = [k for k in need if k not in out]
    if lack:
        raise RuntimeError(
            "缺少关键权重 %s。修法：检查 safetensors 是否为完整 Qwen2 结构" % (lack,))
    return out


# ---------------------------------------------------------------------------
# tiny 模式：随机权重的迷你 Qwen（用于 CPU/CI 验证 SFT 链路，不需要真权重）
# ---------------------------------------------------------------------------

TINY_CFG = dict(V=256, D=64, L=2, NH=4, KV=2, FF=128, THETA=1e6, EPS=1e-6)


def make_tiny_weights(cfg, seed=0):
    """随机初始化同结构权重（形状与 Qwen2.5 完全一致，只是小很多）。"""
    rng = np.random.default_rng(seed)
    D, L, NH, KV, FF, V = cfg["D"], cfg["L"], cfg["NH"], cfg["KV"], cfg["FF"], cfg["V"]
    DH = D // NH
    k = 1.0 / np.sqrt(D)
    w = {"token_embedding.weight": (rng.normal(0, 0.02, (V, D))).astype(np.float32),
         "lm_head.weight": (rng.normal(0, 0.02, (V, D))).astype(np.float32),
         "final_norm.weight": np.ones(D, np.float32)}
    for i in range(L):
        b = "blocks.%d." % i
        w[b + "ln1.weight"] = np.ones(D, np.float32)
        w[b + "ln2.weight"] = np.ones(D, np.float32)
        w[b + "attn.q_proj.weight"] = (rng.normal(0, k, (NH * DH, D))).astype(np.float32)
        w[b + "attn.k_proj.weight"] = (rng.normal(0, k, (KV * DH, D))).astype(np.float32)
        w[b + "attn.v_proj.weight"] = (rng.normal(0, k, (KV * DH, D))).astype(np.float32)
        w[b + "attn.out_proj.weight"] = (rng.normal(0, k, (D, NH * DH))).astype(np.float32)
        w[b + "attn.q_proj.bias"] = np.zeros(NH * DH, np.float32)
        w[b + "attn.k_proj.bias"] = np.zeros(KV * DH, np.float32)
        w[b + "attn.v_proj.bias"] = np.zeros(KV * DH, np.float32)
        w[b + "ffn.w1.weight"] = (rng.normal(0, k, (FF, D))).astype(np.float32)
        w[b + "ffn.w2.weight"] = (rng.normal(0, k, (FF, D))).astype(np.float32)
        w[b + "ffn.w3.weight"] = (rng.normal(0, k, (D, FF)) / np.sqrt(FF)).astype(np.float32)
    return w


def tiny_sequences(cfg, n=4, seq_len=32, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, cfg["V"], size=seq_len).astype(np.int64).tolist() for _ in range(n)]


def save_params(model, path):
    """把当前参数存成 npz（fp32），供后续生成/续训。"""
    arrays = {k: np.asarray(v.to_numpy(), dtype=np.float32) for k, v in model.p.items()}
    np.savez(path, **arrays)
    return path


def load_params(model, path):
    """把 npz 里的权重灌回模型（原地覆盖，保持 Parameter 对象不变）。

    注意：CUDA 张量的数据在 `_gpu`（GpuBuf）里，`.data` 是 None ——
    直接 `p.data[...] = arr` 会 TypeError。所以要按设备分开处理。
    权重绑定时 lm_head 与词嵌入是**同一个对象**，只灌一次。
    """
    z = np.load(path, allow_pickle=False)
    seen, n = set(), 0
    for k, p in model.p.items():
        if id(p) in seen or k not in z.files:
            continue
        arr = np.ascontiguousarray(np.asarray(z[k], dtype=np.float32))
        if getattr(p, "device", "cpu") == "cuda":
            p._gpu = T._cuda().h2d(arr)
        else:
            p.data[...] = arr
        seen.add(id(p))
        n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", default="", help="HF 权重目录（含 model.safetensors）")
    ap.add_argument("--gguf", default=r"D:\DSH-Workspace\_gguf_test\qwen2.5-0.5b-fp16.gguf")
    ap.add_argument("--ref", default=os.path.join(REPO, "tools", "_qwen_ref.npz"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--forward-only", action="store_true")
    ap.add_argument("--steps", type=int, default=0, help="SFT 步数（0 表示不训）")
    ap.add_argument("--tiny", action="store_true", help="用随机权重的迷你模型验证 SFT 链路")
    ap.add_argument("--data", default="", help="jsonl 语料（user/assistant 字段）")
    ap.add_argument("--tokenizer", default="", help="tokenizer.json 路径")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--limit", type=int, default=8, help="最多取多少条样本")
    ap.add_argument("--save", default="", help="训练后保存权重的 npz 路径")
    ap.add_argument("--load", default="", help="从 npz 加载权重（生成/续训前）")
    ap.add_argument("--holdout", type=int, default=0,
                    help="最后 N 条留作留出集（不参与训练），探针用其中第一条")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--tiny-lr", type=float, default=3e-3)
    ap.add_argument("--tol", type=float, default=1e-5)
    a = ap.parse_args(argv)

    T.set_default_device(a.device)
    print("device=%s" % (T.get_device(),))

    if a.tiny:
        cfg = TINY_CFG
        model = Qwen25(cfg, make_tiny_weights(cfg))
        seqs = tiny_sequences(cfg)
        steps = a.steps or 3
        print("tiny 模式：D=%d L=%d NH=%d KV=%d FF=%d V=%d，样本 %d 条，steps=%d"
              % (cfg["D"], cfg["L"], cfg["NH"], cfg["KV"], cfg["FF"], cfg["V"],
                 len(seqs), steps))
        probe = seqs[0]                     # 固定探针，保证可比
        losses, probes = sft(model, seqs, steps, a.tiny_lr, a.seq_len, probe=probe)
        print("")
        print("训练 loss（轮转样本，不可比）：%.4f → %.4f" % (losses[0], losses[-1]))
        print("探针 loss（固定序列，可比）：  %.6f → %.6f" % (probes[0], probes[-1]))
        ok = probes[-1] < probes[0]
        print("结论：%s" % ("PASS —— SFT 链路可用（前向/反向/AdamW 全通）" if ok else "FAIL"))
        return 0 if ok else 1

    cfg = CFG
    if a.safetensors:
        print("读取 safetensors 目录：%s" % a.safetensors)
        w = load_safetensors_weights(a.safetensors, cfg)
    else:
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

    if a.steps > 0:
        if not a.data or not a.tokenizer:
            print("SFT 需要 --data 与 --tokenizer")
            return 2
        tok = load_tokenizer(a.tokenizer)
        seqs = build_sequences(tok, a.data, a.seq_len, a.limit)
        if not seqs:
            print("语料里没有可用的 user/assistant 样本")
            return 2
        print("样本 %d 条，seq_len=%d，steps=%d，lr=%g"
              % (len(seqs), a.seq_len, a.steps, a.lr))
        if a.holdout and len(seqs) > a.holdout:
            train_seqs = seqs[:-a.holdout]
            probe = seqs[-a.holdout:]         # 整段留出集：训练时**从未见过**
            print("训练 %d 条 / 留出 %d 条（探针取自留出集，可排除'背下训练样本'）"
                  % (len(train_seqs), a.holdout))
        else:
            train_seqs = seqs
            probe = seqs[0]
        losses, probes = sft(model, train_seqs, a.steps, a.lr, a.seq_len, probe=probe)
        drop = (probes[0] - probes[-1]) / max(abs(probes[0]), 1e-9)
        print("")
        print("探针 loss：%.6f → %.6f（变化 %.1f%%）" % (probes[0], probes[-1], 100 * drop))
        if a.save:
            save_params(model, a.save)
            print("权重已保存：%s" % a.save)
        gain = (probes[0] - probes[-1]) / max(abs(probes[0]), 1e-9)
        if gain >= 0.05:
            verdict = "PASS —— 留出集下降 %.1f%%，确实学到了" % (100 * gain)
            rc = 0
        elif gain > 0:
            verdict = ("未学到 —— 留出集只降 %.2f%%，在噪声范围内（判据 ≥5%%）。"
                       "修法：加步数/加语料/调 lr，别把噪声当成果" % (100 * gain))
            rc = 1
        else:
            verdict = "FAIL —— 留出集反而上升 %.1f%%（过拟合或 lr 过大）" % (-100 * gain)
            rc = 1
        print("结论：%s" % (verdict,))
        return rc

    print("没有指定动作：用 --forward-only 验证前向，或 --steps 做 SFT")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
