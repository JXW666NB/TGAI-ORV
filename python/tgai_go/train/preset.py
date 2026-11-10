# -*- coding: utf-8 -*-
"""**开箱即用层**（0.9.21）：让"10 行以内开始训练"成立。

## 设计原则

1. **只包装，不复制**：预设模型由已有的 `nn` 层搭出来（Linear/Embedding/LayerNorm/
   RMSNorm/functional.sdpa），不是另写一套 forward，所以数学与 `docs/REVIEW` 的对照表一致。
2. **可拆**：`preset` 返回的是普通 `nn.Module`，用户随时能用 `model.layers` 拆开改；
   不要预设就用自己写的 Module，`fit()` 一样吃。
3. **说清边界**：这是"小白的起跑线"，不是魔法。默认 fp32、默认无 AMP、默认单卡。

## 用法（4 行）

```python
import tgai_go.train as T

model = T.preset.transformer(vocab=32000, d_model=512, n_layers=8, n_heads=8)
data  = T.tokens(list_of_token_ids, seq=512, batch=4)      # 或 T.text("corpus.txt", ...)
T.fit(model, data, steps=1000, lr=3e-4)                    # 内含 AdamW+cosine+裁剪
```

`T.fit` 返回 `history`（每步 loss），可以直接画曲线。
"""
import numpy as np

from . import (  # noqa: F401
    Embedding, LayerNorm, Linear, Module, ModuleList, Parameter, RMSNorm,
)
from . import functional as _F
from . import nn as _nn
from . import optim as _optim
from . import sched as _sched

__all__ = ["transformer", "Transformer", "tokens", "fit"]


# ═══════════════════════════════════════════════════════════
# 预设架构：pre-LN 的 decoder-only Transformer（与 LLaMA/GPT 同族）
# ═══════════════════════════════════════════════════════════
class Transformer(Module):
    """pre-LN decoder-only Transformer：`x -> [LN -> Attn + res] x N -> LN -> lm_head`。

    参数（默认值刻意小，CPU 也能秒级跑起来验证）：

    | 名字 | 含义 |
    | --- | --- |
    | `vocab` | 词表大小 |
    | `d_model` | 隐藏维 |
    | `n_layers` | 层数 |
    | `n_heads` | 头数（须整除 `d_model`） |
    | `d_ff` | FFN 中间维，默认 `4 * d_model` |
    | `norm` | `"rms"`（LLaMA 风格，默认）或 `"layer"` |
    | `tie_embedding` | 是否让 lm_head 与词嵌入共享权重（默认 True） |
    | `dropout` | 残差后 dropout，0 表示不用 |
    """

    def __init__(self, vocab=1024, d_model=128, n_layers=4, n_heads=4, d_ff=None,
                 norm="rms", tie_embedding=True, dropout=0.0, eps=1e-6):
        super(Transformer, self).__init__()
        if int(d_model) % int(n_heads):
            raise ValueError("d_model(%d) 必须能被 n_heads(%d) 整除" % (d_model, n_heads) +
                             "。修法：改成 512/8 这种组合")
        self.vocab = int(vocab)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads
        d_ff = int(d_ff or 4 * self.d_model)

        self.embed = Embedding(self.vocab, self.d_model)
        norm_cls = RMSNorm if str(norm).lower().startswith("rms") else LayerNorm
        self.blocks = ModuleList([_Block(self.d_model, self.n_heads, d_ff, norm_cls, eps, dropout)
                                  for _ in range(self.n_layers)])
        self.norm_f = norm_cls(self.d_model) if norm_cls is RMSNorm else norm_cls(self.d_model, eps=eps)
        self.lm_head = Linear(self.d_model, self.vocab, bias=False)
        if tie_embedding:
            # 共享同一份权重（PyT 里是 tie_weights()，这里直接指向同一 Parameter）
            self.lm_head.weight = self.embed.weight
        self.tie_embedding = bool(tie_embedding)

    def forward(self, idx, targets=None):
        """`idx` 是 int 的 `[seq]` / `[batch, seq]`；给了 `targets` 就返回标量 loss。

        batch 维是**真支持**的：内部按 `[B*S, D]` 算，注意力按 `[B*H, S, Dh]` 分组。
        """
        idx = np.asarray(idx)
        squeeze = idx.ndim == 1
        if squeeze:
            idx = idx[None, :]
        if idx.ndim != 2:
            raise ValueError("预设 Transformer 只接受 [seq] 或 [batch, seq]，收到 %s" % (idx.shape,) +
                             "。修法：把序列拼成 2D，或检查输入的 ndim")
        B, S = int(idx.shape[0]), int(idx.shape[1])
        x = self.embed(idx.reshape(B * S)).reshape(B, S, self.d_model)
        for blk in self.blocks:
            x = blk(x, B, S)
        h = self.norm_f(x.reshape(B * S, self.d_model))
        logits = self.lm_head(h).reshape(B, S, self.vocab)
        if targets is None:
            return logits.reshape(S, self.vocab) if squeeze else logits
        tgt = np.asarray(targets)
        if squeeze:
            tgt = tgt.reshape(1, -1)
        loss = _F.cross_entropy(logits.reshape(B * S, self.vocab), tgt.reshape(B * S))
        return loss

    def n_params(self):
        """参数总量（含共享权重只算一次）。"""
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            n = 1
            for s in tuple(p.shape):
                n *= int(s)
            total += n
        return total


class _Block(Module):
    """一个 pre-LN 块：LN → 多头因果注意力 → 残差；LN → SwiGLU FFN → 残差。"""

    def __init__(self, d_model, n_heads, d_ff, norm_cls, eps, dropout):
        super(_Block, self).__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        mk = (lambda: norm_cls(d_model)) if norm_cls is RMSNorm else (lambda: norm_cls(d_model, eps=eps))
        self.ln1 = mk()
        self.ln2 = mk()
        self.qkv = Linear(d_model, 3 * d_model, bias=False)
        self.o = Linear(d_model, d_model, bias=False)
        self.w1 = Linear(d_model, d_ff, bias=False)
        self.w3 = Linear(d_model, d_ff, bias=False)
        self.w2 = Linear(d_ff, d_model, bias=False)
        self.drop = _nn.Dropout(dropout) if dropout else None

    def forward(self, x, B=None, S=None):
        """`x` 是 `[B,S,D]`；不给 `B/S` 时按 2D `[S,D]`（batch=1）处理。"""
        import tgai_go.train as _T
        if B is None:
            S = int(x.shape[0])
            B = 1
        D, H, Dh = self.d_model, self.n_heads, self.d_head
        xf = x.reshape(B * S, D)
        h = self.ln1(xf)
        qkv = self.qkv(h)                                  # [B*S, 3D]
        q = qkv[:, 0:D]
        k = qkv[:, D:2 * D]
        v = qkv[:, 2 * D:3 * D]
        # [B*S, D] -> [B, S, H, Dh] -> [B*H, S, Dh]
        def heads(t):
            return t.reshape(B, S, H, Dh).permute(0, 2, 1, 3).reshape(B * H, S, Dh)

        q, k, v = heads(q), heads(k), heads(v)
        att = _F.scaled_dot_product_attention(q, k, v, causal=True)
        # [B*H,S,Dh] -> [B,S,H,Dh] -> [B*S, D]
        att = att.reshape(B, H, S, Dh).permute(0, 2, 1, 3).reshape(B * S, D)
        y = xf + self.o(att)
        if self.drop is not None:
            y = self.drop(y)
        h2 = self.ln2(y)
        z = _F.silu(self.w1(h2)) * self.w3(h2)             # SwiGLU
        y = y + self.w2(z)
        if self.drop is not None:
            y = self.drop(y)
        return y.reshape(B, S, D)


def transformer(**kwargs):
    """`T.preset.transformer(...)` 的简写，返回 `Transformer`。"""
    return Transformer(**kwargs)


# ═══════════════════════════════════════════════════════════
# 数据：token id 序列切块（最小的、能跑通的那一种）
# ═══════════════════════════════════════════════════════════
class _TokenBlocks(object):
    """把一长串 token id 切成定长块，每块 `seq+1` 个（最后一个当 label）。"""

    def __init__(self, ids, seq=128, drop_last=True):
        import tgai_go.train as T
        ids = np.asarray(ids, np.int64).reshape(-1)
        n = (ids.size - 1) // int(seq)
        if n <= 0:
            raise ValueError("token 太少：只有 %d 个，seq=%d 至少要 %d 个"
                             % (ids.size, seq, seq + 1) + "。修法：给更多文本，或减小 seq")
        self.seq = int(seq)
        self.blocks = [ids[i * self.seq:(i + 1) * self.seq + 1] for i in range(n)]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        b = self.blocks[i]
        return np.ascontiguousarray(b[:-1]), np.ascontiguousarray(b[1:])


def tokens(ids, seq=128):
    """`T.tokens([...])` —— 最省事的数据入口（不依赖 BPE/文件）。"""
    return _TokenBlocks(ids, seq=seq)


class Tokenizer(object):
    """极简 BPE 分词器（`train_tokenizer()` 的返回值）。

    只为"从零开始训自己的模型"这条链路存在：给一段文本就能造词表、能 encode，
    不需要外部 tokenizer 文件。词表规模建议 2000~8000（太小会把词切得很碎，太大在
    小语料上会过拟合）。真正的工程实践里应当换成成熟的分词器 —— 这个只保证可用、可复现。
    """

    def __init__(self, vocab, merges):
        self.vocab = list(vocab)
        self.merges = list(merges)
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.rank = {p: r for r, p in enumerate(self.merges)}
        self.merged = {}
        for (a, b) in self.merges:
            self.merged[(a, b)] = self.stoi[self.vocab[a] + self.vocab[b]]

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        """文本 → id 列表；规则与训练时一致（反复合并 rank 最小的相邻对）。"""
        from .bpe import encode_ref
        return encode_ref(text, self.vocab, self.merges)

    def __repr__(self):
        return "Tokenizer(vocab_size=%d, merges=%d)" % (self.vocab_size, len(self.merges))


def train_tokenizer(text, vocab_size=4000):
    """`T.train_tokenizer(text, vocab_size)` —— 从**你自己的语料**训一个分词表。

    这是"新手四步"里的第 2 步。返回的对象有 `.vocab_size` 与 `.encode(text)`。
    """
    from .bpe import train_bpe
    if not isinstance(text, str) or not text:
        raise ValueError("train_tokenizer 需要非空字符串（先用 "
                         "open(path, encoding='utf-8').read() 读入语料）" +
                         "。修法：检查语料路径与编码")
    vocab, merges = train_bpe(text, int(vocab_size))
    return Tokenizer(vocab, merges)


def load_corpus(path, encoding="utf-8"):
    """`T.load_corpus(path)` —— 读纯文本语料（第 1 步）。

    也接受 `.jsonl`：每行的 `text` 字段会被拼起来（与 `T.data.read_jsonl` 的约定一致）。
    """
    import io as _io
    import json as _json
    with _io.open(path, "r", encoding=encoding, errors="replace") as f:
        raw = f.read()
    if path.lower().endswith(".jsonl"):
        parts = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except ValueError:
                parts.append(line)
                continue
            if isinstance(obj, dict):
                parts.append(str(obj.get("text") or obj.get("content") or ""))
            else:
                parts.append(str(obj))
        return "\n".join(parts)
    return raw


# ═══════════════════════════════════════════════════════════
# 训练循环：一行把模型练起来
# ═══════════════════════════════════════════════════════════
def fit(model, data, steps=100, lr=3e-4, weight_decay=0.1, clip=1.0, warmup=0.03,
        opt=None, sched=None, amp=False, log_every=0, device=None, acc=None):
    """跑 `steps` 步训练，返回 `{"loss": [...], "steps": n}`。

    内含（都可覆盖）：`AdamW` + 带 warmup 的 cosine、梯度裁剪、可选 AMP。
    `data` 需要是**可索引**的（长度 + `[i] -> (x, y)`），或者给个 iterable。
    """
    import tgai_go.train as T
    if device is not None:
        T.set_default_device(device)
    params = list(model.parameters())
    if not params:
        raise ValueError("模型没有参数" + "。修法：确认 model.parameters() 非空")
    opt = opt or _optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
    sched = sched or _sched.CosineLR(opt, T_max=int(steps),
                                     warmup_steps=int(float(warmup) * int(steps)))

    n = len(data) if hasattr(data, "__len__") else None
    if n is None:
        raise ValueError("data 需要 len() 与下标访问" + "。修法：用 T.tokens(...) 或自己的 Dataset")

    history = []
    for step in range(int(steps)):
        x, y = data[step % n]
        if amp:
            with T.amp():
                loss = model(x, y)
        else:
            loss = model(x, y)
        for p in params:
            p.grad = None
        loss.backward()
        if clip:
            T.clip_grad_norm_(params, float(clip))
        opt.step()
        try:
            sched.step()
        except TypeError:
            sched.step(step + 1)
        v = float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy") else loss))
        history.append(v)
        if log_every and (step + 1) % int(log_every) == 0:
            print("[fit] step %d/%d loss %.6f" % (step + 1, steps, v), flush=True)
    return {"loss": history, "steps": int(steps)}
