# -*- coding: utf-8 -*-
"""Qwen2.5 的引擎原生实现与 SFT 循环（库内版本，供 `tgai_go.api` 直接调用）。

这个模块是**从 examples/qwen25_sft.py 搬过来的单一真源**：
两者共用同一份代码，避免"示例与库各一份、行为悄悄分叉"。

覆盖：
  * `CFG`（Qwen2.5-0.5B 结构默认值）
  * `Qwen25` 模型（GQA + qkv bias + RoPE rotate_half + SwiGLU + 绑定词嵌入）
  * `build_sequences` / `_make_xy`（SFT 序列与 mask）
  * `sft`（训练循环：AdamW + 梯度裁剪 + 探针评估）
  * 权重：`load_safetensors_weights` / `make_tiny_weights` / `save_params` / `load_params`
  * `cfg_from_checkpoint`（checkpoints 命名 -> 模型 cfg 的唯一转换点）

GGUF 读取器留在 `examples/qwen25_sft.py`（它依赖仓库里的 `tools/gguf_to_tg.py`，
属于开发期工具，不属于运行时依赖）。
"""
import os

import numpy as np

import tgai_go.train as T
from tgai_go.train import checkpoints as C
from tgai_go.train import functional as F
from tgai_go.train import nn_ops as O


CFG = dict(V=151936, D=896, L=24, NH=14, KV=2, FF=4864, THETA=1e6, EPS=1e-6,
           DH=64, QK_NORM=False, QKV_BIAS=True, TIE=True)

def rope_inv_freq(dh, theta, scaling=None):
    """RoPE 的逆频率。支持 config.json 里的 `rope_scaling`。

    已实现：
      * `None` —— 标准 RoPE；
      * `{"rope_type"|"type": "linear", "factor": f}` —— inv_freq / f（线性插值）；
      * `{"rope_type"|"type": "llama3", "factor": f, "low_freq_factor": lo,
         "high_freq_factor": hi, "original_max_position_embeddings": L}` ——
         Llama-3 的**分段**缩放：低频整体除以 factor、高频不动、中间平滑过渡。

    没实现的类型（yarn / longrope / dynamic / mrope）**明确报错**，不静默忽略 ——
    静默忽略会让长上下文输出错且没有任何提示。
    """
    inv = 1.0 / (theta ** (np.arange(0, dh, 2, dtype=np.float64) / dh))
    if not scaling:
        return inv
    stype = str(scaling.get("rope_type") or scaling.get("type") or "").lower()
    factor = float(scaling.get("factor") or 1.0)
    if stype in ("", "default", "none"):
        return inv
    if stype == "linear":
        return inv / factor
    if stype == "llama3":
        lo = float(scaling.get("low_freq_factor") or 1.0)
        hi = float(scaling.get("high_freq_factor") or 4.0)
        old_len = float(scaling.get("original_max_position_embeddings") or 8192)
        low_wavelen = old_len / lo
        high_wavelen = old_len / hi
        wavelen = 2.0 * np.pi / inv
        # 波长比 low_wavelen 还长（=低频）⇒ 整体除以 factor
        scaled = np.where(wavelen > low_wavelen, inv / factor, inv)
        # 中间带做平滑过渡，避免频率出现跳变
        smooth = (old_len / wavelen - lo) / max(hi - lo, 1e-12)
        mid = (1.0 - smooth) * inv / factor + smooth * inv
        is_mid = (wavelen <= low_wavelen) & (wavelen >= high_wavelen)
        return np.where(is_mid, mid, scaled)
    raise ValueError(
        "rope_scaling 类型 %r 尚未实现（已支持: linear / llama3）。"
        "修法：① 用不带 scaling 的同族模型验证；② 或在本文件 rope_inv_freq 里补上该类型"
        "（参考 transformers 的 _compute_default_rope_parameters）；"
        "③ 千万不要直接忽略 —— 长上下文会静默算错" % (stype,))


def rope_tables(seq, dh, theta, scaling=None):
    """rotate_half 布局的 cos/sin：[S, dh]（前半与后半相同）。"""
    inv = rope_inv_freq(dh, theta, scaling)
    ang = np.outer(np.arange(seq, dtype=np.float64), inv)
    c = np.cos(ang).astype(np.float32)
    s = np.sin(ang).astype(np.float32)
    return np.concatenate([c, c], axis=-1), np.concatenate([s, s], axis=-1)


class Qwen25:
    """引擎原生 Qwen2.5。参数用 T.nn.Parameter，融合核用 T.nn.RMSNorm。"""

    def __init__(self, cfg, w, tie=None):
        self.cfg = cfg
        P = T.nn.Parameter
        w = dict(w)
        # Qwen3 官方 safetensors **带 lm_head.weight**（tie 模型里它与词嵌入相同）。
        # 仅凭"有没有 lm_head"判 tie 会把它当成不绑定 ⇒ 微调后两份分家。
        if cfg.get("TIE", True) and "lm_head.weight" in w:
            if not np.array_equal(w["lm_head.weight"], w["token_embedding.weight"]):
                raise ValueError(
                    "cfg 说 TIE=True，但 lm_head 与词嵌入不一致。"
                    "修法：检查加载器是否把独立 lm_head 混进来了")
            w.pop("lm_head.weight")
        self.p = {k: P(T.tensor(np.ascontiguousarray(v.astype(np.float32))))
                  for k, v in w.items()}
        # Qwen2.5-0.5B/1.5B 官方是**权重绑定**的（没有独立 lm_head）：
        # 这时必须让 lm_head 与词嵌入**共享同一个 Parameter 对象**，
        # 复制一份会让两条梯度各走各的，训练后两者就分家了。
        if tie is None:
            tie = bool(cfg.get("TIE", "lm_head.weight" not in self.p))
        self.tie = bool(tie)
        # 梯度检查点开关（显存换时间；用 cfg 传更方便做 A/B）
        self.use_ckpt = bool(cfg.get("GRAD_CKPT", False))
        # 段的粒度：默认 = 每层一次（等价于原来行为）；1 = 整段一次
        self.ckpt_segments = int(cfg.get("CKPT_SEGMENTS") or cfg.get("L") or 1)
        if self.tie:
            self.p["lm_head.weight"] = self.p["token_embedding.weight"]
        # 直接挂共享 Parameter：不要先随机初始化一张 V*D 的表再扔掉
        # （0.5B 模型那是 545 MB，实测会留在分配器缓存里）
        self.emb = T.nn.Embedding(cfg["V"], cfg["D"], weight=self.p["token_embedding.weight"])
        self.ln1, self.ln2 = [], []
        for i in range(cfg["L"]):
            b = "blocks.%d." % i
            m1 = T.nn.RMSNorm(cfg["D"], eps=cfg["EPS"])
            m1.weight = self.p[b + "ln1.weight"]
            m2 = T.nn.RMSNorm(cfg["D"], eps=cfg["EPS"])
            m2.weight = self.p[b + "ln2.weight"]
            self.ln1.append(m1)
            self.ln2.append(m2)
        DH = int(cfg.get("DH") or cfg["D"] // cfg["NH"])
        self.DH = DH
        # ★ 两处形状自洽性必须显式检查，否则错得**不响**：
        #   1) GQA 要求 NH % KV == 0。不整除时 repeat_interleave(NH//KV) 会少头，
        #      之后 reshape 才炸（甚至可能恰好不炸，静默算错）。
        #   2) q_proj 的行数必须是 NH*DH；DH 取错（比如用 D//NH）时能跑到 reshape 才崩。
        if int(cfg["KV"]) <= 0 or int(cfg["NH"]) % int(cfg["KV"]) != 0:
            raise ValueError(
                "GQA: num_attention_heads(%d) 必须能被 num_key_value_heads(%d) 整除。"
                "修法：检查权重目录里的 config.json（num_key_value_heads）是否与这份权重匹配"
                % (int(cfg["NH"]), int(cfg["KV"])))
        if DH <= 0 or DH % 2 != 0:
            raise ValueError(
                "head_dim=%d 非法（必须是正偶数，RoPE 要两两配对）。"
                "修法：从 config.json 的 head_dim 或 q_proj.shape[0]//n_heads 取" % (DH,))
        wq = self.p.get("blocks.0.attn.q_proj.weight")
        if wq is not None and int(wq.shape[0]) != int(cfg["NH"]) * DH:
            raise ValueError(
                "形状不自洽：blocks.0.attn.q_proj.weight 有 %d 行，但 n_heads*head_dim=%d"
                "（n_heads=%d, head_dim=%d）。修法：head_dim 取错了 —— "
                "Qwen3-0.6B 是 128（不是 1024/16=64）"
                % (int(wq.shape[0]), int(cfg["NH"]) * DH, int(cfg["NH"]), DH))
        self.qn, self.kn = [], []
        if cfg.get("QK_NORM"):
            for i in range(cfg["L"]):
                b = "blocks.%d." % i
                for mods, key in ((self.qn, b + "attn.q_norm.weight"),
                                  (self.kn, b + "attn.k_norm.weight")):
                    m = T.nn.RMSNorm(DH, eps=cfg["EPS"])     # 只归一化 head_dim
                    m.weight = self.p[key]
                    mods.append(m)
        self.final = T.nn.RMSNorm(cfg["D"], eps=cfg["EPS"])
        self.final.weight = self.p["final_norm.weight"]
        self._rope = {}

    def params(self):
        # tie 时 lm_head 与词嵌入是**同一个 Parameter**；重复传给 AdamW 会被更新两次
        # （实测 lr=1e-2 一步 0.98 而非 0.99；裁剪范数 6.928 vs 5.657）。
        seen, out = set(), []
        for prm in self.p.values():
            if id(prm) in seen:
                continue
            seen.add(id(prm))
            out.append(prm)
        return out

    def _cos_sin(self, S, DH):
        key = (S, DH)
        cs = self._rope.get(key)
        if cs is None:
            cos, sin = rope_tables(S, DH, self.cfg["THETA"],
                                   self.cfg.get("ROPE_SCALING"))
            cs = (T.tensor(cos[None, :, :]), T.tensor(sin[None, :, :]))
            self._rope[key] = cs
        return cs

    @staticmethod
    def _rope_apply(x, cos_t, sin_t):
        dh = int(x.shape[-1])
        h = dh // 2
        rot = T.cat([-x[:, :, h:], x[:, :, :h]], dim=-1)
        return x * cos_t + rot * sin_t

    def _layer(self, x, i):
        """单层前向（抽出来是为了能用 `T.checkpoint` 包住：反向时重算这一层）。"""
        cfg = self.cfg
        # 这些量原来在 forward 里算好、闭包共享；抽成方法后必须自己算一遍
        D, NH, KV = cfg["D"], cfg["NH"], cfg["KV"]
        DH = int(cfg.get("DH") or D // NH)
        B, S = int(x.shape[0]), int(x.shape[1])
        p = self.p
        cos_t, sin_t = self._cos_sin(S, DH)   # 已按 (S, DH) 缓存，重算无成本
        b = "blocks.%d." % i
        h = self.ln1[i](x).reshape(B * S, D)
        q = h @ p[b + "attn.q_proj.weight"].T
        k = h @ p[b + "attn.k_proj.weight"].T
        v = h @ p[b + "attn.v_proj.weight"].T
        if cfg.get("QKV_BIAS", True):        # Qwen3 无 qkv bias
            q = q + p[b + "attn.q_proj.bias"]
            k = k + p[b + "attn.k_proj.bias"]
            v = v + p[b + "attn.v_proj.bias"]
        q = q.reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        k = k.reshape(B, S, KV, DH).permute(0, 2, 1, 3).reshape(B * KV, S, DH)
        v = v.reshape(B, S, KV, DH).permute(0, 2, 1, 3).reshape(B * KV, S, DH)
        if cfg.get("QK_NORM"):               # per-head RMSNorm，必须在 RoPE **之前**
            q = self.qn[i](q)
            k = self.kn[i](k)
        q = self._rope_apply(q, cos_t, sin_t)
        k = self._rope_apply(k, cos_t, sin_t)
        # GQA：KV 头复制到 Q 头数（在 [B,KV,S,DH] 上按第 1 维展开，避免跨 batch 打乱）
        k = k.reshape(B, KV, S, DH).repeat_interleave(NH // KV, dim=1).reshape(B * NH, S, DH)
        v = v.reshape(B, KV, S, DH).repeat_interleave(NH // KV, dim=1).reshape(B * NH, S, DH)
        # 低显存注意力（不物化 P）：CUDA + dh==64 + causal 时可用。
        # 实测（V100，BH=8/S=128）：前向 relerr 5.7e-08、dq 2.4e-06 / dk 2.3e-06 / dv 2.1e-07，
        # 与物化实现等价；默认关闭，SDPA_LOWMEM=1 打开（等显存/速度收益实测后再定默认）。
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            lowmem=bool(cfg.get("SDPA_LOWMEM", False)))
        o = o.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, NH * DH)
        x = x + (o @ p[b + "attn.out_proj.weight"].T).reshape(B, S, D)

        h2 = self.ln2[i](x).reshape(B * S, D)
        g = h2 @ p[b + "ffn.w1.weight"].T
        u = h2 @ p[b + "ffn.w2.weight"].T
        act = F.silu(g) * u
        x = x + (act @ p[b + "ffn.w3.weight"].T).reshape(B, S, D)
        return x


    def forward(self, ids, hidden=False):
        """ids: numpy int64 [B, S] → logits [B, S, V]（hidden=True 时只到 final norm 的 z）。

        `hidden=True` 是给**分块 CE** 用的：先拿 z [B*S, D]，再按 token 分块投影，
        避免一次materialize [B*S, V] 这个大张量（B=4/S=512 时它是 1.24 GB）。
        """
        cfg = self.cfg
        D, L, NH, KV, FF = cfg["D"], cfg["L"], cfg["NH"], cfg["KV"], cfg["FF"]
        DH = int(cfg.get("DH") or D // NH)      # Qwen3-0.6B: 128 != 1024/16
        B, S = int(ids.shape[0]), int(ids.shape[1])
        p = self.p
        cos_t, sin_t = self._cos_sin(S, DH)

        x = self.emb(T.tensor(ids) if T.__name__ == "torch" else ids).reshape(B, S, D)

        if self.use_ckpt:
            # 梯度检查点：前向不建图（激活用完即弃），反向时按段重算。
            # 段的粒度由 CKPT_SEGMENTS 决定（PyTorch 的 segments 语义）：
            #   = L -> 每层一次（省最多、重算最频繁）；= 1 -> 整段一次（只存边界）。
            # 实测 B=4/S=512：关 28.55 GiB -> 开 19.97 GiB（-33%），数值逐位一致。
            x = T.checkpoint_sequential(
                [lambda h, i=i: self._layer(h, i) for i in range(L)], x,
                segments=self.ckpt_segments)
        else:
            for i in range(L):
                x = self._layer(x, i)

        z = self.final(x).reshape(B * S, D)
        if hidden:
            return z
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
    n = len(ids)                          # 真实 token 数（补齐前）
    pad = ids[-1]
    ids = (ids + [pad] * seq_len)[:seq_len]
    xb = np.asarray([ids[:-1]], dtype=np.int64)
    yb = np.asarray([ids[1:]], dtype=np.int64)
    cut = max(0, mask_from - 1)          # 预测 ids[i+1]，回答从 mask_from 开始
    if cut:
        yb[:, :cut] = -100
    # ★ 补位必须掩掉（-100），否则模型有大量 loss 花在"重复输出最后一个 token"上。
    #   实测症状：36 个真实 token 的样本补齐到 128 后，非掩码目标有 127 个 ——
    #   等于 92 个补位 token 都在参与训练，真正的回答信号被淹没；
    #   表现为 loss 掉得飞快（补位太好猜）但模型不会说话、加数据也没用。
    #   yb[i] = ids[i+1]，真实目标只到 i = n-2，所以从 n-1 开始都是补位。
    if n - 1 < seq_len - 1:
        yb[:, max(0, n - 1):] = -100
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


def sft(model, seqs, steps, lr, seq_len, log_every=1, probe=None, clip=0.0,
        warmup=0, probe_every=25, batch=1, shuffle=True, seed=0,
        start_step=0, opt=None):
    """最小可用 SFT 循环：AdamW + 交叉熵（全 token）。

    注意：训练样本轮转 ⇒ **每步 loss 不可比**（不同样本难度不同）。
    所以这里同时报一条固定 `probe` 序列上的 loss，只看它是否下降。
    """
    import time
    cfg = model.cfg
    if opt is None:
        opt = T.optim.AdamW(model.params(), lr=lr, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.0)
    losses, probes, gnorms = [], [], []
    t0 = time.time()
    base_lr = float(lr)
    rng_s = np.random.default_rng(seed)
    # ★ 预先按 seed 生成整条取样计划：第 s 步固定用 plan[s*batch:(s+1)*batch]。
    #   之前是"跑到 pos==0 才 shuffle"，续训从 step 0 重排 ⇒ 顺序错位、loss 不可比。
    need = (start_step + steps) * batch
    plan = []
    while len(plan) < need:
        idx = list(range(len(seqs)))
        if shuffle:
            rng_s.shuffle(idx)
        plan.extend(idx)
    # 探针很贵（8 条 × 完整前向 ≈ 0.4 s），不能每步都跑
    if probe is not None:
        base = _probe_multi(model, probe, seq_len)
        probes.append(base)
        print("探针基线（训练前，留出集平均）= %.6f" % (base,))
    for step in range(start_step, start_step + steps):
        # 线性 warmup：真实大模型上 lr 一步到位会炸（实测 probe 0.42 -> 17.2）
        # 注意用**真实步号** step（续训时 start_step>0，不能重跑 warmup）
        if warmup and step < warmup:
            opt.lr = base_lr * float(step + 1) / float(warmup)
        else:
            opt.lr = base_lr
        if hasattr(opt, "zero_grad"):
            opt.zero_grad()
        else:
            for prm in model.params():
                prm.zero_grad()

        # mini-batch：逐条前向/反向，梯度自动累加（batch=1 时噪声大到留出集反而变差）
        idxs = plan[step * batch:(step + 1) * batch]
        for k, si in enumerate(idxs):
            xb, yb = _make_xy(seqs[si], seq_len)
            if cfg.get("CHUNK_CE"):
                # 分块 CE：不落地 [B*S, V]，省 3 份词表大小的张量
                zc = model.forward(xb, hidden=True)
                loss = chunked_ce(zc, model.p["lm_head.weight"], yb.reshape(-1),
                                  chunk=int(cfg.get("CHUNK_TOKENS") or 8192))
            else:
                logits = model.forward(xb)
                loss = F.cross_entropy(logits.reshape(xb.size, cfg["V"]), yb.reshape(-1))
            (loss * (1.0 / batch)).backward()
        train_loss = float(loss.to_numpy())
        # 梯度裁剪：真实大模型上 lr 稍大就会炸（实测 probe 从 0.42 冲到 17.2），
        # 裁剪是把「能训」变成「训得稳」的必要一步。
        gnorm = O.clip_grad_norm_(model.params(), clip) if clip else 0.0
        if gnorm:
            gnorms.append(float(gnorm))
        opt.step()

        losses.append(train_loss)
        pl = None
        if probe is not None and ((step + 1) % probe_every == 0
                                  or step + 1 == start_step + steps):
            pl = _probe_multi(model, probe, seq_len)
            probes.append(pl)
        if (step + 1) % log_every == 0:
            extra = ("| probe %.6f " % probes[-1]) if pl is not None else ""
            print("step %3d | train %6.4f %s| gnorm %7.2f | %.1f s/step"
                  % (step + 1, train_loss, extra, gnorm,
                     (time.time() - t0) / (step + 1 - start_step)))
    # gnorm 中位数是判断"学习率是否合适"最直接的指标，比看 loss 曲线快得多：
    # 中位数远大于 clip 阈值 ⇒ 每步都被裁、有效步长被压扁，方向也被大梯度主导。
    if gnorms:
        g = np.asarray(gnorms, np.float64)
        print("gnorm 统计：中位数 %.2f / p90 %.2f / 最大 %.2f（clip=%s）%s"
              % (float(np.median(g)), float(np.percentile(g, 90)), float(g.max()), clip,
                 "  ⚠ 中位数远高于 clip 阈值，学习率可能偏大" if clip and np.median(g) > 5 * clip else ""))
    return losses, probes


# ---------------------------------------------------------------------------
# safetensors 加载（HF 官方权重目录）
# ---------------------------------------------------------------------------

def hf_name_to_ours(name):
    """薄封装：HF 名 → 我方名（与工具共用 checkpoints 里的同一张表）。"""
    return C.hf_name_to_ours(name)


def load_safetensors_weights(dirpath, cfg=None):
    """薄封装：读 HF safetensors 目录，实现在 checkpoints。"""
    return C.load_safetensors_dir(dirpath).weights


TINY_CFG = dict(V=256, D=64, L=2, NH=4, KV=2, FF=128, THETA=1e6, EPS=1e-6)


def make_tiny_weights(cfg, seed=0):
    """随机初始化同结构权重（形状与 Qwen2.5 完全一致，只是小很多）。"""
    rng = np.random.default_rng(seed)
    D, L, NH, KV, FF, V = cfg["D"], cfg["L"], cfg["NH"], cfg["KV"], cfg["FF"], cfg["V"]
    DH = int(cfg.get("DH") or D // NH)
    qk_norm = bool(cfg.get("QK_NORM"))
    qkv_bias = bool(cfg.get("QKV_BIAS", True))
    tied = bool(cfg.get("TIE", True))
    k = 1.0 / np.sqrt(D)
    w = {"token_embedding.weight": (rng.normal(0, 0.02, (V, D))).astype(np.float32),
         **({} if tied else {"lm_head.weight": (rng.normal(0, 0.02, (V, D))).astype(np.float32)}),
         "final_norm.weight": np.ones(D, np.float32)}
    for i in range(L):
        b = "blocks.%d." % i
        w[b + "ln1.weight"] = np.ones(D, np.float32)
        w[b + "ln2.weight"] = np.ones(D, np.float32)
        w[b + "attn.q_proj.weight"] = (rng.normal(0, k, (NH * DH, D))).astype(np.float32)
        w[b + "attn.k_proj.weight"] = (rng.normal(0, k, (KV * DH, D))).astype(np.float32)
        w[b + "attn.v_proj.weight"] = (rng.normal(0, k, (KV * DH, D))).astype(np.float32)
        w[b + "attn.out_proj.weight"] = (rng.normal(0, k, (D, NH * DH))).astype(np.float32)
        if qkv_bias:
            w[b + "attn.q_proj.bias"] = np.zeros(NH * DH, np.float32)
            w[b + "attn.k_proj.bias"] = np.zeros(KV * DH, np.float32)
            w[b + "attn.v_proj.bias"] = np.zeros(KV * DH, np.float32)
        if qk_norm:
            # 真实 Qwen3 的 q_norm/k_norm 不是全 1（实测 mean 1.96 / 3.57），
            # 这里也用非平凡值，避免"不接线也能过"
            w[b + "attn.q_norm.weight"] = (1.0 + 0.1 * rng.normal(size=DH)).astype(np.float32)
            w[b + "attn.k_norm.weight"] = (1.0 + 0.1 * rng.normal(size=DH)).astype(np.float32)
        w[b + "ffn.w1.weight"] = (rng.normal(0, k, (FF, D))).astype(np.float32)
        w[b + "ffn.w2.weight"] = (rng.normal(0, k, (FF, D))).astype(np.float32)
        w[b + "ffn.w3.weight"] = (rng.normal(0, k, (D, FF)) / np.sqrt(FF)).astype(np.float32)
    return w


def tiny_sequences(cfg, n=4, seq_len=32, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, cfg["V"], size=seq_len).astype(np.int64).tolist() for _ in range(n)]


def _to_np(x):
    """numpy 数组 / Tensor / GpuBuf 统一转成 np.float32。"""
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=np.float32)
    if hasattr(x, "to_numpy"):
        return np.asarray(x.to_numpy(), dtype=np.float32)
    return np.asarray(T._cuda().d2h(x), dtype=np.float32)


def save_params(model, path, opt=None, step=0):
    """存参数（+ 优化器状态）成 npz，供生成 / **续训**。

    只存参数不够：AdamW 的 m/v/t 丢了，续训等于重启动量，轨迹不连续。
    """
    arrays = {k: _to_np(v) for k, v in model.p.items()}
    arrays["__step"] = np.asarray([int(step)], np.int64)
    if opt is not None:
        arrays["__opt_t"] = np.asarray([int(getattr(opt, "t", 0))], np.int64)
        for i, (m, v) in enumerate(zip(opt.m, opt.v)):
            arrays["__opt_m_%d" % i] = _to_np(m)
            arrays["__opt_v_%d" % i] = _to_np(v)
    np.savez(path, **arrays)
    return path


def load_params(model, path, opt=None):
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
    # 优化器状态（有就恢复；老存档没这些键 ⇒ 保持"只恢复参数"的旧行为）
    state = {"params": n, "step": int(z["__step"][0]) if "__step" in z.files else 0,
             "opt": False}
    if opt is not None and "__opt_t" in z.files:
        opt.t = int(z["__opt_t"][0])
        rt = None
        for i in range(len(opt.m)):
            km, kv = "__opt_m_%d" % i, "__opt_v_%d" % i
            if km not in z.files or kv not in z.files:
                continue
            am = np.ascontiguousarray(np.asarray(z[km], dtype=np.float32))
            av = np.ascontiguousarray(np.asarray(z[kv], dtype=np.float32))
            if isinstance(opt.m[i], np.ndarray):
                opt.m[i][...] = am
                opt.v[i][...] = av
            else:                                  # GpuBuf：换缓冲
                rt = rt or T._cuda()
                opt.m[i] = rt.h2d(am)
                opt.v[i] = rt.h2d(av)
        state["opt"] = True
    return state


def build_lm_sequences(tok, path, seq_len, limit=None):
    """把**纯文本**切成定长 token 块，用于预训练（全 token 参与 loss）。

    与 `build_sequences` 的区别：那个走 Qwen 的对话模板并只用答案算 loss（SFT），
    这个不做任何模板、不遮 prompt —— 每个 token 都是训练目标。

    支持 `.txt`（整篇）与 `.jsonl`（每行的 `text` 字段），空行/空文本跳过。
    末尾不足 seq_len 的残块**丢弃**（拼半个序列当一条样本会引入伪边界）。
    """
    import json
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "build_lm_sequences: 找不到语料文件 %s。修法：传 .txt 或 .jsonl 路径" % (path,))
    chunks = []
    cur = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.endswith(".jsonl"):
                try:
                    line = json.loads(line).get("text") or ""
                except json.JSONDecodeError:
                    raise ValueError(
                        "build_lm_sequences: %s 有一行不是合法 JSON。"
                        "修法：预训练语料可以是 .txt（整篇）或每行 {\"text\": ...}" % (path,))
                if not line:
                    continue
            ids = tok.encode(line).ids
            cur.extend(ids)
            cur.append(int(tok.token_to_id("<|endoftext|>")) if
                       tok.token_to_id("<|endoftext|>") is not None else 0)
            while len(cur) >= int(seq_len):
                chunks.append(cur[:int(seq_len)])
                cur = cur[int(seq_len):]
                if limit and len(chunks) >= int(limit):
                    return chunks
    return chunks


def make_random_weights(cfg, seed=0):
    """按 cfg 随机初始化一套权重（预训练起点）。

    就是 `make_tiny_weights` 的别名 —— 那个函数本来就按 cfg 通用生成，
    名字里的 "tiny" 只是当初的用处。这里单独导出，避免读代码的人误会。
    """
    return make_tiny_weights(cfg, seed=seed)


def chunked_ce(z, W, target, chunk=8192, ignore_index=-100):
    """分块交叉熵：`z` [N,D]（final norm 之后的隐状态）、`W` [V,D]、`target` [N]。

    与"先算整块 logits 再 cross_entropy"在数值上等价（测试逐位对比），
    但**不落地** [N,V]：每块只算 chunk×V 的 logits，包在 `T.checkpoint` 里，
    反向时按块重算。
    """
    N = int(z.shape[0])
    tgt = np.asarray(target.to_numpy() if hasattr(target, "to_numpy") else target,
                     dtype=np.int64).reshape(-1)
    valid = tgt != int(ignore_index)
    total = int(valid.sum())
    if total == 0:
        return z.sum() * 0.0                     # 全掩码：零 loss，保持可反传

    acc = None
    step = max(1, int(chunk))
    for s0 in range(0, N, step):
        s1 = min(s0 + step, N)
        m = int(valid[s0:s1].sum())
        if m == 0:
            continue
        hc = z[s0:s1]                      # 引擎 Tensor 支持切片；别把整块传进去
        lg = T.checkpoint(lambda hh: hh @ W.T, hc)
        ce = T.functional.cross_entropy(lg, tgt[s0:s1], ignore_index)
        ce = ce * float(m) / float(total)        # 换算成全局均值
        acc = ce if acc is None else acc + ce
    return acc


def cfg_from_checkpoint(ck):
    """把 `checkpoints.Checkpoint` 的结构配置转换成模型类要的 cfg。

    两套命名并存是历史原因（checkpoints 用 HF 风格 n_embd/n_layer，模型类用 D/L），
    这里做**唯一**的转换点，避免每个脚本各写一遍（写错就静默出错）。
    """
    c = ck.config
    return dict(V=int(c["vocab_size"] or 0), D=int(c["n_embd"]), L=int(c["n_layer"]),
                NH=int(c["n_head"]), KV=int(c.get("n_head_kv") or c["n_head"]),
                FF=int(c.get("n_ff") or 0),
                THETA=float(c.get("rope_theta", 1e6)),
                EPS=float(c.get("eps", 1e-6)),
                DH=int(c.get("head_dim") or (int(c["n_embd"]) // int(c["n_head"]))),
                QK_NORM=bool(c.get("QK_NORM") or any("q_norm" in k for k in ck.weights)),
                QKV_BIAS=bool(c.get("QKV_BIAS",
                                    any(k.endswith("q_proj.bias") for k in ck.weights))),
                TIE=bool(c.get("tie_embeddings", True)),
                ROPE_SCALING=c.get("rope_scaling") or None)

