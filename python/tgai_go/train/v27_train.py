# -*- coding: utf-8 -*-
"""V2.7 全参训练循环（ORV）：**逐层反向**，把梯度显存压到"单层"。

为什么这样设计（实测依据，见 docs/TRAIN_V27_ORV.md）：
* 3.033B 全参 fp32 权重 11.3 GB、fp16 动量 11.3 GB；
* 若梯度也全量常驻（fp32 11.3 GB）合计 33.9 GB ⇒ 超 32 GB；
* 逐层反向把梯度变成"算完这层立刻 step 再释放" ⇒ 峰值约 23 GB。

用法：
    tr = V27Trainer(cfg).load_all(sd)          # sd: {name: torch/np 张量}
    opts = tr.make_optimizers(lr=1e-5)
    xs = tr.forward_all(x0, cos, sin)          # 不建图，只缓存每层输入
    tr.backward_and_step(xs, g_out, cos, sin, opts)
"""
from __future__ import annotations

import math

import numpy as np

import tgai_go.train as T
from tgai_go.train import _cuda
from tgai_go.train import Linear, MoE, RMSNorm


def _np(z):
    """Tensor/ndarray → ndarray（CPU 上 grad 是 ndarray，GPU 上是 Tensor）。"""
    return z.to_numpy() if hasattr(z, "to_numpy") else np.asarray(z)


def _as_np(a):
    """torch tensor / ndarray → fp32 连续 ndarray（流式装载要求逐张量）。"""
    if hasattr(a, "float") and hasattr(a, "numpy"):
        a = a.float().numpy()
    return np.ascontiguousarray(a, np.float32)


class V27Trainer(object):
    """按层组织，支持逐层反向；权重经 `load_state_dict_stream` 流式上卡。"""

    def __init__(self, cfg, n_layers=None):
        self.cfg = dict(cfg)
        self.D = int(cfg["d_model"])
        self.NH = int(cfg["n_heads"])
        self.DH = self.D // self.NH
        self.FF = int(cfg["d_ff"])
        self.E = int(cfg["n_experts"])
        self.K = int(cfg["n_activated"])
        self.NL = int(n_layers if n_layers is not None else cfg["n_layers"])
        self.layers = []

    # ---------- 建层 ----------
    def _build_layer(self):
        D, NH, DH, FF, E, K = self.D, self.NH, self.DH, self.FF, self.E, self.K
        L = dict(ln1=RMSNorm(D), ln2=RMSNorm(D), q=[], k=[], v=[], o=[])
        for _ in range(NH):
            L["q"].append(Linear(D, DH, bias=False))
            L["k"].append(Linear(D, DH, bias=False))
            L["v"].append(Linear(D, DH, bias=False))
            L["o"].append(Linear(DH, D, bias=False))
        L["moe"] = MoE(D, FF, E, top_k=K, gate="softmax_topk",
                       qb_bias=np.zeros(E, np.float32))
        # 构造期分配的随机权重在 GPU 上会占满一份（MoE 每层 384 MB）；
        # 这里**连设备缓冲一起释放**，之后由流式装载只分配一次 ⇒ 避免 2~3 倍开销。
        for m in [L["ln1"], L["ln2"], L["moe"]] + L["q"] + L["k"] + L["v"] + L["o"]:
            for prm in m.parameters():
                if getattr(prm, "device", "cpu") == "cuda":
                    buf = getattr(prm, "_gpu", None)
                    if buf is not None and int(getattr(buf, "n", 0)) > 0:
                        try:
                            _cuda().free(int(buf.ptr))
                        except Exception:
                            pass
                    prm._lazy_shape = tuple(int(x) for x in buf.shape) if buf is not None else prm._lazy_shape
                    prm._gpu = None
                    prm.data = None
        if getattr(L["moe"].qb_bias, "device", "cpu") == "cuda":
            L["moe"].qb_bias.data = None
        return L

    # ---------- 装载 ----------
    def load_layer(self, i, sd):
        """把第 i 层权重流式装进已在卡上的模块（宿主峰值 = 单张量）。"""
        D, NH, DH = self.D, self.NH, self.DH
        L = self._build_layer()
        p = "blocks.%d." % i

        def items():
            yield "ln1.weight", _as_np(sd[p + "ln1.weight"])
            yield "ln2.weight", _as_np(sd[p + "ln2.weight"])
        for mod, key in ((L["ln1"], "ln1.weight"), (L["ln2"], "ln2.weight")):
            mod.load_state_dict_stream(iter([(key.split(".")[-1], _as_np(sd[p + key]))]))

        qw = _as_np(sd[p + "attn.q_proj.weight"])
        kw = _as_np(sd[p + "attn.k_proj.weight"])
        vw = _as_np(sd[p + "attn.v_proj.weight"])
        ow = _as_np(sd[p + "attn.out_proj.weight"])
        for h in range(NH):
            sl = slice(h * DH, (h + 1) * DH)
            L["q"][h].load_state_dict_stream(iter([("weight", np.ascontiguousarray(qw[sl]))]))
            L["k"][h].load_state_dict_stream(iter([("weight", np.ascontiguousarray(kw[sl]))]))
            L["v"][h].load_state_dict_stream(iter([("weight", np.ascontiguousarray(vw[sl]))]))
            L["o"][h].load_state_dict_stream(
                iter([("weight", np.ascontiguousarray(ow[:, sl]))]))
        del qw, kw, vw, ow

        def moe_items():
            yield "router.weight", _as_np(sd[p + "moe.router.weight"])
            for e in range(self.E):
                yield ("experts.%d.gate.weight" % e,
                       _as_np(sd[p + "moe.experts.%d.w1.weight" % e]))
                yield ("experts.%d.up.weight" % e,
                       _as_np(sd[p + "moe.experts.%d.w2.weight" % e]))
                yield ("experts.%d.down.weight" % e,
                       _as_np(sd[p + "moe.experts.%d.w3.weight" % e]))
        L["moe"].load_state_dict_stream(moe_items())
        qb = _as_np(sd[p + "moe._qb_bias"]) if (p + "moe._qb_bias") in sd else None
        if qb is not None:
            L["moe"].qb_bias = T.tensor(qb)
            if getattr(L["moe"].qb_bias, "device", "cpu") == "cuda":
                L["moe"].qb_bias.data = None
        self.layers.append(L)
        return L

    def load_all(self, sd, on_layer=None):
        for i in range(self.NL):
            self.load_layer(i, sd)
            # 每层装完把池里的旧块真正归还驱动：流式装载会替换每个参数的 _gpu，
            # 旧块进池但不会被复用（尺寸不同），不回收就会把 3B 权重放大 3 倍。
            if getattr(self.layers[0]["ln1"].weight, "device", "cpu") == "cuda":
                _cuda().empty_cache()
            if on_layer:
                on_layer(i)
        return self

    # ---------- 参数与优化器 ----------
    def modules_of(self, i):
        L = self.layers[i]
        return [L["ln1"], L["ln2"], L["moe"]] + L["q"] + L["k"] + L["v"] + L["o"]

    def params_of(self, i):
        out = []
        for m in self.modules_of(i):
            out += [p for p in m.parameters() if getattr(p, "requires_grad", False)]
        return out

    def make_optimizers(self, lr=1e-5, **kw):
        return [T.optim.AdamW(self.params_of(i), lr=lr, **kw) for i in range(self.NL)]

    # ---------- 前向 ----------
    def fwd(self, x, i, cos, sin):
        L = self.layers[i]
        h = L["ln1"](x)
        acc = None
        inv = 1.0 / math.sqrt(self.DH)
        for hh in range(self.NH):
            qh = T._rope_composed(L["q"][hh](h), cos, sin)
            kh = T._rope_composed(L["k"][hh](h), cos, sin)
            vh = L["v"][hh](h)
            pr = T.functional.softmax((qh @ kh.T) * inv, axis=-1)
            part = L["o"][hh](pr @ vh)
            acc = part if acc is None else acc + part
        y = x + acc
        return y + L["moe"](L["ln2"](y))

    def forward_all(self, x0, cos, sin, keep=None):
        """不建图地跑完所有层，缓存每层输入。`keep=lambda i,x: None` 可用于逐步落盘。"""
        xs = [np.asarray(x0, np.float32) if not isinstance(x0, np.ndarray) else x0]
        for i in range(self.NL):
            y = self.fwd(T.tensor(xs[-1]), i, cos, sin)
            arr = np.asarray(_np(y), np.float32)
            xs.append(arr)
            if keep:
                keep(i, arr)
        return xs

    # ---------- 逐层反向 + step ----------
    def backward_and_step(self, xs, g_out, cos, sin, opts=None):
        """从最后一层往回：重算该层前向 → 对 (y·g) 求和反向 → 拿到该层梯度与上游梯度 → step。"""
        g = np.asarray(g_out, np.float32)
        stats = []
        for i in range(self.NL - 1, -1, -1):
            x_in = T.tensor(xs[i], requires_grad=True)
            y = self.fwd(x_in, i, cos, sin)
            gt = T.tensor(g)
            (y * gt).sum().backward()
            n_grad = sum(1 for p in self.params_of(i) if getattr(p, "grad", None) is not None)
            if x_in.grad is not None:
                g = np.asarray(_np(x_in.grad), np.float32)
            if opts is not None:
                opts[i].step()
                opts[i].zero_grad()
            for p in self.params_of(i):
                p.grad = None
            stats.append(dict(layer=i, with_grad=n_grad))
        return g, stats

    # ---------- LM head（tied embed + CE）----------
    def attach_head(self, sd, embed_key="token_embedding.weight", norm_key="final_norm.weight"):
        """挂上 final_norm 与 tied 输出权重（V2.7 没有独立 lm_head）。"""
        self.final_norm = RMSNorm(self.D)
        self.final_norm.load_state_dict_stream(iter([("weight", _as_np(sd[norm_key]))]))
        self.head = Linear(self.D, int(sd[embed_key].shape[0]), bias=False)
        self.head.load_state_dict_stream(iter([("weight", _as_np(sd[embed_key]))]))
        return self

    def head_params(self):
        return [p for p in (self.final_norm.weight, self.head.weight)
                if getattr(p, "requires_grad", False)]

    def loss_and_head_backward(self, x_last, targets, cos, sin):
        """算 CE loss，反传到第 0..N-1 层的上游梯度，并返回 (loss, 上游梯度 g)。

        注意：这里顺便把 head（final_norm/embed）的梯度算出来，调用方 step 完要清。
        """
        x_t = T.tensor(np.asarray(x_last, np.float32), requires_grad=True)
        logits = self.head(self.final_norm(x_t))
        loss = T.functional.cross_entropy(logits, np.asarray(targets, np.int64))
        loss.backward()
        g = np.asarray(_np(x_t.grad), np.float32) if x_t.grad is not None else None
        return loss, g

    def head_grads(self):
        return [getattr(p, "grad", None) for p in self.head_params()]
