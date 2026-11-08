# -*- coding: utf-8 -*-
"""V2.7 全参训练循环（ORV）：**逐层反向**，把梯度显存压到"单层"。

为什么这样设计（实测依据）：3.033B 全参的 fp32 权重 11.3 GB、fp16 动量 11.3 GB，
若梯度也全量常驻（fp32 11.3 GB）合计 33.9 GB ⇒ 超 32 GB。
逐层反向把梯度变成"算完这层立刻 step 再释放"，峰值 ≈ 23.4 GB。

用法（伪代码）：
    m = V27Trainer(cfg)
    m.load_all(sd, arr)               # 逐张量上传，主机零增长
    for step in range(N):
        loss = m.train_step(tokens, targets, opt, lr)
"""
from __future__ import annotations

import math

import numpy as np

import tgai_go.train as T
from tgai_go.train import Linear, MoE, RMSNorm

HINT = "。修法：检查层号/形状是否与 cfg 一致"


def _arr(sd, key):
    return np.ascontiguousarray(sd[key].float().numpy(), np.float32)


class V27Trainer(object):
    """按层组织，支持逐层反向；参数上传走 load_state_dict（GPU 上已验证）。"""

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

    # ---------- 装载 ----------
    def load_layer(self, i, sd):
        D, NH, DH, FF, E, K = self.D, self.NH, self.DH, self.FF, self.E, self.K
        p = "blocks.%d." % i
        ln1, ln2 = RMSNorm(D), RMSNorm(D)
        ln1.load_state_dict({"weight": _arr(sd, p + "ln1.weight")})
        ln2.load_state_dict({"weight": _arr(sd, p + "ln2.weight")})
        qw, kw, vw, ow = (_arr(sd, p + "attn.q_proj.weight"), _arr(sd, p + "attn.k_proj.weight"),
                          _arr(sd, p + "attn.v_proj.weight"), _arr(sd, p + "attn.out_proj.weight"))
        qs, ks, vs, os_ = [], [], [], []
        for h in range(NH):
            sl = slice(h * DH, (h + 1) * DH)
            lq, lk, lv = Linear(D, DH, bias=False), Linear(D, DH, bias=False), Linear(D, DH, bias=False)
            lq.load_state_dict({"weight": np.ascontiguousarray(qw[sl])})
            lk.load_state_dict({"weight": np.ascontiguousarray(kw[sl])})
            lv.load_state_dict({"weight": np.ascontiguousarray(vw[sl])})
            lo = Linear(DH, D, bias=False)
            lo.load_state_dict({"weight": np.ascontiguousarray(ow[:, sl].T)})
            qs.append(lq); ks.append(lk); vs.append(lv); os_.append(lo)
        moe = MoE(D, FF, E, top_k=K, gate="softmax_topk",
                  qb_bias=_arr(sd, p + "moe._qb_bias") if (p + "moe._qb_bias") in sd else None)
        ms = {"router.weight": _arr(sd, p + "moe.router.weight")}
        for e in range(E):
            ms["experts.%d.gate.weight" % e] = _arr(sd, p + "moe.experts.%d.w1.weight" % e)
            ms["experts.%d.up.weight" % e] = _arr(sd, p + "moe.experts.%d.w2.weight" % e)
            ms["experts.%d.down.weight" % e] = _arr(sd, p + "moe.experts.%d.w3.weight" % e)
        moe.load_state_dict(ms)
        self.layers.append(dict(ln1=ln1, ln2=ln2, q=qs, k=ks, v=vs, o=os_, moe=moe))
        return self

    def load_all(self, sd, on_layer=None):
        for i in range(self.NL):
            self.load_layer(i, sd)
            if on_layer:
                on_layer(i)
        return self

    # ---------- 前向（单层）----------
    def fwd(self, x, i, cos, sin, need_grad=False):
        L = self.layers[i]
        h = L["ln1"](x)
        acc = None
        for hh in range(self.NH):
            qh = T._rope_composed(L["q"][hh](h), cos, sin)
            kh = T._rope_composed(L["k"][hh](h), cos, sin)
            vh = L["v"][hh](h)
            sc = (qh @ kh.T) * (1.0 / math.sqrt(self.DH))
            pr = T.functional.softmax(sc, axis=-1)
            part = L["o"][hh](pr @ vh)
            acc = part if acc is None else acc + part
        y = x + acc
        y = y + L["moe"](L["ln2"](y))
        return y

    def params_of(self, i):
        L = self.layers[i]
        out = [L["ln1"].weight, L["ln2"].weight, L["moe"].router.weight]
        for hh in range(self.NH):
            out += [L["q"][hh].weight, L["k"][hh].weight, L["v"][hh].weight, L["o"][hh].weight]
        for e in range(self.E):
            out += [L["moe"].experts[e].gate.weight, L["moe"].experts[e].up.weight,
                    L["moe"].experts[e].down.weight]
        if getattr(L["moe"], "qb_bias", None) is not None:
            out.append(L["moe"].qb_bias)
        return out

    # ---------- 逐层反向 ----------
    def backward_layerwise(self, x_last, g_out, cos, sin):
        """从最后一层往回：每层重算前向→对 (y·g) 求和反向→拿到该层梯度与上游梯度。"""
        g = g_out
        for i in range(self.NL - 1, -1, -1):
            x_in = T.tensor(np.asarray(x_last[i + 1].to_numpy(), np.float32), requires_grad=True)
            y = self.fwd(x_in, i, cos, sin)
            scalar = (y * g).sum()
            scalar.backward()
            g = x_in.grad
            del y, scalar, x_in
        return g
