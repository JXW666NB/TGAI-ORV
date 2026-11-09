# -*- coding: utf-8 -*-
"""阶段三（第 8 项）：**不依赖 `.tg` 的自定义模型**，ORV 与 torch 逐步 loss 对照。

这里刻意模拟"用户自己的模型"：
  * 只用公开的建模块 API（`nn.Embedding` / `nn.Linear` / `nn.RMSNorm` / `functional.silu`）；
  * 不用 TGAI 的 `.tg` 容器、不用 `V27Trainer`、不读模型文件；
  * 结构、初始化、超参在两侧完全一致，逐步比较 loss。

判据（目标）：**逐 step loss 相对误差 ≤1e-4**。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402

torch = pytest.importorskip("torch")
F = torch.nn.functional

V, D, H, NL, SEQ, STEPS = 32, 16, 4, 2, 12, 8
LR = 1e-3


def _init_arrays(seed=0):
    """两侧共用的初始化（同一份 numpy 数值）。"""
    rng = np.random.default_rng(seed)
    p = {}
    p["emb"] = (rng.normal(0, 0.02, (V, D))).astype(np.float32)
    for i in range(NL):
        p["ln1_%d" % i] = np.ones(D, np.float32)
        p["wq_%d" % i] = (rng.normal(0, 0.1, (D, D))).astype(np.float32)
        p["wk_%d" % i] = (rng.normal(0, 0.1, (D, D))).astype(np.float32)
        p["wv_%d" % i] = (rng.normal(0, 0.1, (D, D))).astype(np.float32)
        p["wo_%d" % i] = (rng.normal(0, 0.1, (D, D))).astype(np.float32)
        p["ln2_%d" % i] = np.ones(D, np.float32)
        p["w1_%d" % i] = (rng.normal(0, 0.1, (4 * D, D))).astype(np.float32)
        p["w2_%d" % i] = (rng.normal(0, 0.1, (D, 4 * D))).astype(np.float32)
    p["fn"] = np.ones(D, np.float32)
    p["head"] = (rng.normal(0, 0.1, (V, D))).astype(np.float32)
    return p


def _causal_mask(Tlen):
    return np.triu(np.full((Tlen, Tlen), -1e9, np.float32), k=1)


class OrvModel(object):
    def __init__(self, p):
        self.p = p
        self.emb = T.nn.Embedding(V, D, weight=T.nn.Parameter(p["emb"].copy()))
        self.ln1 = [T.nn.RMSNorm(D) for _ in range(NL)]
        self.ln2 = [T.nn.RMSNorm(D) for _ in range(NL)]
        self.wq = [T.nn.Linear(D, D) for _ in range(NL)]
        self.wk = [T.nn.Linear(D, D) for _ in range(NL)]
        self.wv = [T.nn.Linear(D, D) for _ in range(NL)]
        self.wo = [T.nn.Linear(D, D) for _ in range(NL)]
        self.w1 = [T.nn.Linear(D, 4 * D) for _ in range(NL)]
        self.w2 = [T.nn.Linear(4 * D, D) for _ in range(NL)]
        # 把共用初始化的数值灌进去
        for i in range(NL):
            self.wq[i].weight.data[...] = p["wq_%d" % i]
            self.wk[i].weight.data[...] = p["wk_%d" % i]
            self.wv[i].weight.data[...] = p["wv_%d" % i]
            self.wo[i].weight.data[...] = p["wo_%d" % i]
            self.w1[i].weight.data[...] = p["w1_%d" % i]
            self.w2[i].weight.data[...] = p["w2_%d" % i]
            np.copyto(np.asarray(self.wq[i].bias.data, np.float32), np.zeros(D, np.float32))
            np.copyto(np.asarray(self.wk[i].bias.data, np.float32), np.zeros(D, np.float32))
            np.copyto(np.asarray(self.wv[i].bias.data, np.float32), np.zeros(D, np.float32))
            np.copyto(np.asarray(self.wo[i].bias.data, np.float32), np.zeros(D, np.float32))
            np.copyto(np.asarray(self.w1[i].bias.data, np.float32), np.zeros(4 * D, np.float32))
            np.copyto(np.asarray(self.w2[i].bias.data, np.float32), np.zeros(D, np.float32))
        self.head = T.nn.Linear(D, V)
        self.head.weight.data[...] = p["head"]
        self.fn = T.nn.RMSNorm(D)                 # 最终 norm 必须是模型成员（否则每步新建、参数不一致）
        np.copyto(np.asarray(self.fn.weight.data, np.float32), p["fn"])
        np.copyto(np.asarray(self.head.bias.data, np.float32), np.zeros(V, np.float32))

    def parameters(self):
        ps = [self.emb.weight, self.head.weight, self.head.bias, self.fn.weight]
        for i in range(NL):
            ps += [self.ln1[i].weight, self.ln2[i].weight,
                   self.wq[i].weight, self.wk[i].weight, self.wv[i].weight, self.wo[i].weight,
                   self.w1[i].weight, self.w2[i].weight,
                   self.wq[i].bias, self.wk[i].bias, self.wv[i].bias, self.wo[i].bias,
                   self.w1[i].bias, self.w2[i].bias]
        return ps

    def __call__(self, ids):
        mask = T.tensor(_causal_mask(len(ids)))
        x = self.emb(T.tensor(np.asarray(ids, np.int64)))
        for i in range(NL):
            h = self.ln1[i](x)
            q = self.wq[i](h)
            k = self.wk[i](h)
            v = self.wv[i](h)
            sc = (q @ k.T) * (1.0 / np.sqrt(D / H))
            pr = T.nn.functional.softmax(sc + mask, axis=-1)
            x = x + self.wo[i](pr @ v)
            h2 = self.ln2[i](x)
            x = x + self.w2[i](T.nn.functional.silu(self.w1[i](h2)))
        return self.head(self.fn(x))


def _torch_model(p):
    import torch.nn as tn

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = tn.Embedding(V, D)
            self.ln1 = tn.ModuleList([tn.RMSNorm(D, eps=1e-6) for _ in range(NL)])
            self.ln2 = tn.ModuleList([tn.RMSNorm(D, eps=1e-6) for _ in range(NL)])
            self.wq = tn.ModuleList([tn.Linear(D, D) for _ in range(NL)])
            self.wk = tn.ModuleList([tn.Linear(D, D) for _ in range(NL)])
            self.wv = tn.ModuleList([tn.Linear(D, D) for _ in range(NL)])
            self.wo = tn.ModuleList([tn.Linear(D, D) for _ in range(NL)])
            self.w1 = tn.ModuleList([tn.Linear(D, 4 * D) for _ in range(NL)])
            self.w2 = tn.ModuleList([tn.Linear(4 * D, D) for _ in range(NL)])
            self.head = tn.Linear(D, V)
            self.fn = tn.RMSNorm(D, eps=1e-6)
            with torch.no_grad():
                self.emb.weight.copy_(torch.from_numpy(p["emb"]))
                for i in range(NL):
                    self.wq[i].weight.copy_(torch.from_numpy(p["wq_%d" % i]))
                    self.wk[i].weight.copy_(torch.from_numpy(p["wk_%d" % i]))
                    self.wv[i].weight.copy_(torch.from_numpy(p["wv_%d" % i]))
                    self.wo[i].weight.copy_(torch.from_numpy(p["wo_%d" % i]))
                    self.w1[i].weight.copy_(torch.from_numpy(p["w1_%d" % i]))
                    self.w2[i].weight.copy_(torch.from_numpy(p["w2_%d" % i]))
                    self.wq[i].bias.zero_(); self.wk[i].bias.zero_()
                    self.wv[i].bias.zero_(); self.wo[i].bias.zero_()
                    self.w1[i].bias.zero_(); self.w2[i].bias.zero_()
                self.head.weight.copy_(torch.from_numpy(p["head"]))
                self.head.bias.zero_()

        def forward(self, ids):
            mask = torch.from_numpy(_causal_mask(len(ids)))
            x = self.emb(ids)
            for i in range(NL):
                h = self.ln1[i](x)
                q, k, v = self.wq[i](h), self.wk[i](h), self.wv[i](h)
                sc = (q @ k.T) * (1.0 / np.sqrt(D / H))
                pr = torch.softmax(sc + mask, dim=-1)
                x = x + self.wo[i](pr @ v)
                h2 = self.ln2[i](x)
                x = x + self.w2[i](F.silu(self.w1[i](h2)))
            return self.head(self.fn(x))

    return M()


def _ids_list(seed=1, n=STEPS):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, V, SEQ + 1) for _ in range(n)]


def test_custom_model_loss_matches_torch_step_by_step():
    p = _init_arrays(0)
    ids_list = _ids_list()

    om = OrvModel(p)
    o_opt = T.optim.AdamW(om.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    orv_losses = []
    for ids in ids_list:
        logits = om(ids[:-1])
        loss = T.nn.functional.cross_entropy(logits, ids[1:])
        orv_losses.append(float(np.asarray(loss.data, np.float32)))
        loss.backward()
        o_opt.step()
        o_opt.zero_grad()

    tm = _torch_model(p)
    t_opt = torch.optim.AdamW(tm.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    torch_losses = []
    for ids in ids_list:
        logits = tm(torch.from_numpy(ids[:-1]).long())
        loss = F.cross_entropy(logits, torch.from_numpy(ids[1:]).long())
        torch_losses.append(float(loss.detach()))
        loss.backward()
        t_opt.step()
        t_opt.zero_grad()

    worst = 0.0
    for i, (a, b) in enumerate(zip(orv_losses, torch_losses)):
        r = abs(a - b) / max(1e-9, abs(b))
        worst = max(worst, r)
        assert r <= 1e-4, "step %d: ORV %.6f vs torch %.6f, 相对 %.2e" % (i, a, b, r)

    print("[custom-model] %d 步 loss 最差相对误差 = %.2e（判据 1e-4）" % (len(orv_losses), worst))
    assert worst <= 1e-4
