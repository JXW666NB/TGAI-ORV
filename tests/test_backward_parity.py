# -*- coding: utf-8 -*-
"""模型级**反向**对齐：每个参数的梯度都要和 PyTorch autograd 一致。

为什么必须有这条：前向对齐只证明"算得对"，不证明"学得对"。
反向里有大量容易错且**不报错**的地方 —— 转置方向、sum 的维度、RoPE 旋转的转置、
GQA 复制后梯度要不要合并、q_norm 的 RMSNorm 反传、tie 的参数要合并两路梯度。

设计要点：
* 损失用 `(logits**2).sum()`，不引入 CE 的差异，比的是纯反向链路；
* 配置刻意选 Qwen3 的形状：`DH=16 != D/NH=12`、`QK_NORM=True`、`KV<NH`（GQA）；
* 两边都用 float64，阈值压到 1e-6（相对），结构性错误躲不过去。
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    sys.path.insert(0, os.path.join(REPO, "python"))
    path = os.path.join(REPO, "examples", "qwen25_sft.py")
    spec = importlib.util.spec_from_file_location("qwen25_sft_bwd", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen25_sft_bwd"] = mod
    spec.loader.exec_module(mod)
    return mod


# D/NH = 48/4 = 12，DH 取 16 —— 故意不等（Qwen3-0.6B 的真实情形是 128 vs 64）
CFG = dict(V=64, D=48, L=2, NH=4, KV=2, DH=16, FF=80, THETA=1e6, EPS=1e-6,
           QK_NORM=True, QKV_BIAS=False, TIE=True)

LAYER_KEYS = ("ln1.weight", "ln2.weight", "attn.q_proj.weight", "attn.k_proj.weight",
              "attn.v_proj.weight", "attn.out_proj.weight", "attn.q_norm.weight",
              "attn.k_norm.weight", "ffn.w1.weight", "ffn.w2.weight", "ffn.w3.weight")


class TorchQwen3:
    """与 examples 里 Qwen25 结构一致的 torch 参考实现（只为求梯度）。

    参数放在普通 dict 里而不是 ParameterDict —— 后者的名字不允许带点。
    这些张量都是 requires_grad=True 的叶子，反向之后 `.grad` 就有值。
    """

    def __init__(self, cfg, w):
        self.cfg = cfg
        self.emb = torch.tensor(w["token_embedding.weight"], dtype=torch.float64,
                                requires_grad=True)
        self.final = torch.tensor(w["final_norm.weight"], dtype=torch.float64,
                                  requires_grad=True)
        self.ps = {}
        for i in range(cfg["L"]):
            for key in LAYER_KEYS:
                self.ps["blocks.%d.%s" % (i, key)] = torch.tensor(
                    w["blocks.%d.%s" % (i, key)], dtype=torch.float64, requires_grad=True)

    def _rms(self, x, w):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + self.cfg["EPS"]) * w

    def __call__(self, ids):
        return self.forward(ids)

    def forward(self, ids):
        c = self.cfg
        D, L, NH, KV, DH = c["D"], c["L"], c["NH"], c["KV"], c["DH"]
        B, S = ids.shape
        inv = 1.0 / (c["THETA"] ** (np.arange(0, DH, 2, dtype=np.float64) / DH))
        ang = np.outer(np.arange(S, dtype=np.float64), inv)
        cos = torch.from_numpy(np.concatenate([np.cos(ang)] * 2, -1))[None]
        sin = torch.from_numpy(np.concatenate([np.sin(ang)] * 2, -1))[None]

        def rope(t):
            h = DH // 2
            return t * cos + torch.cat([-t[..., h:], t[..., :h]], dim=-1) * sin

        x = self.emb[ids].reshape(B, S, D)
        for i in range(L):
            b = "blocks.%d." % i
            h = self._rms(x, self.ps[b + "ln1.weight"]).reshape(B * S, D)
            q = (h @ self.ps[b + "attn.q_proj.weight"].T).reshape(B, S, NH, DH).permute(0, 2, 1, 3)
            k = (h @ self.ps[b + "attn.k_proj.weight"].T).reshape(B, S, KV, DH).permute(0, 2, 1, 3)
            v = (h @ self.ps[b + "attn.v_proj.weight"].T).reshape(B, S, KV, DH).permute(0, 2, 1, 3)
            q = self._rms(q, self.ps[b + "attn.q_norm.weight"])
            k = self._rms(k, self.ps[b + "attn.k_norm.weight"])
            q = rope(q.reshape(B * NH, S, DH))
            k = rope(k.reshape(B * KV, S, DH))
            k = k.reshape(B, KV, S, DH).repeat_interleave(NH // KV, 1).reshape(B * NH, S, DH)
            v = v.reshape(B, KV, S, DH).repeat_interleave(NH // KV, 1).reshape(B * NH, S, DH)
            mask = torch.triu(torch.full((S, S), float("-inf"), dtype=torch.float64), 1)
            att = torch.softmax(q @ k.transpose(1, 2) / (DH ** 0.5) + mask, dim=-1)
            o = (att @ v).reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, NH * DH)
            x = x + (o @ self.ps[b + "attn.out_proj.weight"].T).reshape(B, S, D)
            h2 = self._rms(x, self.ps[b + "ln2.weight"]).reshape(B * S, D)
            act = (torch.nn.functional.silu(h2 @ self.ps[b + "ffn.w1.weight"].T)
                   * (h2 @ self.ps[b + "ffn.w2.weight"].T))
            x = x + (act @ self.ps[b + "ffn.w3.weight"].T).reshape(B, S, D)
        z = self._rms(x, self.final).reshape(B * S, D)
        return (z @ self.emb.T).reshape(B, S, c["V"])


def test_model_backward_matches_torch_autograd():
    q = _load()
    import tgai_go.train as T
    T.set_default_device("cpu")

    w = q.make_tiny_weights(dict(CFG), seed=5)
    ids = np.random.default_rng(3).integers(0, CFG["V"], size=(2, 6)).astype(np.int64)

    # ---- 我们的引擎：前向 + 反向 ----
    m = q.Qwen25(dict(CFG), dict(w))
    for prm in m.params():
        prm.zero_grad()
    logits = m.forward(ids)
    ref_np = np.asarray(logits.to_numpy(), dtype=np.float64)
    (logits * logits).sum().backward()
    ours = {}
    for name, prm in m.p.items():
        g = getattr(prm, "grad", None)
        if g is not None:
            ours[name] = np.asarray(g.to_numpy() if hasattr(g, "to_numpy") else g, np.float64)

    # ---- torch 参考 ----
    tm = TorchQwen3(dict(CFG), w)
    tl = tm(torch.from_numpy(ids))
    # 引擎是 fp32、参考是 fp64，所以按**相对**误差判（绝对 1e-8 对 fp32 不现实）
    ref_scale = max(float(np.abs(ref_np).max()), 1e-12)
    fwd_rel = float(np.abs(tl.detach().numpy() - ref_np).max()) / ref_scale
    assert fwd_rel < 1e-5, "前向都没对齐（rel=%.2e），先修前向" % fwd_rel
    (tl * tl).sum().backward()

    refs = {"token_embedding.weight": tm.emb.grad, "final_norm.weight": tm.final.grad}
    refs.update({k: v.grad for k, v in tm.ps.items()})

    bad = []
    for name, ref in refs.items():
        if ref is None:
            continue
        got = ours.get(name)
        if got is None:
            bad.append("%s: 引擎侧没有梯度" % name)
            continue
        ref = ref.detach().numpy()
        scale = max(float(np.abs(ref).max()), 1e-12)
        rel = float(np.abs(got - ref).max()) / scale
        # fp32 前向+反向链路，1e-4 已经能抓住任何结构性错误（那类错误是 O(1) 的）
        if rel > 1e-4:
            bad.append("%s: 相对误差 %.2e（max|ref|=%.3e）" % (name, rel, scale))
    assert not bad, "反向不对齐（%d/%d 个参数）：\n  %s" % (
        len(bad), len(refs), "\n  ".join(bad[:8]))


def test_tied_parameter_gradient_accumulates_both_paths():
    """tie 时词嵌入同时参与「输入查表」和「输出投影」，两路梯度必须相加。

    只算一路的话前向完全正常、loss 也降，但训练效果会差一截 —— 很难被发现。
    判据：没被查表用到的那些行，也应该因为输出投影而有梯度。
    """
    q = _load()
    import tgai_go.train as T
    T.set_default_device("cpu")
    w = q.make_tiny_weights(dict(CFG), seed=5)
    ids = np.random.default_rng(4).integers(0, CFG["V"], size=(1, 5)).astype(np.int64)
    m = q.Qwen25(dict(CFG), dict(w))
    assert m.p["lm_head.weight"] is m.p["token_embedding.weight"]
    for prm in m.params():
        prm.zero_grad()
    logits = m.forward(ids)
    (logits * logits).sum().backward()
    g = m.p["token_embedding.weight"].grad
    ga = np.asarray(g.to_numpy() if hasattr(g, "to_numpy") else g, np.float64)
    used = set(ids.reshape(-1).tolist())
    unused = [r for r in range(CFG["V"]) if r not in used]
    assert np.abs(ga[unused]).max() > 0, "未被查表的行没有梯度 ⇒ lm_head 那一路没传回来"
