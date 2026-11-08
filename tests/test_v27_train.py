# -*- coding: utf-8 -*-
"""V2.7 训练循环（逐层反向）的 CPU 回归测试 —— 不碰 GPU，进 CI。

覆盖：
* 流式装载（`load_state_dict_stream`）能把权重按名装进按层组织的模块
* `forward_all` 不建图、输出有限
* `backward_and_step` 每层都能拿到梯度、上游梯度能传、step 后参数确实变了
"""
import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train.v27_train import V27Trainer  # noqa: E402

CFG = dict(d_model=32, n_heads=4, d_ff=64, n_experts=4, n_activated=2,
           n_layers=2, rope_theta=10000.0)


def _rope(seq, dh, theta):
    j = np.arange(0, dh, 2, dtype=np.float32) / dh
    inv = theta ** (-j)
    t = np.arange(seq, dtype=np.float32)[:, None]
    ang = t * inv[None, :]
    ang = np.repeat(ang, 2, axis=1)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def _fake_state(cfg, seed=0):
    rng = np.random.default_rng(seed)
    D, FF, E, NL, NH, DH = cfg["d_model"], cfg["d_ff"], cfg["n_experts"], cfg["n_layers"], cfg["n_heads"], cfg["d_model"] // cfg["n_heads"]
    sd = {}
    for i in range(NL):
        p = "blocks.%d." % i
        sd[p + "ln1.weight"] = rng.normal(0, .02, D).astype(np.float32)
        sd[p + "ln2.weight"] = rng.normal(0, .02, D).astype(np.float32)
        for k in ("q", "k", "v", "out"):
            sd[p + "attn.%s_proj.weight" % k] = rng.normal(0, .02, (D, D)).astype(np.float32)
        sd[p + "moe.router.weight"] = rng.normal(0, .02, (E, D)).astype(np.float32)
        sd[p + "moe._qb_bias"] = rng.normal(0, .1, E).astype(np.float32)
        for e in range(E):
            sd[p + "moe.experts.%d.w1.weight" % e] = rng.normal(0, .02, (FF, D)).astype(np.float32)
            sd[p + "moe.experts.%d.w2.weight" % e] = rng.normal(0, .02, (FF, D)).astype(np.float32)
            sd[p + "moe.experts.%d.w3.weight" % e] = rng.normal(0, .02, (D, FF)).astype(np.float32)
    return sd


def test_forward_backward_step_cpu():
    import tgai_go.train as T

    sd = _fake_state(CFG)
    tr = V27Trainer(CFG).load_all(sd)
    assert len(tr.layers) == CFG["n_layers"]

    TK = 8
    cos_np, sin_np = _rope(TK, CFG["d_model"] // CFG["n_heads"], CFG["rope_theta"])
    cos, sin = T.tensor(cos_np), T.tensor(sin_np)

    x0 = np.random.default_rng(1).normal(0, .1, (TK, CFG["d_model"])).astype(np.float32)
    xs = tr.forward_all(x0, cos, sin)
    assert len(xs) == CFG["n_layers"] + 1
    assert np.isfinite(xs[-1]).all(), "前向输出出现 NaN/Inf"

    before = [np.array(p.data, copy=True) for p in tr.params_of(0)]
    opts = tr.make_optimizers(lr=1e-3)
    g_out = np.random.default_rng(2).normal(0, .01, (TK, CFG["d_model"])).astype(np.float32)
    g, stats = tr.backward_and_step(xs, g_out, cos, sin, opts)

    assert len(stats) == CFG["n_layers"]
    for s in stats:
        assert s["with_grad"] > 0, "层 %d 一个梯度都没拿到" % s["layer"]
    assert np.isfinite(g).all(), "回传的上游梯度出现 NaN/Inf"

    after = [np.array(p.data, copy=True) for p in tr.params_of(0)]
    changed = sum(1 for a, b in zip(before, after) if not np.array_equal(a, b))
    assert changed > 0, "step 之后参数一个都没变"


def test_stream_load_rejects_missing_tensor():
    sd = _fake_state(CFG)
    sd.pop("blocks.0.ln1.weight")
    with pytest.raises(KeyError):
        V27Trainer(CFG).load_all(sd)
def test_lm_head_loss_cpu():
    """真实 CE loss：tied embed 输出 + 交叉熵，梯度能反传到最后一层输入。"""
    import tgai_go.train as T

    cfg = dict(CFG, n_layers=2)
    sd = _fake_state(cfg)
    sd["token_embedding.weight"] = np.random.default_rng(5).normal(0, .02, (40, cfg["d_model"])).astype(np.float32)
    sd["final_norm.weight"] = np.random.default_rng(6).normal(0, .02, cfg["d_model"]).astype(np.float32)
    tr = V27Trainer(cfg).load_all(sd).attach_head(sd)

    TK = 8
    cos_np, sin_np = _rope(TK, cfg["d_model"] // cfg["n_heads"], cfg["rope_theta"])
    cos, sin = T.tensor(cos_np), T.tensor(sin_np)
    x0 = np.random.default_rng(1).normal(0, .1, (TK, cfg["d_model"])).astype(np.float32)
    xs = tr.forward_all(x0, cos, sin)

    targets = np.random.default_rng(2).integers(0, 40, TK)
    loss, g = tr.loss_and_head_backward(xs[-1], targets, cos, sin)
    lv = float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy") else loss))
    assert np.isfinite(lv) and lv > 0, "CE loss 异常：%r" % lv
    assert g is not None and np.isfinite(g).all(), "CE 梯度没回传到最后一层"
    gs = tr.head_grads()
    assert any(x is not None for x in gs), "head 参数没拿到梯度"