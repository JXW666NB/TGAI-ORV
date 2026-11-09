# -*- coding: utf-8 -*-
"""完整 attention 块的等价性：per-head 循环 vs 全宽投影 + 切片（CPU）。

与 test_fused_attn_math.py 的区别：那个只验证投影/拼接/梯度回拆；这里把
scale、causal mask、softmax、输出投影、多头累加**都包含进来**。

权重约定与 ORV 的 Linear 一致：`[out, in]`
  q/k/v : [DH, DT]    （DT → DH）
  o     : [DT, DH]    （DH → DT）
rope 不参与对照：两种写法都保留"逐头 rope"，它不随投影合并而改变。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402

DT, NH, DH, SEQ = 16, 4, 4, 6
INV = 1.0 / np.sqrt(DH)


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def _mk(seed=0):
    rng = np.random.default_rng(seed)
    w = {}
    for k in ("q", "k", "v"):
        w[k] = [rng.normal(0, 0.2, (DH, DT)).astype(np.float32) for _ in range(NH)]
    w["o"] = [rng.normal(0, 0.2, (DT, DH)).astype(np.float32) for _ in range(NH)]
    x = rng.normal(0, 1, (SEQ, DT)).astype(np.float32)
    return w, x


def _mask():
    return T.tensor(np.triu(np.full((SEQ, SEQ), -1e9, np.float32), k=1))


def test_attention_block_equals_fused_projection():
    w, x = _mk(0)
    mask = _mask()

    xt = T.tensor(x.copy())
    acc = None
    for i in range(NH):
        q = xt @ T.tensor(w["q"][i]).T
        k = xt @ T.tensor(w["k"][i]).T
        v = xt @ T.tensor(w["v"][i]).T
        pr = T.functional.softmax((q @ k.T) * INV + mask, axis=-1)
        part = (pr @ v) @ T.tensor(w["o"][i]).T
        acc = part if acc is None else acc + part
    ref = np.asarray(acc.data, np.float32)

    xf = T.tensor(x.copy())
    Wq = T.tensor(np.concatenate(w["q"], axis=0))
    Wk = T.tensor(np.concatenate(w["k"], axis=0))
    Wv = T.tensor(np.concatenate(w["v"], axis=0))
    Wo = T.tensor(np.concatenate(w["o"], axis=1))       # [DT, NH*DH]
    q_all, k_all, v_all = xf @ Wq.T, xf @ Wk.T, xf @ Wv.T
    acc2 = None
    for i in range(NH):
        sl = slice(i * DH, (i + 1) * DH)
        pr = T.functional.softmax((q_all[:, sl] @ k_all[:, sl].T) * INV + mask, axis=-1)
        part = (pr @ v_all[:, sl]) @ Wo[:, sl].T        # 每头取 o 的对应列
        acc2 = part if acc2 is None else acc2 + part
    got = np.asarray(acc2.data, np.float32)

    assert _rel(got, ref) <= 1e-5, ("块输出", _rel(got, ref))


def test_attention_block_gradients_match():
    w, x = _mk(1)
    mask = _mask()
    rng = np.random.default_rng(9)
    gw = rng.normal(0, 1, (SEQ, DT)).astype(np.float32)

    xt = T.tensor(x.copy(), requires_grad=True)
    wq = [T.tensor(w["q"][i].copy(), requires_grad=True) for i in range(NH)]
    acc = None
    for i in range(NH):
        q = xt @ wq[i].T
        k = xt @ T.tensor(w["k"][i]).T
        v = xt @ T.tensor(w["v"][i]).T
        pr = T.functional.softmax((q @ k.T) * INV + mask, axis=-1)
        part = (pr @ v) @ T.tensor(w["o"][i]).T
        acc = part if acc is None else acc + part
    (acc * T.tensor(gw)).sum().backward()
    g_head = [np.asarray(t.grad.data, np.float32).copy() for t in wq]
    gx_head = np.asarray(xt.grad.data, np.float32).copy()

    xf = T.tensor(x.copy(), requires_grad=True)
    Wq = T.tensor(np.concatenate(w["q"], axis=0), requires_grad=True)
    Wk = T.tensor(np.concatenate(w["k"], axis=0))
    Wv = T.tensor(np.concatenate(w["v"], axis=0))
    Wo = T.tensor(np.concatenate(w["o"], axis=1))
    q_all, k_all, v_all = xf @ Wq.T, xf @ Wk.T, xf @ Wv.T
    acc2 = None
    for i in range(NH):
        sl = slice(i * DH, (i + 1) * DH)
        pr = T.functional.softmax((q_all[:, sl] @ k_all[:, sl].T) * INV + mask, axis=-1)
        part = (pr @ v_all[:, sl]) @ Wo[:, sl].T
        acc2 = part if acc2 is None else acc2 + part
    (acc2 * T.tensor(gw)).sum().backward()

    gq_full = np.asarray(Wq.grad.data, np.float32)
    for i in range(NH):
        r = _rel(gq_full[i * DH:(i + 1) * DH], g_head[i])
        assert r <= 1e-5, ("q 权重梯度 head %d" % i, r)
    gx_full = np.asarray(xf.grad.data, np.float32)
    assert _rel(gx_full, gx_head) <= 1e-5, ("输入梯度", _rel(gx_full, gx_head))
