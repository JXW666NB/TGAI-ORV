# -*- coding: utf-8 -*-
"""边界与数值稳定性：这些在"别人拿来训"时最先踩到。

真实场景举例：
* 一个 batch 里的样本**全被 loss mask 掉**（数据里混进没有答案的行）——
  如果 CE 返回 NaN，NaN 会顺着梯度污染整个模型，而日志上只看到 loss=nan；
* 别人的模型 `num_key_value_heads` 与 `num_attention_heads` 不整除（比如 GQA 16/3），
  如果只是 `repeat_interleave(NH // KV)`，会**少头**然后错位 —— 要么傻崩，要么静默算错；
* `head_dim` 取错（Qwen3 的 128 被当成 64）时，最好在构造阶段就报，而不是等 reshape。
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

import tgai_go.train as T
from tgai_go.train import nn_ops as O

F = T.functional          # 包里的 F 命名空间实际挂在 T.functional 上（别写成 from ... import F）


def _load_example():
    path = os.path.join(REPO, "examples", "qwen25_sft.py")
    spec = importlib.util.spec_from_file_location("qwen25_sft_edge", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen25_sft_edge"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


def test_cross_entropy_empty_batch_is_zero_not_nan():
    """空 batch 必须返回有限的 0，**不能是 NaN**（NaN 会毁掉整个训练）。"""
    logits = T.tensor(np.zeros((0, 5), np.float32), requires_grad=True)
    target = T.tensor(np.zeros((0,), np.int64))
    loss = O.cross_entropy(logits, target)
    v = float(np.asarray(loss.to_numpy()))
    assert np.isfinite(v) and v == 0.0, v
    loss.backward()
    g = logits.grad
    ga = np.asarray(g.to_numpy() if hasattr(g, "to_numpy") else g)
    assert ga.shape == (0, 5)
    assert np.isfinite(ga).all()


def test_cross_entropy_all_ignored_is_zero():
    """整批标签都是 -100（没有答案）时，同样要给 0 和零梯度。"""
    logits = T.tensor(np.random.randn(3, 7).astype(np.float32), requires_grad=True)
    target = T.tensor(np.full((3,), -100, np.int64))
    loss = F.cross_entropy(logits, target)
    v = float(np.asarray(loss.to_numpy()).reshape(-1)[0])
    assert np.isfinite(v) and v == 0.0, v


def test_cross_entropy_extreme_logits_stay_finite():
    """logits 到 1e4 量级也不能溢出（减最大值之后再 exp）。"""
    lg = T.tensor(np.array([[1e4, 0.0, -1e4], [0.0, 1e4, 0.0]], np.float32))
    v = float(np.asarray(O.cross_entropy(lg, T.tensor(np.array([0, 1], np.int64))).to_numpy()))
    assert np.isfinite(v), v
    assert abs(v) < 1.0, v          # 正确类别占绝对优势 ⇒ loss 接近 0


def test_model_rejects_gqa_not_divisible():
    """NH=6, KV=4 不整除 —— 必须在构造阶段报，并说清修法。"""
    q = _load_example()
    cfg = dict(V=32, D=48, L=1, NH=6, KV=4, DH=8, FF=64,
               THETA=1e6, EPS=1e-6, QK_NORM=False, QKV_BIAS=False, TIE=True)
    w = q.make_tiny_weights(dict(cfg), seed=0)
    with pytest.raises(ValueError) as ei:
        q.Qwen25(dict(cfg), w)
    msg = str(ei.value)
    assert "GQA" in msg and "整除" in msg and "修法" in msg, msg


def test_model_rejects_head_dim_mismatch():
    """DH 与 q_proj 行数不自洽时，构造阶段就报，并点名 head_dim。"""
    q = _load_example()
    cfg = dict(V=32, D=48, L=1, NH=4, KV=2, DH=8, FF=64,
               THETA=1e6, EPS=1e-6, QK_NORM=False, QKV_BIAS=False, TIE=True)
    w = q.make_tiny_weights(dict(cfg), seed=0)      # q_proj 行数 = 4*8 = 32
    cfg_bad = dict(cfg)
    cfg_bad["DH"] = 12                              # 4*12 = 48 ≠ 32
    with pytest.raises(ValueError) as ei:
        q.Qwen25(cfg_bad, w)
    msg = str(ei.value)
    assert "head_dim" in msg and "形状不自洽" in msg, msg


def test_model_rejects_odd_head_dim():
    q = _load_example()
    cfg = dict(V=32, D=48, L=1, NH=4, KV=2, DH=9, FF=64,
               THETA=1e6, EPS=1e-6, QK_NORM=False, QKV_BIAS=False, TIE=True)
    w = q.make_tiny_weights(dict(cfg), seed=0)
    with pytest.raises(ValueError) as ei:
        q.Qwen25(dict(cfg), w)
    assert "正偶数" in str(ei.value)


def test_softmax_single_position_is_finite():
    out = O.log_softmax(T.tensor(np.zeros((1, 1, 4), np.float32)))
    a = np.asarray(out.to_numpy())
    assert a.shape == (1, 1, 4)
    assert np.isfinite(a).all()
    assert np.allclose(np.exp(a).sum(-1), 1.0, atol=1e-6)
