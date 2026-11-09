# -*- coding: utf-8 -*-
"""与 torch.nn 的权重级互操作（阶段 E 判据）。

判据（E2）：
  * 同一权重、同一输入，两侧 logits 相对误差 ≤1e-5；
  * `to_torch(from_torch(m))` 的参数**逐位相同**；
  * state_dict 名字映射保持原样；
  * 不支持的类型给出明确报错（含"修法"）。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402

torch = pytest.importorskip("torch")
tn = torch.nn


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def test_sequential_logits_match_torch():
    torch.manual_seed(0)
    m = tn.Sequential(tn.Linear(8, 16), tn.RMSNorm(16, eps=1e-6), tn.Linear(16, 4))
    x = np.random.default_rng(0).normal(0, 1, (5, 8)).astype(np.float32)

    ref = m(torch.from_numpy(x)).detach().numpy()
    ours = np.asarray(T.from_torch(m)(T.tensor(x)).data, np.float32)
    assert _rel(ours, ref) <= 1e-5, _rel(ours, ref)


def test_embedding_leaf_matches_torch():
    torch.manual_seed(1)
    m = tn.Embedding(11, 6)
    ids = np.array([0, 3, 10, 3], np.int64)
    ref = m(torch.from_numpy(ids)).detach().numpy()
    ours = np.asarray(T.from_torch(m)(T.tensor(ids)).data, np.float32)
    assert _rel(ours, ref) <= 1e-5, _rel(ours, ref)


def test_conv2d_leaf_matches_torch():
    torch.manual_seed(2)
    m = tn.Conv2d(3, 5, 3, stride=1, padding=1)
    x = np.random.default_rng(1).normal(0, 1, (2, 3, 7, 7)).astype(np.float32)
    ref = m(torch.from_numpy(x)).detach().numpy()
    ours = np.asarray(T.from_torch(m)(T.tensor(x)).data, np.float32)
    assert _rel(ours, ref) <= 1e-5, _rel(ours, ref)


def test_round_trip_params_are_bit_identical():
    torch.manual_seed(3)
    m = tn.Sequential(tn.Linear(6, 5), tn.RMSNorm(5, eps=1e-6), tn.Linear(5, 2, bias=False))
    back = T.to_torch(T.from_torch(m))
    a = m.state_dict()
    b = back.state_dict()
    assert set(a.keys()) == set(b.keys())
    for k in a:
        x = a[k].detach().numpy()
        y = b[k].detach().numpy()
        assert x.shape == y.shape
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32)), k


def test_state_dict_names_preserved_both_ways():
    torch.manual_seed(4)
    m = tn.Sequential(tn.Linear(3, 4), tn.Linear(4, 2))
    sd = T.state_dict_from_torch(m)
    assert list(sd.keys()) == ["0.weight", "0.bias", "1.weight", "1.bias"]
    ours = T.from_torch(m)
    back = T.state_dict_to_torch(ours)
    assert sorted(back.keys()) == sorted(sd.keys())
    for k in sd:
        assert np.array_equal(np.asarray(back[k].numpy()).view(np.uint32), sd[k].view(np.uint32)), k


def test_unsupported_module_raises_with_hint():
    class Custom(tn.Module):
        def forward(self, x):
            return x

    with pytest.raises(T.InteropError, match="修法"):
        T.from_torch(Custom())


def test_layernorm_is_now_supported():
    """LayerNorm 缺口已补：应当能转换且数值一致（不再是报错路径）。"""
    torch.manual_seed(9)
    m = tn.LayerNorm(8, eps=1e-5)
    ours = T.from_torch(m)
    x = np.random.default_rng(3).normal(0, 1, (3, 8)).astype(np.float32)
    assert _rel(np.asarray(ours(T.tensor(x)).data, np.float32),
                m(torch.from_numpy(x)).detach().numpy()) <= 1e-5
