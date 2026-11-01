# -*- coding: utf-8 -*-
"""`Embedding(weight=)`：直接挂已有参数，避免白分配一张随机词表。

实测背景（2026-11-01，V100）：`Qwen25.__init__` 先建 `Embedding(V, D)`（0.5B 模型
即 545 MB 随机表），下一行立刻把 weight 换成共享 Parameter —— 白分配，
且会留在分配器缓存里。消费级 8/12 GB 卡上这种浪费是致命的。
"""
import numpy as np
import pytest

import tgai_go.train as T


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


def test_embedding_with_weight_shares_the_same_parameter():
    w = T.nn.Parameter(np.arange(12, dtype=np.float32).reshape(4, 3))
    emb = T.nn.Embedding(4, 3, weight=w)
    assert emb.weight is w, "必须共享同一个 Parameter，而不是复制"


def test_embedding_with_weight_gathers_from_it():
    w = T.nn.Parameter(np.arange(12, dtype=np.float32).reshape(4, 3))
    emb = T.nn.Embedding(4, 3, weight=w)
    out = np.asarray(emb(np.array([0, 2], np.int64)).to_numpy(), np.float64)
    assert out.shape == (2, 3)
    assert np.allclose(out, [[0, 1, 2], [6, 7, 8]])


def test_embedding_with_wrong_shape_weight_raises_with_fix():
    w = T.nn.Parameter(np.zeros((5, 3), np.float32))
    with pytest.raises(ValueError) as ei:
        T.nn.Embedding(4, 3, weight=w)
    msg = str(ei.value)
    assert "形状" in msg and "修法" in msg and "(5, 3)" in msg


def test_plain_embedding_still_random_inits():
    emb = T.nn.Embedding(4, 3)
    assert tuple(int(x) for x in emb.weight.shape) == (4, 3)
    assert float(np.abs(np.asarray(emb.weight.to_numpy(), np.float64)).max()) > 0
