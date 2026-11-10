# -*- coding: utf-8 -*-
"""PyT → ORV 迁移覆盖面（0.9.21）：更多层可转、`nn.Conv1d/2d` 就位、失败时信息准确。

判据：
1. `from_torch` 能转 Linear/Embedding/LayerNorm/RMSNorm/Conv1d/Conv2d/Dropout/
   BatchNorm1d/2d/MaxPool1d/2d/AvgPool1d/2d，且**参数逐位相同**；
2. `T.nn.Conv1d` / `T.nn.Conv2d` 存在（PyT 用户就是在这个命名空间里找）；
3. 不支持的自定义 Module 抛 `InteropError`，且报错里**列出已支持类型**、且不再声称"没有 LayerNorm"。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train.interop import InteropError, from_torch

torch = pytest.importorskip("torch")
tn = torch.nn


def _np(x):
    return np.asarray(x.detach().cpu().numpy(), np.float32)


def _param(t):
    return np.asarray(t.to_numpy() if hasattr(t, "to_numpy") else t, np.float32)


def test_nn_namespace_has_conv_modules():
    """PyT 用户在 nn 下找 Conv1d/Conv2d；缺了就得改代码，等于没对齐。"""
    assert hasattr(T.nn, "Conv1d"), "T.nn 缺 Conv1d"
    assert hasattr(T.nn, "Conv2d"), "T.nn 缺 Conv2d"
    # 与顶层是同一个对象（不是副本），避免两处行为分叉
    assert T.nn.Conv2d is T.Conv2d
    assert T.nn.Conv1d is T.Conv1d


@pytest.mark.parametrize("make,attr", [
    (lambda: tn.Linear(8, 4), "weight"),
    (lambda: tn.Embedding(10, 6), "weight"),
    (lambda: tn.LayerNorm(8), "weight"),
    (lambda: tn.Conv1d(4, 6, 3, padding=1), "weight"),
    (lambda: tn.Conv2d(3, 5, 3, padding=1), "weight"),
])
def test_parametric_layers_copy_weights_exactly(make, attr):
    m = make()
    o = from_torch(m)
    assert np.array_equal(_param(getattr(o, attr)), _np(getattr(m, attr))), type(m).__name__


def test_dropout_and_pools_convert():
    assert from_torch(tn.Dropout(0.25)).p == 0.25
    for m in (tn.MaxPool1d(2), tn.MaxPool2d(2), tn.AvgPool1d(2), tn.AvgPool2d(2)):
        o = from_torch(m)
        assert type(o).__name__ == type(m).__name__


def test_batchnorm_keeps_running_stats_and_affine_weights():
    m = tn.BatchNorm2d(4)
    with torch.no_grad():
        m.running_mean.fill_(0.5)
        m.running_var.fill_(2.0)
        m.weight.fill_(1.5)
        m.bias.fill_(-0.25)
    o = from_torch(m)
    assert np.array_equal(_param(o.weight), _np(m.weight))
    assert np.array_equal(_param(o.bias), _np(m.bias))
    assert np.allclose(np.asarray(o.running_mean, np.float32), _np(m.running_mean))


def test_unsupported_module_error_is_accurate():
    class Custom(tn.Module):
        def forward(self, x):
            return x

    with pytest.raises(InteropError) as ei:
        from_torch(Custom())
    msg = str(ei.value)
    assert "Linear" in msg and "Conv2d" in msg, "报错必须列出已支持类型"
    assert "没有 LayerNorm" not in msg, "报错信息里不应再有'ORV 没有 LayerNorm'这种过时说法"
