# -*- coding: utf-8 -*-
"""错误信息质量标准：算子名 + 形状 + 修法。

目标里写死了这条：「错误信息：每条都写清『哪个算子 / 什么形状 / 怎么修』」。
但"写清了"没法靠肉眼保证，所以用测试钉住三条：

1. 消息里出现**算子名**（`linear:` / `RMSNorm:` / `cross_entropy:` …）；
2. 消息里出现**具体形状或数值**（至少一个数字 —— 只说"形状不匹配"等于没说）；
3. 消息里出现**修法**二字，后面跟着可执行建议。

这样以后谁把异常改成 `raise ValueError("invalid shape")`，CI 立刻红。

覆盖范围：训练侧最常踩的六类形状/取值错误。
"""
import numpy as np
import pytest

from tgai_go.train import nn_ops as O
import tgai_go.train as T


def _msg(fn):
    """跑一次必定失败的调用，返回异常消息；没抛异常则测试失败。"""
    with pytest.raises(Exception) as ei:              # noqa: PT011
        fn()
    return str(ei.value)


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


CASES = {
    # name: (触发函数, 必须出现的算子名)
    "linear 形状不符": (
        lambda: O.linear(T.tensor(np.zeros((2, 3), np.float32)),
                         T.tensor(np.zeros((4, 5), np.float32))), "linear"),
    "linear weight 非二维": (
        lambda: O.linear(T.tensor(np.zeros((2, 3), np.float32)),
                         T.tensor(np.zeros(3, np.float32))), "linear"),
    "functional rms_norm 维度不符": (
        lambda: O.rms_norm(T.tensor(np.zeros((3, 4), np.float32)),
                           T.tensor(np.ones(7, np.float32))), "rms_norm"),
    "RMSNorm 模块维度不符": (
        lambda: T.nn.RMSNorm(7)(T.tensor(np.zeros((3, 4), np.float32))), "RMSNorm"),
    "Embedding 越界": (
        lambda: T.nn.Embedding(5, 4)(T.tensor(np.array([[9, 1]], np.int64))), "Embedding"),
    "cross_entropy 目标越界": (
        lambda: O.cross_entropy(T.tensor(np.zeros((2, 5), np.float32)),
                                T.tensor(np.array([1, 9], np.int64))), "cross_entropy"),
    "cross_entropy logits 非二维": (
        lambda: O.cross_entropy(T.tensor(np.zeros((2, 3, 5), np.float32)),
                                T.tensor(np.array([1, 2], np.int64))), "cross_entropy"),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_message_has_op_shape_and_fix(name):
    fn, op = CASES[name]
    msg = _msg(fn)
    assert op in msg, "%s：消息里没有算子名 %r\n实际：%s" % (name, op, msg)
    assert any(ch.isdigit() for ch in msg), \
        "%s：消息里没有任何具体形状/数值（只说'不匹配'等于没说）\n实际：%s" % (name, msg)
    assert "修法" in msg, "%s：消息里没有『修法』\n实际：%s" % (name, msg)


def test_cross_entropy_accepts_tensor_target():
    """标签传 Tensor 是常见写法，不应报 int() argument must be ... not 'Tensor'。

    这里直接给合法标签，验证能被接受（曾经的失败信息与"标签越界"完全无关）。
    """
    logits = T.tensor(np.random.rand(2, 5).astype(np.float32))
    target = T.tensor(np.array([1, 2], np.int64))
    loss = O.cross_entropy(logits, target)
    assert np.isfinite(float(loss.to_numpy()))


def test_error_messages_are_not_bare_numpy():
    """不该把 numpy 的裸异常直接甩给用户（那种消息里没有算子名）。"""
    bare = [
        "operands could not be broadcast together",
        "index 9 is out of bounds for axis",
        "cannot reshape array",
    ]
    for name, (fn, op) in CASES.items():
        msg = _msg(fn)
        for b in bare:
            assert b not in msg, "%s：仍是 numpy 裸异常：%s" % (name, msg)
