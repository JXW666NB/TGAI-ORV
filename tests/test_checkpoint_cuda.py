# -*- coding: utf-8 -*-
"""回归测试：`T.checkpoint` 必须真的"前向不建图 + 反向重算"（而不是静默直通）。

为什么要有这条：原实现在 CUDA 上按 `device == "cpu"` 过滤，直接 `return fn(*inputs)`，
**一点显存都不省**，但调用方与文档都以为它在省显存 —— 静默的错误信念比崩溃更危险。
判据（与设备无关）：`fn` 必须被调用 **两次**（前向一次、反向重算一次）。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

import tgai_go.train as T


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


def test_checkpoint_recomputes_in_backward():
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return (x * x).sum()

    x = T.tensor(np.array([1.0, 2.0, 3.0], np.float32), requires_grad=True)
    out = T.checkpoint(f, x)
    assert calls["n"] == 1, "前向应只跑一次"
    out.backward()
    assert calls["n"] == 2, "反向必须重算一次（否则就是静默直通）"
    assert np.allclose(np.asarray(x.grad, np.float64), [2.0, 4.0, 6.0])


def test_checkpoint_output_keeps_input_edge_but_drops_intermediates():
    """检查点输出只连**输入**，不连中间激活。

    2026-11-12 改：原来这里断言 `out._parents == ()`，即"完全断开"。
    那个方向是错的 —— 外层图在段与段之间断开后，反向只能靠嵌套递归往上一层钻，
    于是**一次调用把整条链全部重算**，所有段的激活同时驻留（实测 6 层就 8.47 GiB）。
    PyTorch 的 `CheckpointFunction` 是反过来的：外层图保持连通，
    只把中间激活切断，重算时再用 `detach_variable` 让嵌套反传停在输入处。

    所以现在要守的是两件事：
      * `out._parents` **恰好是**需要梯度的输入（外层图连通）；
      * 中间激活没有被 `out` 引用（不出现第三个节点）。
    """
    def f(x):
        mid = x * 2.0          # 中间激活，不该被 out 引用
        return mid * 3.0

    x = T.tensor(np.array([1.0], np.float32), requires_grad=True)
    out = T.checkpoint(f, x)
    assert out.requires_grad
    assert len(out._parents) == 1, "输出应连到输入，保持外层图连通"
    assert out._parents[0] is x, "连的必须是输入本身，不是中间张量"
    # 数值仍然等价于直接调用
    assert np.allclose(np.asarray(out.to_numpy(), np.float32), np.array([6.0], np.float32))


def test_checkpoint_with_no_grad_input_passes_through():
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return x * 2.0

    x = T.tensor(np.array([1.0], np.float32), requires_grad=False)
    out = T.checkpoint(f, x)
    assert calls["n"] == 1, "没有需要梯度的输入时不该做无用的重算"
    assert float(np.asarray(out.to_numpy(), np.float64).ravel()[0]) == 2.0


@pytest.mark.skipif(not os.environ.get("TG_RUN_CUDA_TESTS"),
                    reason="需要 GPU：设 TG_RUN_CUDA_TESTS=1 在 GPU 机器上跑")
def test_checkpoint_works_on_cuda():
    """在 GPU 机器上验证 CUDA 路径（CI 里跳过）。"""
    T.set_default_device("cuda")
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return (x * x).sum()

    x = T.tensor(np.array([1.0, 2.0, 3.0], np.float32), requires_grad=True)
    out = T.checkpoint(f, x)
    out.backward()
    assert calls["n"] == 2, "CUDA 上也必须重算（这曾是空操作）"
    assert np.allclose(np.asarray(x.grad.to_numpy(), np.float64), [2.0, 4.0, 6.0])
