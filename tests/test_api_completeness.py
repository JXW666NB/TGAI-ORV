# -*- coding: utf-8 -*-
"""日常表达式 API 的完备性：写模型时最常用的那些写法必须能用且**梯度对**。

来源：一次可用性探测（写 15 个常见表达式）发现 6 个缺口 ——
`2.0/x`、`x.relu()`、`x.tanh()`、`functional.gelu`、`x.unsqueeze()`、`x > 0`。
这些不是"高级功能"，是新人写第一个模型就会撞上的。

注意探测本身也有坑：上一版用 `hasattr(x, "m") else x` 做兜底，
结果"属性不存在"被当成通过（`masked_fill` 就是这样被误判为 OK 的）。
这里一律**直接调用**，不做兜底。
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tgai_go.train as T


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield


def _np(v):
    """CPU 上 grad 是 numpy 数组，CUDA 上是 Tensor —— 统一取出数值。"""
    return np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v, np.float64)


def _x(requires_grad=True, vals=(-1.5, -0.5, 0.5, 1.5)):
    return T.tensor(np.array([vals], np.float32), requires_grad=requires_grad)


def test_right_division_by_scalar_has_correct_grad():
    """`2.0 / x` 的梯度必须是 -2/x²。"""
    x = _x(vals=(0.5, 1.0, 2.0, 4.0))
    y = 2.0 / x
    assert np.allclose(y.to_numpy(), 2.0 / x.to_numpy(), rtol=1e-6)
    (y * y).sum().backward()          # d/dx sum((2/x)^2) = -8/x^3
    got = _np(x.grad)
    want = -8.0 / (x.to_numpy() ** 3)
    assert np.allclose(got, want, rtol=1e-5), (got, want)


def test_comparisons_return_bool_arrays_usable_as_masks():
    x = _x()
    for expr, want in ((x > 0, [False, False, True, True]),
                       (x >= 0.5, [False, False, True, True]),
                       (x < 0, [True, True, False, False]),
                       (x <= -0.5, [True, True, False, False])):
        assert isinstance(expr, np.ndarray) and expr.dtype == bool
        assert expr.tolist() == [list(map(bool, want))], (expr, want)

    # 掩码要能直接用于 where 与索引
    y = T.where(x > 0, x, x * 0.1)
    assert np.allclose(y.to_numpy(), [[-0.15, -0.05, 0.5, 1.5]])
    assert x.to_numpy()[x > 0].tolist() == [0.5, 1.5]


@pytest.mark.parametrize("name,torch_fn", [
    ("relu", lambda t: torch.relu(t)),
    ("tanh", lambda t: torch.tanh(t)),
    ("gelu", lambda t: torch.nn.functional.gelu(t, approximate="tanh")),
])
def test_activations_match_torch_value_and_grad(name, torch_fn):
    vals = (-2.0, -0.3, 0.0, 0.7, 2.5)
    x = _x(vals=vals)
    y = getattr(x, name)()
    r = torch.tensor([vals], requires_grad=True)
    rt = torch_fn(r)
    assert np.allclose(y.to_numpy(), rt.detach().numpy(), rtol=1e-6, atol=1e-7), name

    (y * y).sum().backward()
    (rt * rt).sum().backward()
    assert np.allclose(_np(x.grad), r.grad.numpy(), rtol=1e-5, atol=1e-6), name


def test_functional_exposes_the_same_three():
    x = _x()
    for name in ("relu", "tanh", "gelu"):
        fn = getattr(T.functional, name, None)
        assert fn is not None, "T.functional 缺 %s（用户第一反应就是找它）" % name
        assert np.allclose(fn(x).to_numpy(), getattr(x, name)().to_numpy(), rtol=1e-6)


def test_masked_fill_zeroes_grad_at_masked_positions():
    x = _x()
    y = x.masked_fill(x < 0, 0.0)
    assert np.allclose(y.to_numpy(), [[0.0, 0.0, 0.5, 1.5]])
    (y * y).sum().backward()
    g = _np(x.grad)
    assert np.allclose(g, [[0.0, 0.0, 1.0, 3.0]]), g   # 2*x 只在未掩码处


def test_masked_fill_shape_mismatch_names_fix():
    x = _x()
    with pytest.raises(ValueError) as ei:
        x.masked_fill(np.zeros((3, 3), bool), 0.0)
    msg = str(ei.value)
    assert "masked_fill" in msg and "修法" in msg


def test_cuda_ops_require_a_cuda_device():
    """relu / abs / clamp / log 在 CUDA 上已走 C 端内核（0.9.16）。

    在没有 CUDA 的机器上，强行把张量标成 cuda 必须**明确报错**（不是静默算错），
    且报错信息要能让人看出是 CUDA/驱动问题。

    ★ 2026-11-12：本用例靠"只在属性上伪装 device"来模拟无卡情形，在有 N 卡的机器上
    内核真能跑，于是等不到异常。CUDA 可用时跳过。
    """
    assert hasattr(x_tensor_cls(), "_relu_cuda"), "relu 的 CUDA 分支应已接上内核"
    # 本机 CUDA 真的可用时，"伪装 device 看它报错"这个场景不存在 ⇒ 跳过。
    # 不可用时（CI 就是这种），照旧用伪装法验证报错清晰 —— 那时的它才有意义。
    import numpy as _np
    import tgai_go.train as _T
    _usable = False
    try:
        _T.set_default_device("cuda")
        _T.tensor(_np.zeros((1,), "float32"))
        _usable = True
    except Exception:                           # noqa: BLE001
        pass
    finally:
        _T.set_default_device("cpu")
    if _usable:
        pytest.skip("本机 CUDA 可用，伪装 device 的用例不适用")
    x = _x(requires_grad=False)
    x.device = "cuda"                                   # 只在属性上伪装，不碰真设备
    with pytest.raises(Exception) as ei:
        x.relu()
    msg = str(ei.value).lower()
    assert ("cuda" in msg) or ("nvcuda" in msg) or ("驱动" in msg) or ("gpu" in msg), msg


def x_tensor_cls():
    """拿 Tensor 类本身（避免在测试里再 import 一堆名字）。"""
    return type(_x())


def test_cpu_only_list_is_empty():
    """守住"没有残留的 CPU-only 算子"这条线：源码里不应再有 _cpu_only 调用。"""
    import io
    import os
    import re
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = io.open(os.path.join(here, "python", "tgai_go", "train", "__init__.py"),
                  encoding="utf-8").read()
    leftovers = sorted(set(re.findall(r'_cpu_only\("([^"]+)"\)', src)))
    assert leftovers == [], "还有 CUDA 上只能走 CPU 的算子：%s" % (leftovers,)
