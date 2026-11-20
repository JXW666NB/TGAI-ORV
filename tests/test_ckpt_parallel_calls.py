# -*- coding: utf-8 -*-
"""`_checkpoint_layer` 在**一次前向里并列调用 k 次**时不能丢梯度（回归测试）。

## 背景（2026-11-21 实测定位）

`_checkpoint_layer(fn, x)` 的反向里做的是 `sub = fn(x); sub.backward(g)` —— 一次
**整图反向**。它会顺着 `x._parents` 一路走到 embedding，而 `Tensor._backward_cuda`
遍历时会把沿途每个节点的 `_parents/_backward` 拆掉（"反向消费完就拆图引用"那段优化）。
于是第 1 个块的嵌套反传把 `x` 之上的整条链拆空，第 2..k 个块的梯度走到 `x` 就断掉。

修之前的最小复现（GPU）：

```
W(叶子) --matmul--> x --切 k 块--> 每块 checkpoint(fn) --> 相加 --> backward
```
`|dW|` **恒定** 510.40946（k=1/2/4/8 全都一样），而不包 ckpt 的基准是
510 → 752 → 1195（k=4 起覆盖满，封顶）。在 v3 训练里的表现是 `--ce-chunk` k 块时
`gnorm ≈ base/√k`（2 块 15.3866 → 10.7590，4 块 → 7.28，8 块 → 5.12）。

修法：嵌套反传**在 x 处切断**（临时把 x 当叶子），只算"本段自己的参数 + `x.grad`"，
再由外层遍历沿恢复后的 `_parents` 继续往上走。

## 这个测试防什么

1. k 次并列调用后，上游参数的梯度**必须**等于不包 ckpt 的同 k 基准
   （不是"随 k 衰减"）。这条是硬的：它同时覆盖梯度累积、多 chunk、
   "同一组参数被多次调用"等场景。
2. k=1 时两者仍相等 —— 保证正常用法不受影响。

只跑 CUDA：`Tensor._from_gpu` 只有 CUDA 后端有，CPU 上 `_checkpoint_layer` 根本不会
走那条分支（`x.requires_grad` 为 True 时也只是 `fn(x)`），测了等于没测。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import v3_train_cli as CLI


def _has_cuda():
    try:
        T._cuda()
        return True
    except Exception:      # noqa: BLE001
        return False


_D, _ROWS, _CH = 64, 64, 16          # d=64；x 共 64 行；每块 16 行


def _make():
    """W 是叶子参数；z 是常数输入。x = z @ W.T ⇒ **参数在 x 上游**（这是复现前提）。"""
    rt = T._cuda()
    rng = np.random.default_rng(5)
    W = T.Tensor._from_gpu(rt.h2d(rng.normal(0, 0.1, (_D, _D)).astype(np.float32)),
                           requires_grad=True)
    z = T.Tensor._from_gpu(rt.h2d(rng.normal(0, 1, (_ROWS, _D)).astype(np.float32)))
    return W, z


def _gnorm(W):
    g = W.grad
    if g is None:
        return 0.0
    a = np.asarray(g.to_numpy() if hasattr(g, "to_numpy") else g, dtype=np.float64)
    return float(np.sqrt((a * a).sum()))


def _run(k, use_ckpt):
    W, z = _make()
    x = z @ W.transpose(0, 1)
    total = None
    for i in range(k):
        xc = x[i * _CH:(i + 1) * _CH]

        def fn(inp):
            return (inp * inp).sum()

        part = CLI._checkpoint_layer(fn, xc) if use_ckpt else fn(xc)
        total = part if total is None else total + part
    total.backward()
    return _gnorm(W)


@pytest.mark.skipif(not _has_cuda(), reason="需要 CUDA（_from_gpu 只在 CUDA 后端存在）")
def test_checkpoint_parallel_calls_keep_upstream_gradients():
    """k 次并列调用后，上游参数的梯度必须等于不包 ckpt 的基准（不是 base/k）。"""
    T.set_default_device("cuda")
    for k in (1, 2, 4):
        want = _run(k, False)
        got = _run(k, True)
        assert want > 0.0, "基准梯度为 0，测试本身失效（k=%d）" % k
        # 修之前 k=2/4 分别只有 0.679/0.427 倍；留 1% 余量防浮点抖动
        assert abs(got - want) <= 1e-2 * want, (
            "k=%d 并列调用丢了梯度：不包 ckpt |dW|=%.9g，包 ckpt |dW|=%.9g（比值 %.4f）。"
            "修法：`_checkpoint_layer` 的嵌套反传必须在 x 处切断（临时把 x 当叶子），"
            "否则它会越过 x 拆掉外层图，后面的块梯度全丢。"
            % (k, want, got, got / want if want else float("nan")))


@pytest.mark.skipif(not _has_cuda(), reason="需要 CUDA（_from_gpu 只在 CUDA 后端存在）")
def test_checkpoint_single_call_unchanged():
    """k=1（既有用法）必须与不包 ckpt 完全一致。"""
    T.set_default_device("cuda")
    want = _run(1, False)
    got = _run(1, True)
    assert abs(got - want) <= 1e-6 * max(want, 1e-12), \
        "k=1 的 ckpt 路径变了：%.9g vs %.9g" % (want, got)
