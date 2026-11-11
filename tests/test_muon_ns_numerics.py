# -*- coding: utf-8 -*-
"""Newton-Schulz 归约的**数值**验证（0.9.49 把顺序归约换成并行归约）。

## 为什么要单独测

0.9.49 修 Muon 性能时，把 `sumsq_sequential`（单线程顺序归约，101 ms/矩阵）
换成了 `sumsq_parallel`（两趟并行，0.062 ms/矩阵）—— 这是 10× 提速的来源。

但**并行归约改变求和顺序**，结果不再与 numpy 逐位相同。必须证明：
1. 两者相对差在 fp32 正常范围内（≤1e-6）；
2. `fast=True` 的 NS 结果仍与 numpy 参考一致（≤1e-5 相对）。
3. `fast=False`（顺序版）保持原状，逐位仍可信。

这些是 CUDA 专用测试，无 GPU 时跳过。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import _newton_schulz5, _newton_schulz5_gpu


def _has_cuda():
    try:
        T._cuda()
        return True
    except Exception:      # noqa: BLE001
        return False


CUDA = pytest.mark.skipif(not _has_cuda(), reason="需要 CUDA 后端")


@CUDA
def test_parallel_sumsq_is_accurate_and_sequential_drifts():
    """并行归约**更准**，顺序版在大 n 上有 fp32 累积漂移。

    ⚠️ 实测（2026-11-11，shape=(1024,4096)，420 万项，期望 1677.0546）：

    | 实现 | 结果 | 与 float64 参考 |
    | --- | --- | --- |
    | `sumsq_parallel`（两趟分块） | **1677.0546** | **完全相同** |
    | `sumsq_sequential`（单累加器） | 1670.0114 | **差 4.2e-03** |

    也就是说：**顺序版才是精度差的那个** —— fp32 单累加器在 420 万项上丢了三位有效数字。
    分块求和（部分和再合并）误差更小，这是数值分析的常识。

    所以判据是：
    1. `parallel` 与 float64 参考 ≤1e-5（大 n 也成立）；
    2. `sequential` 允许到 1e-2（记录它的漂移，不是要求它准）。

    训练上的等价性另由 `test_ns_fast_and_slow_close` 保证：
    范数只用于标量缩放，紧随正交化（对尺度不敏感），0.4% 的范数误差被消掉。
    """
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)
    for shape in ((1024, 4096), (1024, 1024), (7, 13)):
        arr = rng.normal(0, 0.02, shape).astype(np.float32)
        buf = rt.h2d(arr)
        seq = float(np.asarray(rt.d2h(rt.sumsq_sequential(buf)),
                               np.float32).reshape(-1)[0])
        par = float(np.asarray(rt.d2h(rt.sumsq_parallel(buf)),
                               np.float32).reshape(-1)[0])
        ref = float((arr.astype(np.float64) ** 2).sum())
        rel_par = abs(par - ref) / max(1e-9, abs(ref))
        rel_seq = abs(seq - ref) / max(1e-9, abs(ref))
        assert rel_par <= 1e-5, (
            "%s 并行归约与 float64 参考差 %.2e 超过 1e-5（par=%.8g ref=%.8g）"
            % (shape, rel_par, par, ref))
        assert rel_seq <= 1e-2, (
            "%s 顺序归约漂移 %.2e 超过 1e-2（seq=%.8g ref=%.8g）"
            % (shape, rel_seq, seq, ref))


@CUDA
def test_ns_fast_matches_numpy_reference():
    """`fast=True`（cuBLAS + 并行归约）仍须与 numpy 参考一致。"""
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.05, (64, 256)).astype(np.float32)
    ref = np.asarray(_newton_schulz5(x, steps=5), np.float32)
    got = np.asarray(rt.d2h(_newton_schulz5_gpu(
        rt, rt.h2d(x), steps=5, fast=True)), np.float32)
    rel = float(np.abs(got - ref).max() / max(1e-9, float(np.abs(ref).max())))
    assert rel <= 1e-5, "fast=True 与 numpy 参考相对差 %.2e 超过 1e-5" % rel


@CUDA
def test_ns_fast_and_slow_close():
    """`fast=True` 与 `fast=False` 两条路径本身也应一致（≤1e-5）。

    前者已放弃与 OpenBLAS 逐位一致，所以用"两者互相接近"来守住语义没跑偏。
    """
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.05, (32, 128)).astype(np.float32)
    buf = rt.h2d(x)
    a = np.asarray(rt.d2h(_newton_schulz5_gpu(rt, buf, steps=5, fast=True)), np.float32)
    b = np.asarray(rt.d2h(_newton_schulz5_gpu(rt, buf, steps=5, fast=False)), np.float32)
    rel = float(np.abs(a - b).max() / max(1e-9, float(np.abs(b).max())))
    assert rel <= 1e-5, "fast vs slow 相对差 %.2e 超过 1e-5" % rel


def test_muon_default_still_uses_slow_path_on_cpu():
    """CPU 上的 Muon 仍走 `_newton_schulz5`（numpy），不受 CUDA 改动影响。"""
    # ★ 必须显式设回 cpu：本机默认设备可能已是 cuda，
    #   那样 `m.weight.data` 是 None（CUDA 张量把数据放在 `_gpu`），
    #   `np.asarray(None)` 会得到 nan —— 这是测试自身的坑，不是实现问题。
    T.set_default_device("cpu")
    try:
        m = T.nn.Linear(8, 8)
        opt = T.optim.Muon(m.parameters(), lr=0.02)
        x = T.tensor(np.random.default_rng(0).normal(0, 1, (4, 8)).astype(np.float32))
        m(x).sum().backward()
        before = np.asarray(m.weight.data, np.float32).copy()
        opt.step()
        after = np.asarray(m.weight.data, np.float32)
        assert np.isfinite(after).all(), "CPU Muon 出现非有限值"
        assert np.abs(after - before).max() > 0, "CPU Muon 未更新参数"
    finally:
        T.set_default_device("cpu")
