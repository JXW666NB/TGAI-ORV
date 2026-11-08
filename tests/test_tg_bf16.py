# -*- coding: utf-8 -*-
"""`.TG` 的 BF16 读写往返（纯 CPU，进 CI）。

为什么必须有这条：
V2.7 的 `.TG` 实测 `weight_dtype=6`（BF16）、982 张量里 869 个是 bf16；
而 `read_tg.DT` 原先只有 0/1/2/3 —— 也就是说 **ORV 之前根本读不了 V2.7**。
这里用一个小合成文件锁住：写 bf16 → 读回来 → 值必须按 bf16 精度相等（bf16 可表示值无损）。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go import pack as P            # noqa: E402
from tgai_go import read_tg as R        # noqa: E402


def _mk(path, arrays, dtype):
    wr = P.TGWriter(n_layers=1, d_model=8, n_heads=2, d_ff=16, vocab=4,
                    max_seq=8, n_experts=1, n_activated=1, rope_theta=10000,
                    n_kv_heads=0, weight_dtype=dtype)
    wr.add_weights(dict(arrays))
    wr.set_tokenizer(["a", "b", "c", "d"], merges=None, special_ids=[0, 0, 0, 0])
    wr.write(path)


def _bf16_round(x):
    """numpy 里的 bf16 往返（**就近舍入**，与写端一致），用于算期望值。"""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return (u.astype(np.uint32) << 16).view(np.float32)


def test_bf16_constants_exist():
    assert (P.DT_Q8_0, P.DT_Q4_0, P.DT_BF16) == (4, 5, 6)
    assert R.DT[6][0] == "bf16" and R.DT[6][1] == 2


def test_bf16_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.02, size=(8, 16)).astype(np.float32)
    b = np.zeros(16, np.float32)
    p = str(tmp_path / "m.tg")
    _mk(p, {"blocks.0.mlp.up.weight": w, "blocks.0.mlp.up.bias": b}, P.DT_BF16)

    info = R.read_tg(p)
    assert info["header"]["weight_dtype"] == P.DT_BF16, "头里的 dtype 必须是 6"
    assert all(e["dtype"] == P.DT_BF16 for e in info["table"]), "表项也必须是 6"

    got = R.load_tensor(info, "blocks.0.mlp.up.weight")
    assert got.shape == w.shape
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, _bf16_round(w), rtol=0, atol=0)
    # bf16 只有 **7 位尾数** ⇒ 相对误差上限 2^-7 ≈ 0.78%
    #（fp32 是 2^-23；这里断言的是"截断而非乱码"的量级）
    rel = np.abs(got - w).max() / max(1e-9, np.abs(w).max())
    assert rel <= 2 ** -7 + 1e-6, "相对误差 %.3e 超过 bf16 上限 2^-7" % rel


def test_q4_q8_have_dtype_codes():
    """量化容器码要能被识别（真正的 dequant 由 read_tg 的 dequant_* 处理）。"""
    assert R.DT[4][0] == "q8_0"
    assert R.DT[5][0] == "q4_0"

def test_mmap_streaming(tmp_path):
    """默认必须走 mmap：3B 文件（5.6 GB）整块 read() 会吃光内存。"""
    import mmap as _mm
    from tgai_go import read_tg as R
    rng = np.random.default_rng(1)
    w = rng.normal(0, 0.02, size=(4, 8)).astype(np.float32)
    p = str(tmp_path / "s.tg")
    _mk(p, {"blocks.0.mlp.down.weight": w}, P.DT_BF16)

    info = R.read_tg(p)
    try:
        assert isinstance(info["raw"], _mm.mmap), "默认应走 mmap"
        assert info["mmap"] is True
        got = R.load_tensor(info, "blocks.0.mlp.down.weight")
        np.testing.assert_allclose(got, _bf16_round(w), rtol=0, atol=0)
    finally:
        R.close_tg(info)

    # 显式关掉 mmap 也要能读（回退路径）
    info2 = R.read_tg(p, mmap_mode=False)
    assert info2["mmap"] is False
    np.testing.assert_allclose(R.load_tensor(info2, "blocks.0.mlp.down.weight"),
                               _bf16_round(w), rtol=0, atol=0)
    R.close_tg(info2)