# -*- coding: utf-8 -*-
"""存储精度：float32 / float16 / bfloat16 的往返与检查点读写（纯 CPU，CI 可跑）。"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

from tgai_go.train import dtypes as D  # noqa: E402


def test_float32_roundtrip_is_exact():
    a = np.random.default_rng(0).normal(size=(64, 32)).astype(np.float32)
    assert np.array_equal(D.from_storage(D.to_storage(a, "float32"), "float32"), a)


def test_float16_roundtrip_matches_numpy_cast():
    a = np.random.default_rng(1).normal(size=(64, 32)).astype(np.float32)
    got = D.from_storage(D.to_storage(a, "float16"), "float16")
    assert np.array_equal(got, a.astype(np.float16).astype(np.float32))


def test_bfloat16_keeps_float32_exponent_range():
    # bf16 的关键优势：指数范围与 fp32 相同，1e30 / 1e-30 都不会像 fp16 那样溢出/下溢
    a = np.array([1e30, 1e-30, 3.14159265, -2.5], dtype=np.float32)
    got = D.from_storage(D.to_storage(a, "bfloat16"), "bfloat16")
    rel = np.abs(got - a) / np.abs(a)
    assert rel.max() < 2 ** -8, "bf16 相对误差应 < 2^-8，实测 %.3e" % rel.max()
    assert np.isfinite(got).all()


def test_bfloat16_has_only_8_mantissa_bits():
    a = np.array([1.0 + 2 ** -9, 1.0 + 2 ** -7], dtype=np.float32)
    got = D.from_storage(D.to_storage(a, "bfloat16"), "bfloat16")
    # 1+2^-9 应被舍入到 1.0（bf16 只有 7 位显式尾数），1+2^-7 应保留
    assert got[0] == 1.0
    assert got[1] == a[1]


def test_save_load_keeps_dtype_and_config(tmp_path):
    w = {"a": np.arange(12, dtype=np.float32).reshape(3, 4) / 7.0,
         "b": np.ones((5,), dtype=np.float32) * 1e-3}
    for dt, tol in (("float32", 0.0), ("float16", 2e-3), ("bfloat16", 1e-2)):
        p = str(tmp_path / ("w_%s.npz" % dt))
        D.save_weights(p, w, dtype=dt, config={"D": 4})
        back, cfg = D.load_weights(p)
        assert cfg == {"D": 4}
        for k in w:
            err = np.abs(back[k] - w[k]).max() / max(np.abs(w[k]).max(), 1e-30)
            if dt == "float32":
                assert err == 0.0, "fp32 必须逐位无损"
            else:
                assert err <= tol, "%s/%s 相对误差 %.3e > %.3e" % (dt, k, err, tol)


def test_unsupported_dtype_message_says_how_to_fix():
    with pytest.raises(ValueError) as ei:
        D.to_storage(np.zeros(4, np.float32), "float8")
    msg = str(ei.value)
    assert "float8" in msg and "修法" in msg


def test_load_weights_missing_file_message():
    with pytest.raises(FileNotFoundError) as ei:
        D.load_weights("no_such_ckpt_12345.npz")
    assert "修法" in str(ei.value)
