# -*- coding: utf-8 -*-
"""TG4（V4 格式）读写往返 —— 纯 CPU，进 CI。"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go import tg4  # noqa: E402

CFG = dict(n_layers=1, d_model=8, n_heads=2, head_dim=4, d_ff=16, vocab=16,
           max_seq=32, n_kv_heads=0, n_experts=2, n_activated=1, arch=0,
           rope_theta=10000.0, weight_dtype=tg4.DT_BF16)


def _bf(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    return ((x.view(np.uint32) >> 16) << 16).view(np.float32)


def test_roundtrip_enumerable(tmp_path):
    rng = np.random.default_rng(0)
    ts = {"token_embedding.weight": rng.normal(0, .02, (16, 8)).astype(np.float32),
          "blocks.0.moe.experts.0.w1.weight": rng.normal(0, .02, (16, 8)).astype(np.float32),
          "blocks.0.moe._qb_bias": np.arange(2, dtype=np.float32)}
    p = str(tmp_path / "m.tg4")
    tg4.write_tg4(p, ts, CFG, arc={"moe_balance": "both"}, train={"step": 7}, per_tensor_hash=True)
    with tg4.TG4(p) as r:
        assert sorted(r.names()) == sorted(ts), "名字必须可枚举"
        for k, v in ts.items():
            got = r.get(k)
            assert got.shape == v.shape and got.dtype == np.float32
            np.testing.assert_allclose(got, _bf(v), rtol=0, atol=0, err_msg=k)
        assert r.header["n_layers"] == 1 and r.header["weight_dtype"] == tg4.DT_BF16
        assert r.json_section(tg4.SEC_ARC)["moe_balance"] == "both"
        assert r.json_section(tg4.SEC_TRAIN)["step"] == 7
        assert r.json_section(tg4.SEC_HASH) is None       # HASH 是二进制分区
        assert tg4.SEC_HASH in r.sections


def test_unknown_section_is_skipped(tmp_path):
    p = str(tmp_path / "u.tg4")
    tg4.write_tg4(p, {"a": np.zeros((2, 2), np.float32)}, CFG)
    raw = bytearray(open(p, "rb").read())
    # 伪造一个未知 section（类型 999）追加到末尾并登记到表里
    import struct
    off = len(raw)
    raw += b"\x00" * ((-off) % 64) + b"junk"
    new_off = off + ((-off) % 64)
    cnt = struct.unpack_from("<Q", raw, 120)[0]
    struct.pack_into("<IIQQII", raw, 128 + cnt * 32, 999, 0, new_off, 4, 64, 0)
    struct.pack_into("<Q", raw, 120, cnt + 1)
    open(p, "wb").write(bytes(raw))
    with tg4.TG4(p) as r:
        assert r.get("a").shape == (2, 2)
        assert 999 in r.sections, "未知分区应被登记（跳过而不报错）"


def test_missing_tensor_is_loud(tmp_path):
    p = str(tmp_path / "x.tg4")
    tg4.write_tg4(p, {"a": np.zeros((2,), np.float32)}, CFG)
    with tg4.TG4(p) as r:
        try:
            r.get("nope")
        except KeyError as e:
            assert "names()" in str(e)
        else:
            raise AssertionError("取不存在的张量必须报错并给修法")