# -*- coding: utf-8 -*-
"""TG4 的 NAM：二进制记录往返 + **旧 JSON 仍可读**。

## 为什么要有兼容读

2026-11-13 把 NAM 从 JSON 改成二进制定长记录（为了让 C 引擎能读，不必在 C 里写 JSON 解析器）。
改动当时全盘扫过**没有任何真实 TG4 文件**，所以没有迁移负担；但万一别处存在用旧版
`tg4.py` 写出的文件（当初的测试就产出过 `m.tg4`），读端不该直接拒收 ——
`_unpack_nam` 因此按首字节区分：`[` 走 JSON，否则走二进制。
"""
import numpy as np

from tgai_go import tg4


def test_binary_nam_roundtrip_including_unicode_and_shapes():
    meta = [
        dict(name="token_embedding.weight", off=0, dtype=tg4.DT_BF16, shape=(32, 16)),
        dict(name="blocks.0.attn.q.weight", off=1024, dtype=tg4.DT_FP16, shape=(16, 16)),
        dict(name="层.1.偏置", off=2048, dtype=tg4.DT_FP32, shape=(8,)),      # 1 维 + 中文名
        dict(name="blocks.0.ffn.up.weight", off=4096, dtype=tg4.DT_FP16, shape=(64, 32)),
    ]
    buf = tg4._pack_nam(meta)
    got = tg4._unpack_nam(buf)
    assert [r["name"] for r in got] == [m["name"] for m in meta]
    for a, b in zip(got, meta):
        assert a["off"] == b["off"]
        assert a["dtype"] == b["dtype"]
        assert tuple(a["shape"]) == tuple(b["shape"])


def test_old_json_nam_still_parses():
    """旧文件（NAM 是 JSON）必须仍能解析 —— 否则升级后旧 TG4 直接打不开。"""
    import json
    old = json.dumps([
        {"name": "a.weight", "off": 0, "dtype": tg4.DT_BF16, "shape": [4, 4]},
        {"name": "b.weight", "off": 64, "dtype": tg4.DT_FP16, "shape": [8]},
    ]).encode("utf-8")
    got = tg4._unpack_nam(old)
    assert [r["name"] for r in got] == ["a.weight", "b.weight"]
    assert tuple(got[0]["shape"]) == (4, 4)
    assert got[1]["off"] == 64


def test_nam_rejects_more_than_four_dims():
    meta = [dict(name="x", off=0, dtype=tg4.DT_FP32, shape=(2, 2, 2, 2, 2))]
    try:
        tg4._pack_nam(meta)
    except ValueError as exc:
        assert "4 维" in str(exc) and "修法" in str(exc)
    else:
        raise AssertionError("5 维应被拒绝（记录里只有 shape[4]）")
