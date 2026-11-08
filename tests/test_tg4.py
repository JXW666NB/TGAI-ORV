# -*- coding: utf-8 -*-
"""TG4（V4 格式）读写往返 —— 纯 CPU，进 CI。"""
import os
import sys

import numpy as np
import pytest

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
        # HASH 是**二进制**分区（sha256），不是 JSON —— 早先这里误调 json_section，
        # 把 sha256 字节当 UTF-8 解，才报出 0xe0 的假 bug。
        assert tg4.SEC_HASH in r.sections
        off, ln = r.sections[tg4.SEC_HASH]
        assert ln == 32 * len(ts)
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

def test_pt_to_v4_roundtrip(tmp_path):
    """S2 第一步：`.pt` → v4 → 读回，fp32 张量必须**逐位相等**。

    注意：这里测的是"格式搬运"是否无损，不测模型语义。
    CI 没装 torch 时自动跳过（importorskip），不掩盖失败。
    """
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    ts = {"token_embedding.weight": rng.normal(0, .02, (16, 8)).astype(np.float32),
          "blocks.0.ln1.weight": rng.normal(0, .02, (8,)).astype(np.float32),
          "blocks.0.moe._qb_bias": np.arange(2, dtype=np.float32)}
    pt = str(tmp_path / "ck.pt")
    torch.save({"model_state_dict": {k: torch.from_numpy(v) for k, v in ts.items()},
                "model_config": {"n_experts": 2}}, pt)

    obj = torch.load(pt, map_location="cpu", weights_only=False)
    sd = {k: v.numpy() for k, v in obj["model_state_dict"].items()}

    cfg = dict(CFG, weight_dtype=tg4.DT_FP32)     # fp32 存：要求逐位相等
    p = str(tmp_path / "ck.tg4")
    tg4.write_tg4(p, sd, cfg, train={"step": 0})
    with tg4.TG4(p) as r:
        assert sorted(r.names()) == sorted(sd)
        for k, v in sd.items():
            got = r.get(k)
            np.testing.assert_array_equal(got, v, err_msg="%s 不是逐位相等" % k)
def test_stream_write(tmp_path):
    """流式写：峰值内存只与单张量有关；读回必须与 write_tg4 等价。"""
    rng = np.random.default_rng(9)
    ts = [("token_embedding.weight", rng.normal(0, .02, (16, 8)).astype(np.float32)),
          ("blocks.0.moe.experts.0.w3.weight", rng.normal(0, .02, (8, 16)).astype(np.float32))]
    p = str(tmp_path / "s.tg4")
    tg4.write_tg4_stream(p, iter(ts), CFG, train={"step": 0}, per_tensor_hash=True)
    with tg4.TG4(p) as r:
        assert sorted(r.names()) == sorted(k for k, _ in ts)
        for k, v in ts:
            exp = ((np.ascontiguousarray(v, np.float32).view(np.uint32) >> 16) << 16).view(np.float32)
            np.testing.assert_array_equal(r.get(k), exp, err_msg=k)
        assert r.json_section(tg4.SEC_TRAIN)["step"] == 0