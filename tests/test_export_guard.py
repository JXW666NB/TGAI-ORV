# -*- coding: utf-8 -*-
"""导出护栏测试：head_dim 与 .TG 格式表达能力。

两个必须拦住的场景：
1. **权重形状与 head_dim 不自洽** —— 会把 RoPE 表和 q_proj 行数一起弄错；
2. **head_dim != d_model/n_heads** —— `.TG` 头里没有 head_dim 字段，
   现有 C 引擎会按 d_model/n_heads 解析，导出的模型能加载但生成全错。

这两类错误都不报错、只在推理时才炸，所以必须在导出时停下。
"""
import numpy as np
import pytest

from tgai_go.train import export as E


def _params(n_heads, n_kv, dh, d_model):
    return {
        "blocks.0.attn.q_proj.weight": np.zeros((n_heads * dh, d_model), np.float32),
        "blocks.0.attn.k_proj.weight": np.zeros((n_kv * dh, d_model), np.float32),
        "blocks.0.attn.v_proj.weight": np.zeros((n_kv * dh, d_model), np.float32),
        "blocks.0.attn.out_proj.weight": np.zeros((d_model, n_heads * dh), np.float32),
    }


def test_collect_accepts_explicit_head_dim():
    """head_dim 显式给出时，RoPE 表按它生成（Qwen3: 128 而不是 64）。"""
    out = E.collect(_params(16, 8, 128, 1024), 1, 1, 1024, 16, 3072, 100, 8,
                    n_kv_heads=8, head_dim=128)
    cos = out["blocks.0.attn.rope.cos"]
    assert tuple(cos.shape) == (1, 1, 8, 128), cos.shape


def test_collect_rejects_shape_mismatch():
    """q_proj 行数对不上 n_heads*head_dim 时必须报错并给修法。"""
    bad = _params(16, 8, 128, 1024)
    bad["blocks.0.attn.q_proj.weight"] = np.zeros((16 * 64, 1024), np.float32)
    with pytest.raises(ValueError) as ei:
        E.collect(bad, 1, 1, 1024, 16, 3072, 100, 8, n_kv_heads=8, head_dim=128)
    msg = str(ei.value)
    assert "形状不符" in msg and "修法" in msg


def test_save_tg_rejects_non_natural_head_dim(tmp_path):
    """`.TG` 表达不了 head_dim != d_model/n_heads，必须拒绝而不是静默写坏文件。"""
    p = str(tmp_path / "x.TG")
    with pytest.raises(ValueError) as ei:
        E.save_tg(p, _params(16, 8, 128, 1024), n_layers=1, d_model=1024, n_heads=16,
                  d_ff=3072, vocab_tokens=["a", "b"], max_seq=8, n_kv_heads=8,
                  head_dim=128)
    msg = str(ei.value)
    assert "head_dim=128" in msg and "修法" in msg
    assert not (tmp_path / "x.TG").exists(), "拒绝时不应留下半个文件"


def test_save_tg_allows_natural_head_dim(tmp_path):
    """head_dim 与 d_model/n_heads 一致时（Qwen2.5）应正常导出。

    ★ 2026-11-13：导出容器已改成 **TG4**（自描述分区 + NAM 名字表 + 内置分词表）。
    这里不只断言"文件存在"，还要断言**容器真的是 TG4**、名字可枚举、分词表随文件走 ——
    否则"导出成功"可能只是写出了个 v3 老容器。
    """
    import struct
    from tgai_go import tg4
    from tgai_go.read_tg import read_tg, close_tg

    p = str(tmp_path / "ok.TG")
    E.save_tg(p, _params(4, 2, 16, 64), n_layers=1, d_model=64, n_heads=4,
              d_ff=128, vocab_tokens=["a", "b"], max_seq=8, n_kv_heads=2,
              head_dim=16)
    assert (tmp_path / "ok.TG").exists()

    with open(p, "rb") as fh:
        magic = struct.unpack("<I", fh.read(4))[0]
    assert magic == tg4.MAGIC, "导出的应是 TG4 容器（magic=%#x），实际 %#x" % (tg4.MAGIC, magic)

    info = read_tg(p)
    try:
        assert info.get("container") == "TG4"
        names = info.get("nam_names") or []
        assert names, "TG4 必须带名字表"
        # `collect()` 除了权重，还会补 RoPE 的 cos/sin 缓冲（名字里没有 "weight"），
        # 所以只要求"有 weight 项 + 都是点分参数名"。
        assert any("weight" in n for n in names), "应含权重项: %r" % (names[:3],)
        assert all("." in n for n in names), "名字应是点分参数名: %r" % (names[:3],)
        assert len(info["tokens"]) == 2, "分词表应随文件一起（%d 个）" % len(info["tokens"])
    finally:
        close_tg(info)
