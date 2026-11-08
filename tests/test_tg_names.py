# -*- coding: utf-8 -*-
"""`.TG` 名字旁挂：生成 → 回读 → 枚举（纯 CPU，进 CI）。

背景：`.TG` 权重表只有 FNV-1a 哈希、不存名字（推理够用，当检查点不够用）。
旁挂 `<model>.TG.names.json` 是纯附加文件，产线引擎不读它 ⇒ 对推理零风险。
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "tools"))

from tgai_go import pack as P              # noqa: E402
from tgai_go import read_tg as R          # noqa: E402
from tgai_go.train import export as E     # noqa: E402


def _arrays():
    rng = np.random.default_rng(3)
    return {
        "token_embedding.weight": rng.normal(0, 0.02, (6, 4)).astype(np.float32),
        "blocks.0.attn.q_proj.weight": rng.normal(0, 0.02, (4, 4)).astype(np.float32),
        "blocks.0.moe.experts.0.w1.weight": rng.normal(0, 0.02, (8, 4)).astype(np.float32),
    }


def test_save_tg_writes_sidecar(tmp_path):
    p = str(tmp_path / "m.TG")
    E.save_tg(p, _arrays(), n_layers=1, d_model=4, n_heads=1, d_ff=8,
              vocab_tokens=["a", "b", "c", "d"], max_seq=8, n_experts=2,
              n_activated=1, weight_dtype=P.DT_BF16)
    sc = R.sidecar_path(p)
    assert os.path.isfile(sc), "save_tg 必须写出旁挂名字文件"
    doc = json.load(open(sc, encoding="utf-8"))
    # save_tg 会**自动补** rope.cos/rope.sin/moe.gate.weight/moe._qb_bias，
    # 所以 3 个原始张量 → 6 个表项；而且这 6 个都应该有名字（不是只名字原始 3 个）
    assert doc["n_tensors"] == doc["n_named"] == 6, (doc["n_tensors"], doc["n_named"])
    names = sorted(t[0] for t in R.list_tensors(p))
    assert names == [
        "blocks.0.attn.q_proj.weight",
        "blocks.0.attn.rope.cos",
        "blocks.0.attn.rope.sin",
        "blocks.0.moe._qb_bias",
        "blocks.0.moe.experts.0.w1.weight",
        "token_embedding.weight",
    ], names


def test_list_tensors_without_sidecar_is_null_not_fake(tmp_path):
    """没有旁挂时，名字必须是 None —— 绝不编造。"""
    p = str(tmp_path / "n.TG")
    wr = P.TGWriter(n_layers=1, d_model=4, n_heads=1, d_ff=8, vocab=4, max_seq=8,
                    n_experts=1, n_activated=1, rope_theta=10000, n_kv_heads=0,
                    weight_dtype=P.DT_FP32)
    wr.add_weights({"x": np.zeros((4, 4), np.float32)})
    wr.set_tokenizer(["a", "b", "c", "d"], merges=None, special_ids=[0, 0, 0, 0])
    wr.write(p)
    rows = R.list_tensors(p)
    assert len(rows) == 1 and rows[0][0] is None


def test_sidecar_size_mismatch_is_loud(tmp_path):
    """.TG 变了但旁挂没更新 ⇒ 必须报错（否则名字会错位，最坏是静默训错）。"""
    p = str(tmp_path / "s.TG")
    E.save_tg(p, _arrays(), n_layers=1, d_model=4, n_heads=1, d_ff=8,
              vocab_tokens=["a", "b", "c", "d"], max_seq=8, n_experts=2,
              n_activated=1, weight_dtype=P.DT_FP32)
    with open(p, "ab") as f:
        f.write(b"\x00" * 16)          # 模拟文件被改动
    try:
        R.load_names(p)
    except ValueError as e:
        assert "不匹配" in str(e)
    else:
        raise AssertionError("文件大小变了却不报错")