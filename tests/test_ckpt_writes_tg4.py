# -*- coding: utf-8 -*-
"""训练检查点写出的权重必须是 **TG4** 容器，且自带分词表。

## 为什么

2026-11-09 的决定是"以后统一 v4"，而 2026-11-13 才真正把它落成 TG4
（自描述分区 + NAM 名字表 + 内置分词表）。此前写的是"v3 布局 + version=4"，
那种文件的权重表只有 FNV 哈希、没有名字，做不了按名映射/部分加载。

这条用例用 monkeypatch 替掉训练器与优化器落盘（`iter_named_tensors` /
`save_opt_states`），所以**纯 CPU、进 CI**，但走的是真实的 `save_checkpoint` 代码路径。
"""
import json
import os

import numpy as np
import pytest

FIXTURE = r"D:\tgai_nlp\TGAI GO\tests\fixtures\mini_stream.TG"
CFG = dict(n_layers=1, d_model=16, n_heads=2, head_dim=8, d_ff=32, vocab=32,
           max_seq=16, n_experts=0, n_activated=0, rope_theta=10000,
           pad=0, bos=1, eos=2, unk=3)


def _have_fixture():
    return os.path.isfile(FIXTURE)


@pytest.mark.skipif(not _have_fixture(), reason="需要 TGAI GO 的夹具提供真实 tok_blob")
def test_save_checkpoint_writes_tg4_with_tokenizer(tmp_path, monkeypatch):
    from tgai_go import tg4
    from tgai_go.train import ckpt
    from tgai_go.read_tg import read_tg, close_tg
    from tgai_go.train import opt_fp16mv

    names = ["token_embedding.weight", "head.weight"]

    def fake_iter(_tr, _st, want):
        for n in want:
            yield n, np.zeros((32, 16), np.float32)

    monkeypatch.setattr(ckpt, "iter_named_tensors", fake_iter)
    monkeypatch.setattr(opt_fp16mv, "save_opt_states", lambda opts, d: {})

    d = str(tmp_path / "step_000007")
    ckpt.save_checkpoint(d, None, None, [], None, names, FIXTURE, 7, CFG)

    wpath = os.path.join(d, ckpt.CKPT_WEIGHTS)
    assert os.path.isfile(wpath), "应写出 %s" % ckpt.CKPT_WEIGHTS

    # 容器必须是 TG4：靠 magic 判定，而不是靠扩展名
    with open(wpath, "rb") as fh:
        magic = int.from_bytes(fh.read(4), "little")
    assert magic == tg4.MAGIC, "检查点应是 TG4 容器（magic=%#x），实际 %#x" % (tg4.MAGIC, magic)

    info = read_tg(wpath)
    try:
        assert info.get("container") == "TG4"
        assert info["header"]["version"] == 4
        assert len(info["tokens"]) > 0, "分词表必须随权重一起"
        assert [r["name"] for r in map(lambda n, e: {"name": n}, info["nam_names"], info["table"])] \
            == names
    finally:
        close_tg(info)

    with open(os.path.join(d, ckpt.CKPT_META), encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["step"] == 7 and meta["version"] == 4

    # ★ 组合验证：**检查点写出的 TG4 要能被推理引擎直接加载**。
    #   分开验两半（"写的是 TG4" / "引擎能读 TG4"）不够 —— 这条把整条链走通。
    import sys
    dll = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "python", "tgai_go", "_native", "libtgai_engine.dll")
    if sys.platform == "win32" and os.path.isfile(dll):
        import tgai_go
        m = tgai_go.load(wpath)
        assert m.n_layers == 1 and m.d_model == 16
        assert m.vocab_size == 32
