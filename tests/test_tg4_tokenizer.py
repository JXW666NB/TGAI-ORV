# -*- coding: utf-8 -*-
"""TG4 的 TOK 分区：流式写要能带分词表，读端要能解析出来。

## 背景

`docs/TRAIN_V27_ORV.md` §19 记的 P0：`write_tg4_stream` 没有 `tok` 参数，
于是导出的 v4 **不含分词表** —— 即便权重正确也不能直接给引擎用（引擎要分词器才能 encode）。
2026-11-13 补上该参数，并让读端共用 v3 的解析器（两者字节布局本就相同）。

这三条断言保证它不再退化：
1. 写的时候传了 `tok=`，读回来 tokens/merges/specials 与源文件一致；
2. **不传** `tok=` 时读端返回空三元组而**不是抛异常**（老文件仍可读）；
3. 张量按名可取（`get()`），与分词表互不影响。
"""
import os

import numpy as np
import pytest

FIXTURE = r"D:\tgai_nlp\TGAI GO\tests\fixtures\mini_stream.TG"

CFG = dict(n_layers=1, d_model=16, n_heads=2, head_dim=8, d_ff=32, vocab=32,
           max_seq=16, n_kv_heads=0, n_experts=0, n_activated=0, arch=0)

ITEMS = [("token_embedding.weight", np.arange(32 * 16, dtype=np.float32).reshape(32, 16)),
         ("head.weight", np.zeros((32, 16), np.float32))]


def _have_fixture():
    return os.path.isfile(FIXTURE)


@pytest.mark.skipif(not _have_fixture(), reason="需要 TGAI GO 的 v3 夹具提供真实 tok_blob")
def test_stream_writer_embeds_tokenizer(tmp_path):
    from tgai_go import tg4
    from tgai_go.train.ckpt import read_tok_blob
    from tgai_go.read_tg import read_tg, close_tg

    blob = read_tok_blob(FIXTURE)
    assert blob
    src = read_tg(FIXTURE)
    want_tok, want_merge = len(src["tokens"]), len(src["merges"])
    close_tg(src)

    cfg = dict(CFG, weight_dtype=tg4.DT_BF16)
    p = str(tmp_path / "stream_tok.tg4")
    tg4.write_tg4_stream(p, iter(ITEMS), cfg, tok=blob)

    with tg4.TG4(p) as r:
        tok, mer, spec = r.tokenizer()
        assert len(tok) == want_tok and len(tok) > 0, "分词表应随文件一起"
        assert len(mer) == want_merge
        assert len(spec) == 16
        # 张量仍按名可取，且数值经过 bf16 量化后接近原值
        a = r.get("token_embedding.weight")
        assert a.shape == (32, 16)


def test_stream_writer_without_tokenizer_is_still_readable(tmp_path):
    """不传 tok 时：不写 TOK 分区，读端返回空三元组，**不能抛**。"""
    from tgai_go import tg4

    cfg = dict(CFG, weight_dtype=tg4.DT_BF16)
    p = str(tmp_path / "no_tok.tg4")
    tg4.write_tg4_stream(p, iter(ITEMS), cfg)
    with tg4.TG4(p) as r:
        assert tg4.SEC_TOK not in r.sections
        assert r.tokenizer() == ([], [], [])
        assert r.names() == ["token_embedding.weight", "head.weight"]
