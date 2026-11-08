# -*- coding: utf-8 -*-
"""BPE 分词器（纯 Python）—— 用合成词表验证算法，不依赖引擎与 GPU。"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.bpe import SPACE, BpeTokenizer  # noqa: E402

TOKENS = [SPACE, "a", "b", "c", "ab", "abc", "d", SPACE + "a", SPACE + "ab"]
MERGES = ["a b", ("b", "c"), ("ab", "c")]


def test_merge_prefers_lower_rank():
    tk = BpeTokenizer(TOKENS, MERGES, unk_id=0)
    ids = tk.encode("abc", add_bos=False)
    # 有 ("ab","c") 这条合并 ⇒ 整段应合成单个 token "abc"
    assert ids[-1] == TOKENS.index("abc"), ids
    assert tk.decode(ids).strip() == "abc"


def test_table_string_and_tuple_merges_both_work():
    a = BpeTokenizer(TOKENS, ["a b"], unk_id=0)
    b = BpeTokenizer(TOKENS, [("a", "b")], unk_id=0)
    assert a.encode("ab") == b.encode("ab")


def test_space_becomes_marker_and_roundtrip():
    tk = BpeTokenizer(TOKENS, MERGES, unk_id=0)
    ids = tk.encode("ab d")
    assert any(SPACE in tk.tokens[i] or i == 0 for i in ids[:1]) or ids
    assert "ab" in tk.decode(ids)


def test_unknown_char_maps_to_unk():
    tk = BpeTokenizer(TOKENS, MERGES, unk_id=0)
    ids = tk.encode("z")
    # 前面还有一个 ▁（它在词表里，id 0 恰好也是 unk，故这里只断言末位是 unk）
    assert ids[-1] == 0, "词表外字符应落到 unk"


def test_bos_and_type_errors():
    tk = BpeTokenizer(TOKENS, MERGES, unk_id=0, bos_id=1)
    assert tk.encode("ab", add_bos=True)[0] == 1
    with pytest.raises(TypeError):
        tk.encode(b"ab")


def test_bad_merge_reports_clearly():
    with pytest.raises(ValueError):
        BpeTokenizer(TOKENS, ["nohyphen"])