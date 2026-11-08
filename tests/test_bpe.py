# -*- coding: utf-8 -*-
"""BPE 分词器（纯 Python）—— 按 `.TG` 的 **id 对** merges 验证，不依赖引擎与 GPU。

词表结构（与 V2.7 一致）：
    [0..5] 特殊符号, [6..base-1] 基础符号, [base..] 合并产物（id = base + rank）
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.bpe import BOS_ID, SPACE, UNK_ID, BpeTokenizer  # noqa: E402

# 0=PAD 1=UNK 2=BOS 3=EOS 4=IMG 5=AUD 6=▁ 7=a 8=b 9=c  → base=10
TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<IMG>", "<AUD>", SPACE, "a", "b", "c",
          SPACE + "a", "ab", SPACE + "ab"]
MERGES = [(6, 7), (7, 8), (10, 8)]          # ▁+a→10, a+b→11, (▁a)+b→12


def test_base_is_len_tokens_minus_merges():
    tk = BpeTokenizer(TOKENS, MERGES)
    assert tk.base == 10
    assert tk.id_mode is True


def test_merges_apply_and_produce_expected_ids():
    tk = BpeTokenizer(TOKENS, MERGES)
    ids = tk.encode("ab", add_bos=False)
    assert ids == [12], "应合成单个 token ▁ab(=base+2)"
    assert tk.decode(ids) == "ab"


def test_rank_priority_uses_smallest_rank():
    tk = BpeTokenizer(TOKENS, MERGES)
    # 先 (▁,a) rank0 再 (ab→) 说明按 rank 从小到大，而不是从左到右贪心
    assert tk.encode("a", add_bos=False) == [10]


def test_space_marker_and_roundtrip_stable():
    tk = BpeTokenizer(TOKENS, MERGES)
    ids = tk.encode("abc", add_bos=False)
    assert tk.roundtrip_ok(ids), "编码→解码→再编码应稳定"


def test_unknown_char_maps_to_unk_id():
    tk = BpeTokenizer(TOKENS, MERGES)
    ids = tk.encode("z", add_bos=False)
    assert UNK_ID in ids


def test_bos_and_type_error():
    tk = BpeTokenizer(TOKENS, MERGES)
    assert tk.encode("ab", add_bos=True)[0] == BOS_ID
    with pytest.raises(TypeError):
        tk.encode(b"ab")


def test_bad_merge_reports_clearly():
    with pytest.raises(ValueError):
        BpeTokenizer(TOKENS, ["nohyphen"])