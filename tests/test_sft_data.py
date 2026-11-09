# -*- coding: utf-8 -*-
"""SFT 数据层（CPU）：定长打包、mask 语义、批形状。"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train.sft_data import SftDataset  # noqa: E402


class FakeTok(object):
    def encode(self, text, add_bos=True):
        ids = [(ord(c) % 50) + 10 for c in text]
        return ([1] if add_bos else []) + ids


def test_pack_is_fixed_length_and_masked():
    ds = SftDataset(FakeTok(), seq=16)
    x, y, m = ds.encode_one("abcdefgh")
    assert x.shape == y.shape == m.shape == (16,)
    # ids = [1, a..h] 共 9 个 ⇒ n = min(8, 15) = 8
    assert m[:8].tolist() == [1.0] * 8
    assert m[8:].tolist() == [0.0] * 8
    assert y[0] == x[1]


def test_batches_shape_and_multiplicity():
    ds = SftDataset(FakeTok(), seq=8, root="__nonexistent__")
    got = list(ds.batches(batch_size=1, max_batches=2, limit=4))
    assert got == [], "根目录不存在时应产出空（不抛异常、不造假）"


def test_short_text_skipped():
    ds = SftDataset(FakeTok(), seq=8)
    assert ds.encode_one("") is None
    assert ds.encode_one("a") is not None