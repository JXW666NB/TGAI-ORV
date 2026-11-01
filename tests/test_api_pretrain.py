# -*- coding: utf-8 -*-
"""`api.train(method="pretrain")` 与 `build_lm_sequences` 的测试（纯 CPU）。

测试里**自建一个迷你 tokenizer**（WordLevel），不依赖任何外部 tokenizer 文件，
这样 CI 没有 HF 缓存也能跑。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

from tgai_go import api
from tgai_go.train import qwen as Q

TINY = dict(V=64, D=32, L=2, NH=4, KV=2, FF=64, THETA=1e6, EPS=1e-6, DH=8,
            QK_NORM=False, QKV_BIAS=True, TIE=True)


@pytest.fixture()
def mini_tok(tmp_path):
    tok_mod = pytest.importorskip("tokenizers")
    vocab = {w: i for i, w in enumerate(
        ["<unk>", "<|endoftext|>"] + ["w%d" % i for i in range(60)])}
    t = tok_mod.Tokenizer(tok_mod.models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = tok_mod.pre_tokenizers.Whitespace()
    p = tmp_path / "tok.json"
    t.save(str(p))
    return str(p)


def test_build_lm_sequences_chunks_text(tmp_path, mini_tok):
    corpus = tmp_path / "c.txt"
    corpus.write_text(" ".join("w%d" % (i % 50) for i in range(200)), encoding="utf-8")
    seqs = Q.build_lm_sequences(Q.load_tokenizer(mini_tok), str(corpus), 16, None)
    assert seqs and all(len(s) == 16 for s in seqs), "每块必须正好 seq_len"
    assert len(seqs) >= 5


def test_build_lm_sequences_accepts_jsonl_text(tmp_path, mini_tok):
    j = tmp_path / "c.jsonl"
    j.write_text('{"text": "w1 w2 w3 w4 w5 w6 w7 w8 w9"}\n', encoding="utf-8")
    seqs = Q.build_lm_sequences(Q.load_tokenizer(mini_tok), str(j), 4, None)
    assert seqs and len(seqs[0]) == 4


def test_build_lm_sequences_rejects_bad_jsonl(tmp_path, mini_tok):
    j = tmp_path / "bad.jsonl"
    j.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        Q.build_lm_sequences(Q.load_tokenizer(mini_tok), str(j), 4, None)
    assert "修法" in str(ei.value)


def test_pretrain_requires_corpus():
    with pytest.raises(ValueError) as ei:
        api.train(method="pretrain", cfg=TINY)
    assert "corpus" in str(ei.value)


def test_pretrain_tiny_end_to_end_runs(tmp_path, mini_tok):
    corpus = tmp_path / "c.txt"
    corpus.write_text(" ".join("w%d" % (i % 50) for i in range(400)), encoding="utf-8")
    r = api.train(method="pretrain", corpus=str(corpus), cfg=TINY, tokenizer=mini_tok,
                  seq_len=16, steps=5, batch=2, lr=1e-3, holdout=2, lr_warmup=1,
                  tiny=True, quiet=True)
    assert r.returncode == 0
    assert r["n_seqs"] >= 5
    assert r.probe_base and r.probe_final
    assert r.verdict in ("PASS", "FAIL")


def test_dpo_and_grpo_still_raise():
    for m in ("dpo", "grpo"):
        with pytest.raises(NotImplementedError) as ei:
            api.method_supported(m)
        assert "修法" in str(ei.value)
