# -*- coding: utf-8 -*-
"""新手四步的入口测试（0.9.27）：load_corpus → train_tokenizer → tokens → fit。

这些是 README 里对新手承诺的 API，必须真的存在、真的能用，且**不依赖 GPU**。
"""
import io
import os

import numpy as np
import pytest

import tgai_go.train as T


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield
    T.set_default_device("cpu")


def _corpus(n=120, seed=0):
    rng = np.random.default_rng(seed)
    a = ["模型", "数据", "梯度", "注意力"]
    b = ["优化", "训练", "收敛", "更新"]
    c = ["损失", "权重", "分布", "特征"]
    return "".join("%s%s了%s。" % (rng.choice(a), rng.choice(b), rng.choice(c))
                   for _ in range(n))


def test_exports_exist():
    """README 里点名的东西必须都能点出来。"""
    for name in ("load_corpus", "train_tokenizer", "Tokenizer", "tokens", "fit",
                 "preset", "SftDataset", "iter_jsonl_texts", "ppo",
                 "DPOTrainer", "GRPOTrainer", "PPOTrainer",
                 "from_torch", "to_torch"):
        assert hasattr(T, name), "缺少 %s（README 里写了）" % name
    assert hasattr(T.ppo, "gae") and hasattr(T.ppo, "ppo_loss")
    assert hasattr(T.optim, "Muon") and hasattr(T.sched, "CosineLR")


def test_train_tokenizer_roundtrip():
    text = _corpus()
    tok = T.train_tokenizer(text, vocab_size=200)
    assert isinstance(tok, T.Tokenizer)
    assert 50 <= tok.vocab_size <= 200, tok
    ids = tok.encode(text)
    assert len(ids) > 0
    # 注意：`train_bpe` 在"没有出现 ≥2 次的相邻对"时会提前停止，
    # 所以 voc​ab_size 只保证是**上限**，实际可能更小（小语料上很常见）。
    assert tok.vocab_size >= 50, tok
    assert tok.vocab_size <= 200, tok
    # id 必须落在词表内，否则模型 embedding 会越界
    assert min(ids) >= 0 and max(ids) < tok.vocab_size


def test_train_tokenizer_rejects_empty():
    with pytest.raises(ValueError):
        T.train_tokenizer("", vocab_size=100)


def test_load_corpus_text_and_jsonl(tmp_path):
    p = tmp_path / "c.txt"
    io.open(str(p), "w", encoding="utf-8").write("你好世界")
    assert T.load_corpus(str(p)) == "你好世界"

    q = tmp_path / "c.jsonl"
    io.open(str(q), "w", encoding="utf-8").write(
        '{"text": "第一句"}\n{"text": "第二句"}\n')
    got = T.load_corpus(str(q))
    assert "第一句" in got and "第二句" in got


def test_four_step_pipeline_learns():
    """端到端：四步串起来，loss 必须明显下降（否则说明链路有问题）。"""
    text = _corpus(200)
    tok = T.train_tokenizer(text, vocab_size=200)
    ids = tok.encode(text)
    data = T.tokens(ids, seq=32)
    model = T.preset.transformer(vocab=tok.vocab_size, d_model=32, n_layers=1, n_heads=2)
    hist = T.fit(model, data, steps=40, lr=5e-3, log_every=0)
    first, last = hist["loss"][0], hist["loss"][-1]
    assert last < first * 0.9, "loss 没降：%.4f -> %.4f" % (first, last)
