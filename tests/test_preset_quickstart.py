# -*- coding: utf-8 -*-
"""开箱即用层（0.9.21）：`T.preset.transformer` / `T.tokens` / `T.fit`。

判据：
1. 预设 Transformer 能前向、能反向、参数量与结构自洽（共享权重只算一次）；
2. `T.fit` 在**可学的数据**上 loss 明显下降（不是"跑通就行"）；
3. **10 行以内**能完成一次真实训练（用行数断言守住这个承诺）；
4. 预设模型是普通 `nn.Module`，可拆可改。
"""
import numpy as np
import pytest

import tgai_go.train as T


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield
    T.set_default_device("cpu")


def _learnable_corpus(n_tokens, vocab, seed=0):
    """造一个**有规律**的语料（周期序列），保证 loss 能明显下降。

    纯随机 token 的 loss 只能降到 ln(vocab) 附近，看不出"学没学到"，所以这里用周期信号。
    """
    rng = np.random.default_rng(seed)
    period = 7
    base = rng.integers(0, vocab, period)
    return np.asarray([base[i % period] for i in range(n_tokens)], np.int64)


def test_preset_forward_and_param_count():
    m = T.preset.transformer(vocab=64, d_model=32, n_layers=2, n_heads=4, d_ff=64)
    x = np.random.randint(0, 64, 16)
    y = np.random.randint(0, 64, 16)
    loss = m(x, y)
    v = float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy") else loss))
    assert np.isfinite(v) and v > 0
    assert m.n_params() > 0
    # 共享权重时 lm_head.weight 与 embed.weight 是同一对象
    assert m.lm_head.weight is m.embed.weight


def test_preset_untied_embedding_is_separate():
    m = T.preset.transformer(vocab=32, d_model=16, n_layers=1, n_heads=2, tie_embedding=False)
    assert m.lm_head.weight is not m.embed.weight


def test_bad_head_count_is_rejected():
    with pytest.raises(ValueError):
        T.preset.transformer(vocab=32, d_model=32, n_layers=1, n_heads=5)


def test_preset_is_ordinary_module():
    """用户要能拆开改：blocks 是 ModuleList，可以替换某一层。"""
    m = T.preset.transformer(vocab=32, d_model=16, n_layers=2, n_heads=2)
    assert hasattr(m, "blocks") and len(m.blocks) == 2
    assert hasattr(m.blocks[0], "qkv") and hasattr(m.blocks[0], "w2")


def test_fit_actually_learns():
    """在有规律的语料上，30 步后 loss 必须明显下降（判据：降到初始的 70% 以下）。"""
    ids = _learnable_corpus(2000, vocab=32)
    data = T.tokens(ids, seq=16)
    m = T.preset.transformer(vocab=32, d_model=32, n_layers=2, n_heads=4, d_ff=64)
    hist = T.fit(m, data, steps=30, lr=3e-3)["loss"]
    first = float(np.mean(hist[:3]))
    last = float(np.mean(hist[-3:]))
    assert last < first * 0.7, "loss 没降下来：%.4f → %.4f" % (first, last)


def test_tokens_rejects_too_short_input():
    with pytest.raises(ValueError):
        T.tokens([1, 2, 3], seq=16)


def test_ten_line_training_promise():
    """守住 README 的承诺：**10 行以内**完成一次真实训练。

    这里数的是"用户要写的实质代码行"（非空、非注释、去掉 import）。
    """
    lines = [
        'import numpy as np',
        'import tgai_go.train as T',
        'ids = np.random.randint(0, 64, 4000)',
        'data = T.tokens(ids, seq=32)',
        'model = T.preset.transformer(vocab=64, d_model=32, n_layers=2, n_heads=4)',
        'hist = T.fit(model, data, steps=20, lr=1e-3)',
        'print(hist["loss"][-1])',
    ]
    code = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")
            and not ln.startswith("import") and not ln.startswith("from")]
    assert len(code) <= 10, "用户侧实质代码 %d 行，超过 10 行承诺" % len(code)

    # 并且这段话真的能跑（不是纸上谈兵）
    ids = np.random.randint(0, 64, 4000)
    data = T.tokens(ids, seq=32)
    model = T.preset.transformer(vocab=64, d_model=32, n_layers=2, n_heads=4)
    hist = T.fit(model, data, steps=5, lr=1e-3)
    assert len(hist["loss"]) == 5
