# -*- coding: utf-8 -*-
"""温度参数（temperature）回归测试 —— 缺陷 A。

背景：`_sampling.sample_numpy` 的「③ 温度」分支此前是空的
（只有一行注释），`temperature` 参数对采样分布完全没影响。
这里按 llama.cpp 语义锁住行为：

  * temp == 1（含不传 / None）与旧行为逐位一致；
  * temp <= 0 恒等于 argmax（贪心；penalties/DRY/soft-stop 之后再取）；
  * temp 很小（0.01）时概率几乎全部压在 argmax；
  * temp == 2 的分布比 temp == 1 更平（固定 seed 采样统计熵）。

★ 采样器内部是 `np.random.default_rng(seed)`：传固定 seed 时【每次调用都
  新建 RNG】⇒ 同一个 seed 永远返回同一个 token（这是既有行为，不是本测试
  要测的分布）。所以统计分布必须走 seed=None（全局 np.random），并在测试里
  先 `np.random.seed(...)` 固定全局状态。
"""
import numpy as np
import pytest

from tgai_go._sampling import sample_numpy

VOCAB = 16

# 尖峰分布：temp=1 时 argmax 概率 ≈ 0.9993；temp=0.01 时 ≈ 1 - 1e-434
PEAK = np.zeros(VOCAB, dtype=np.float64)
PEAK[0] = 10.0

# 平缓分布：用于"温度越高分布越平"的熵比较
MILD = np.zeros(VOCAB, dtype=np.float64)
MILD[:4] = (5.0, 4.0, 3.0, 2.0)


class _ProbRecorder:
    """替换 np.random.choice，把 sample_numpy 实际使用的概率分布录下来。

    为什么需要：sample_numpy 不直接暴露概率向量。录下它交给 rng.choice 的
    p，就能对"温度到底改没改 logits"做【确定性】断言，而不是靠抽样碰运气。
    """

    def __init__(self):
        self.p = None

    def choice(self, a, p=None, **kw):
        self.p = np.asarray(p, dtype=np.float64).copy()
        return int(np.argmax(self.p))


def _probs_of(logits, temp, monkeypatch):
    rec = _ProbRecorder()
    monkeypatch.setattr(np.random, 'choice', rec.choice)
    sample_numpy(logits, temp, 0, 1.0, VOCAB)      # p=None ⇒ 走全局 np.random
    assert rec.p is not None, 'sample_numpy 没有调用 rng.choice'
    return rec.p


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def test_temp_none_defaults_to_one_bit_identical(monkeypatch):
    """temp=None 与 temp=1.0 必须逐位一致（旧行为 = 不做任何缩放）。"""
    for logits in (MILD, PEAK):
        p_none = _probs_of(logits, None, monkeypatch)
        p_one = _probs_of(logits, 1.0, monkeypatch)
        assert np.array_equal(p_none, p_one), 'temp=None 与 temp=1.0 的概率不一致'
        # temp 与 1.0 的差 <= 1e-9 也要按"不缩放"处理（避免浮点噪声改变结果）
        p_near = _probs_of(logits, 1.0 + 1e-12, monkeypatch)
        assert np.array_equal(p_one, p_near), 'temp≈1.0 不应改变概率分布'


def test_temp_one_means_no_scaling(monkeypatch):
    """temp=1 的分布必须等于原始 logits 的 softmax（逐位不变的语义）。"""
    p = _probs_of(MILD, 1.0, monkeypatch)
    np.testing.assert_allclose(p, _softmax(MILD), rtol=1e-12, atol=0.0)


def test_temp_two_scales_logits(monkeypatch):
    """temp=2 等价于 logits/2 后采样（llama.cpp 语义）。"""
    p = _probs_of(MILD, 2.0, monkeypatch)
    np.testing.assert_allclose(p, _softmax(MILD / 2.0), rtol=1e-12, atol=0.0)


def test_temp_half_sharpens(monkeypatch):
    """temp<1 必须让分布更尖（logits*2）。"""
    p = _probs_of(MILD, 0.5, monkeypatch)
    np.testing.assert_allclose(p, _softmax(MILD * 2.0), rtol=1e-12, atol=0.0)
    assert p[0] > _softmax(MILD)[0]


def test_temp_zero_is_greedy():
    """temp <= 0 ⇒ 恒等于 argmax（不消费随机数）。"""
    for _ in range(20):
        assert sample_numpy(MILD, 0.0, 0, 1.0, VOCAB) == 0
        assert sample_numpy(PEAK, 0.0, 0, 1.0, VOCAB) == 0
        assert sample_numpy(PEAK, -1.0, 0, 1.0, VOCAB) == 0
    # argmax 不在 0 号位时也要正确
    shifted = np.zeros(VOCAB, dtype=np.float64)
    shifted[7] = 9.0
    shifted[3] = 8.0
    assert sample_numpy(shifted, 0.0, 0, 1.0, VOCAB) == 7


def test_temp_zero_greedy_after_penalties():
    """贪心必须作用在【已施加 penalties】的 arr 上（顺序：penalties → temp）。"""
    logits = np.zeros(VOCAB, dtype=np.float64)
    logits[0] = 10.0                       # 原始 argmax = 0
    # presence_penalty=20 把出现过的 0 号压到 -10 ⇒ argmax 变成 1
    p = {'presence_penalty': 20.0}
    assert sample_numpy(logits, 0.0, 0, 1.0, VOCAB, p=p, recent=[0]) == 1


def test_temp_small_concentrates_on_argmax(monkeypatch):
    """temp=0.01：多采样 200 次必须全部是 argmax。"""
    np.random.seed(20261125)
    got = [sample_numpy(PEAK, 0.01, 0, 1.0, VOCAB) for _ in range(200)]
    assert set(got) == {0}, 'temp=0.01 时 argmax 概率应≈1，实际采到了 %r' % (set(got),)
    # 同时断言概率确实≈1（不是"恰好随机到同一个"）
    p = _probs_of(PEAK, 0.01, monkeypatch)
    assert p[0] > 1.0 - 1e-9


def test_temp_two_flatter_than_one_by_entropy():
    """固定 seed 采样统计熵：temp=2.0 必须比 temp=1.0 明显更平。"""
    n = 20000
    np.random.seed(123456)
    t1 = [sample_numpy(MILD, 1.0, 0, 1.0, VOCAB) for _ in range(n)]
    np.random.seed(123456)          # 同一随机序列，只变温度 ⇒ 排除采样噪声来源
    t2 = [sample_numpy(MILD, 2.0, 0, 1.0, VOCAB) for _ in range(n)]

    def entropy(tids):
        cnt = np.bincount(np.asarray(tids, dtype=np.int64), minlength=VOCAB)
        pr = cnt / cnt.sum()
        pr = pr[pr > 0]
        return float(-(pr * np.log(pr)).sum())

    h1, h2 = entropy(t1), entropy(t2)
    # 理论值：temp=1 ≈ 1.0 nat，temp=2 ≈ 1.4 nat；留足余量防版本差异
    assert h2 > h1 + 0.1, ('温度越高分布应越平：H(temp=1)=%.4f, H(temp=2)=%.4f'
                           % (h1, h2))


def test_temp_one_keeps_legacy_sequence():
    """temp=1 与"不传温度"在同一 seed 下产出的 token 序列一致。"""
    for seed in (0, 1, 7, 12345):
        a = [sample_numpy(MILD, 1.0, 0, 1.0, VOCAB, p={'seed': seed})
             for _ in range(8)]
        b = [sample_numpy(MILD, None, 0, 1.0, VOCAB, p={'seed': seed})
             for _ in range(8)]
        assert a == b


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
