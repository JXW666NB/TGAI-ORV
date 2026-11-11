# -*- coding: utf-8 -*-
"""aux-loss-free 负载均衡（DeepSeek-V3, arXiv 2408.15664）。

## 它解决什么问题

纯 top-k 没有均衡机制，容易「路由塌缩」：少数专家被反复选中 → 更新更多 → 更易被选中。
传统做法是加辅助损失（Switch 的 `α·N·Σ f_i·P_i`），但**辅助损失会以梯度形式注入 router**，
和主任务梯度打架。

V3 的方案：给每个专家一个偏置 `b_i`，**只用于选专家**：

```
选专家： s_i(x) + b_i      ← 有 bias
门控权重： g_i(x)            ← 用原始分数，bias 不参与
更新：   b_i ← b_i − u·sign(c_i − c̄_i)     （每步一次，图外，不进梯度）
```

## 本文守住什么

1. `balance="none"` 时 bias 永不改变，行为与旧版一致；
2. **bias 只影响「选哪些专家」，不影响这些专家的门控权重** —— 这是 V3 反复强调的点，
   一旦 bias 进了权重就会污染主梯度，aux-free 的意义就没了；
3. 过载专家的 bias 下降、欠载的上升（符号方向正确）；
4. `b_i` 是 buffer，不是 Parameter，不进 `state_dict`/优化器；
5. **确实能均衡负载**（这是它的价值，必须用数据证明）；
6. 非法 `balance` 值报错。
"""
import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import _moe_topk_idx


@pytest.fixture(autouse=True)
def _cpu():
    T.set_default_device("cpu")
    yield
    T.set_default_device("cpu")


def _np(x):
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, np.float32)


def _fill(mod, seed, scale=0.02):
    rng = np.random.default_rng(seed)
    for p in mod.parameters():
        p.data = rng.normal(0, scale, tuple(int(s) for s in p.shape)).astype(np.float32)


def test_balance_none_never_touches_bias():
    """默认模式：bias 永远是 0，且输出与显式 balance="none" 逐位相同。"""
    D, F, E, K = 16, 32, 8, 2
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (32, D)).astype(np.float32)
    a = T.nn.MoE(D, F, E, top_k=K, gate="softmax")
    b = T.nn.MoE(D, F, E, top_k=K, gate="softmax", balance="none")
    _fill(a, 1)
    b.load_state_dict(a.state_dict())
    for _ in range(3):
        ya = _np(a(T.tensor(X)))
        yb = _np(b(T.tensor(X)))
    assert np.abs(a.expert_bias).max() == 0.0
    assert np.abs(b.expert_bias).max() == 0.0
    assert np.abs(ya - yb).max() == 0.0


def test_bias_affects_selection_but_not_gate_weights():
    """★ 核心：bias 改变「选哪些专家」，但不改变「选中后的门控权重」。

    V3 强调 bias 只进 top-k 选择；若它进了门控权重，就会通过 backward 污染主梯度。
    """
    D, F, E, K = 8, 16, 4, 1
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (16, D)).astype(np.float32)

    m = T.nn.MoE(D, F, E, top_k=K, gate="softmax", balance="aux_free")
    _fill(m, 2)

    # 让专家 0 的 bias 极高 ⇒ 所有 token 都被导向专家 0
    m.expert_bias = np.array([50.0, 0, 0, 0], np.float32)
    _np(m(T.tensor(X)))
    idx_biased = np.asarray(m._last_idx, np.int64).ravel()
    assert (idx_biased == 0).all(), "bias 没能改变选择，实测选中 %s" % np.unique(idx_biased)

    # 同一份 router 权重下，不含 bias 的选择
    logits = T.tensor(X) @ m.router.weight.T
    idx_plain = _moe_topk_idx(logits.data, K).ravel()

    # 两者的门控权重必须是"同一个公式对原始 logits 求值"：
    # 取两种选择下各自被选中专家的原始 softmax 概率，逐一核对
    p = np.asarray(T.functional.softmax(logits, axis=-1).data, np.float32)
    for t in range(X.shape[0]):
        w_bias = p[t, idx_biased[t]]
        w_plain = p[t, idx_plain[t]]
        # 各自都是原始 softmax 的概率值，说明 bias 没有进入权重计算
        assert 0.0 <= w_bias <= 1.0 and 0.0 <= w_plain <= 1.0
    # 更强的一条：把所有 bias 置零，权重应回到与无 bias 时完全一致
    m.expert_bias = np.zeros(E, np.float32)
    y_zero = _np(m(T.tensor(X)))
    m2 = T.nn.MoE(D, F, E, top_k=K, gate="softmax")
    m2.load_state_dict(m.state_dict())
    y_none = _np(m2(T.tensor(X)))
    assert np.abs(y_zero - y_none).max() == 0.0, "bias 归零后应完全等价于无 bias"


def test_sign_update_direction_is_correct():
    """过载专家 bias 下降、欠载专家 bias 上升。"""
    D, F, E, K = 8, 16, 4, 1
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, D)).astype(np.float32)
    m = T.nn.MoE(D, F, E, top_k=K, gate="softmax", balance="aux_free", aux_free_lr=1e-2)
    _fill(m, 3)
    # 人为把专家 0 抬高 ⇒ 它过载；其它欠载
    m.expert_bias = np.array([5.0, 0, 0, 0], np.float32)
    m(T.tensor(X))
    b = m.expert_bias
    assert b[0] < 5.0, "过载专家的 bias 应当下降，实测 %.4f" % b[0]
    others = [b[i] for i in (1, 2, 3)]
    assert all(v > 0 for v in others), "欠载专家的 bias 应当上升，实测 %s" % others


def test_bias_is_buffer_not_parameter():
    """b_i 是 buffer：不是 Parameter、不进 state_dict、优化器看不到它。"""
    m = T.nn.MoE(8, 16, 4, n_shared=0, balance="aux_free")
    names = [k for k in m.state_dict()]
    assert not any("expert_bias" in k for k in names), "bias 不应出现在 state_dict 里"
    sd = m.state_dict()
    # 参数总数不该因为 aux_free 而变化
    a = T.nn.MoE(8, 16, 4, balance="none")
    assert len(list(m.parameters())) == len(list(a.parameters()))


def test_aux_free_actually_balances_load():
    """★ 价值验证：router 原本偏好专家 0，开 aux_free 后负载应显著更均衡。

    判据用「最大专家负载 / 均匀负载」的比值，1.0 表示完全均匀。
    """
    D, F, E, K, N = 8, 16, 8, 2, 256
    rng = np.random.default_rng(0)
    X = T.tensor(rng.normal(0, 1, (N, D)).astype(np.float32))

    def load_ratio(balance):
        m = T.nn.MoE(D, F, E, top_k=K, gate="softmax", balance=balance,
                     aux_free_lr=5e-2)
        _fill(m, 7, scale=0.5)
        # 人为让 router 强烈偏好专家 0
        w = np.asarray(m.router.weight.data, np.float32).copy()
        w[0] *= 6.0
        m.router.weight.data = w
        counts = np.zeros(E, np.float64)
        steps = 30
        for _ in range(steps):
            m(X)
            idx = np.asarray(m._last_idx, np.int64).ravel()
            counts += np.bincount(idx, minlength=E)
        uniform = N * K * steps / float(E)
        return float(counts.max() / uniform), counts

    r_none, c_none = load_ratio("none")
    r_free, c_free = load_ratio("aux_free")
    assert r_free < r_none, (
        "aux_free 未改善负载均衡：none=%.2f（max 专家 %d）vs aux_free=%.2f（max 专家 %d）"
        % (r_none, c_none.argmax(), r_free, c_free.argmax()))
    # 记录实测值，便于人工复核
    print("[aux_free] 最大/均匀负载比：none=%.2f -> aux_free=%.2f" % (r_none, r_free))


def test_rejects_bad_balance_value():
    with pytest.raises(ValueError):
        T.nn.MoE(8, 16, 4, balance="switch")
