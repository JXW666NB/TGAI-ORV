# -*- coding: utf-8 -*-
"""反向重叠归约（`dp_overlap`）的**逐位**正确性测试（纯 CPU，不需要卡）。

## 为什么这些测试必须存在

重叠归约有三种"看起来正常、其实已经算错"的失败方式：

1. **tied 参数被归约了半成品** —— `tie_embedding` 时同一个张量在反向里被
   `_acc` 两次（embedding 反向 + matmul 反向）。若在第一次累加后就认为"就绪"，
   归约出来的是半成品，而梯度照样是有限值、loss 照样降。
   判据只能是「与不重叠版本**逐位**相同」（`assert_array_equal`，不是 allclose）。
2. **漏归约（pending 没归零）** —— 某个叶子写了 `.grad` 却没触发钩子，
   集合通信里它对应的位置是旧值/0，各 rank 静默分叉。
3. **缺失梯度没按 0 参与** —— 各 rank 的条目数不同 ⇒ 共享内存名字不同 ⇒ 卡死；
   或该参数根本拿不到 `.grad` ⇒ 优化器跳过它，与历史行为不一致。

这些都在 CPU 上用 `NumpyTransport`（直通、无集合通信）验：它跑的是**同一条**
钩子/分桶/写回路径，只是把"集合通信"换成 numpy。逐位比较的是梯度数组本身。
"""
import os

import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import dp_overlap as O


# ══════════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════════
def _tied_model(seed=0, vocab=64, d=16, layers=3, heads=4):
    """一个**权重绑定**的小 Transformer（CPU）。"""
    rng = np.random.default_rng(seed)
    m = T.preset.transformer(vocab=vocab, d_model=d, n_layers=layers,
                             n_heads=heads, tie_embedding=True)
    # 固定权重：两边（重叠 / 不重叠）必须从同一组数出发
    for i, p in enumerate(m.parameters()):
        p.data = rng.normal(0, 0.1, tuple(p.shape)).astype(np.float32)
    assert m.lm_head.weight is m.embed.weight, "这个测试要求 tied 权重是同一个对象"
    return m


def _batch(seed=1, vocab=64, seq=8):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab, (1, seq)).astype(np.int64)
    return ids[:, :-1], ids[:, 1:]


def _params_of(m):
    """与 `v3_train_cli.params_of` 同一口径：按 id() 去重。"""
    out, seen = [], set()
    for p in m.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


def _run_step(m, params, ids, labels, session=None, install=True, clear=True):
    """跑一次前向+反向。`session` 给了就在反向前后走重叠路径。

    `clear=False` 保留已有的 `.grad`（用来测「梯度累积必须退回不重叠」）。
    """
    if clear:
        for p in params:
            p.grad = None
    if session is not None and install:
        T._set_bwd_overlap(session)
    try:
        loss = m(ids, labels)
        loss.backward()
        if session is not None:
            st = session.finish()
            return float(np.asarray(loss.to_numpy()).reshape(-1)[0]), st
        return float(np.asarray(loss.to_numpy()).reshape(-1)[0]), None
    finally:
        T._set_bwd_overlap(None)


def _zero_grad(p):
    """生产路径 dist._zero_grad_of 的等价物：给缺失梯度的参数补全 0（不是 None）。"""
    p.grad = np.zeros(tuple(p.shape), np.float32)


def _grads(params):
    return [None if p.grad is None else np.asarray(p.grad, np.float32).copy()
            for p in params]


def _make_session(params, transport=None, order=None, world=1, scale=None,
                  bucket_bytes=None, zero_grad=None):
    elems = [int(np.prod(tuple(p.shape))) for p in params]
    # ★ 默认用**小桶**（4 KiB）：这个测试模型只有 ~53 KB 梯度，用生产默认的
    #   256 MB 会退化成「一个桶」，跨桶搬运/环形复用的代码路径就测不到了。
    plan = O.build_plan(elems, bucket_bytes=(4096 if bucket_bytes is None
                                             else bucket_bytes), order=order)
    tp = transport if transport is not None else O.NumpyTransport()
    sess = O.OverlapSession(params, plan, tp, world=world, scale=scale,
                            zero_grad=zero_grad)
    return sess, tp


# ══════════════════════════════════════════════════════════════════════
# 1. 静态布局
# ══════════════════════════════════════════════════════════════════════
def test_build_plan_covers_every_param_exactly_once():
    elems = [7, 3, 100, 1, 50, 2, 9]
    plan = O.build_plan(elems, bucket_bytes=64)
    flat = [i for g in plan.groups for i in g]
    assert sorted(flat) == list(range(len(elems)))
    assert len(flat) == len(set(flat))
    for b, g in enumerate(plan.groups):
        assert sum(elems[i] for i in g) == plan.bucket_elems[b]
        for i in g:
            assert plan.param_bucket[i] == b
            assert plan.param_off[i] + elems[i] <= plan.bucket_elems[b]
    assert sum(plan.bucket_elems) == sum(elems)


def test_build_plan_is_deterministic_and_reverse_by_default():
    elems = [4, 4, 4, 4]
    a = O.build_plan(elems, bucket_bytes=8)
    b = O.build_plan(elems, bucket_bytes=8)
    assert a.groups == b.groups
    # 默认倒序：最后一个参数最先就绪
    assert a.groups[0] == [3]


def test_build_plan_rejects_bad_order():
    with pytest.raises(ValueError):
        O.build_plan([1, 2, 3], bucket_bytes=64, order=[0, 1])
    with pytest.raises(ValueError):
        O.build_plan([1, 2, 3], bucket_bytes=64, order=[0, 0, 1])


def test_build_plan_oversized_param_gets_own_bucket():
    plan = O.build_plan([1, 1000, 1], bucket_bytes=16)     # 1000 元素 >> 4 元素上限
    assert sum(1 for e in plan.bucket_elems if e == 1000) == 1
    assert len(plan.groups) == 3
    big = [b for b, g in enumerate(plan.groups) if g == [1]]
    assert big, "超限参数必须自成一个桶：%s" % (plan.groups,)


def test_plan_order_adapter_puts_head_first_emb_last():
    class _P(object):
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return self.name

    emb, ln_f, head = _P("emb"), _P("ln_f"), _P("head")
    layers = [_P("l0"), _P("l1"), _P("l2")]
    params = [emb, ln_f, head] + layers
    order = O.plan_order_adapter(params, first=[head], last=[emb])
    assert order[0] == 2          # head 最前
    assert order[-1] == 0         # emb 最后
    assert sorted(order) == list(range(len(params)))
    # tied：head is emb ⇒ first/last 指向同一个下标，只出现一次
    order2 = O.plan_order_adapter([emb, ln_f] + layers, first=[emb], last=[emb])
    assert sorted(order2) == list(range(len(params) - 1))


# ══════════════════════════════════════════════════════════════════════
# 2. ★ tied 参数：与不重叠版本**逐位**相同
# ══════════════════════════════════════════════════════════════════════
def test_tied_embedding_bitwise_identical_to_no_overlap():
    m = _tied_model()
    params = _params_of(m)
    ids, labels = _batch()

    _, _ = _run_step(m, params, ids, labels)               # 基线（不重叠）
    base = _grads(params)

    sess, tp = _make_session(params, zero_grad=_zero_grad)
    _, st = _run_step(m, params, ids, labels, session=sess)
    assert st["active"] is True, "会话没生效：%s" % (st.get("reason"),)
    got = _grads(params)

    assert st["zero_filled"] == 0
    assert st["buckets"] >= 2, "桶太小的话测不到跨桶搬运"
    for i, (a, b) in enumerate(zip(base, got)):
        assert b is not None, "参数 #%d 的梯度丢了" % (i,)
        # ★ 逐位，不是 allclose
        np.testing.assert_array_equal(
            a, b, err_msg="tied 模型下参数 #%d 与不重叠版本不逐位相同" % (i,))
    # 绑定的那个张量必须**恰好**出现在参数表里一次
    tied = m.embed.weight
    assert sum(1 for p in params if p is tied) == 1


def test_tied_leaf_fires_exactly_once():
    """tied 张量在反向里被 `_acc` 两次，但"写 .grad"只能有一次。"""
    m = _tied_model()
    params = _params_of(m)
    ids, labels = _batch()
    sess, tp = _make_session(params, zero_grad=_zero_grad)
    fired = []
    orig = sess.leaf_ready
    sess.leaf_ready = lambda node: (fired.append(id(node)), orig(node))[1]
    _run_step(m, params, ids, labels, session=sess)
    tied_id = id(m.embed.weight)
    assert fired.count(tied_id) == 1, (
        "tied 权重的梯度就绪钩子触发了 %d 次（应为 1）："
        "按反向边计数会让桶永远等不到齐" % (fired.count(tied_id),))


def test_tied_grad_equals_both_paths_sum():
    """独立验证：tied 参数的梯度 = embedding 路径贡献 + lm_head 路径贡献。

    做法：同一模型跑两次 —— 一次正常，一次把 `lm_head.weight` 换成**不共享**的
    另一个张量（各自只算一条路径），两次梯度相加应当等于共享时的梯度。
    """
    m = _tied_model()
    params = _params_of(m)
    ids, labels = _batch()
    _run_step(m, params, ids, labels)
    shared = np.asarray(m.embed.weight.grad, np.float32).copy()

    # ① 只留 embedding 路径：把 head 换成**数值相同但对象不同**的新张量
    #    （必须是同一组数，否则 lm_head 那条路径的贡献与共享时不可比）
    m.lm_head.weight = T.Tensor(
        np.asarray(m.embed.weight.data, np.float32).copy(), requires_grad=True)
    params2 = _params_of(m)
    _run_step(m, params2, ids, labels)
    only_emb = np.asarray(m.embed.weight.grad, np.float32).copy()
    head_alone = np.asarray(m.lm_head.weight.grad, np.float32).copy()

    # 共享时的梯度 = 两条路径之和（fp32 加法逐位可复现：a+b 与 b+a 相同）
    assert np.allclose(shared, only_emb + head_alone, rtol=0, atol=1e-7), (
        "tied 梯度不等于两条路径贡献之和：max|d|=%g"
        % float(np.abs(shared - (only_emb + head_alone)).max()))


# ══════════════════════════════════════════════════════════════════════
# 3. 缺失梯度 / 不参与参数 / 冻结参数
# ══════════════════════════════════════════════════════════════════════
def test_param_not_used_in_graph_is_zero_filled():
    """`params` 里有一个图里根本不用的张量 ⇒ 它按 0 参与，最后 .grad 是全 0。"""
    m = _tied_model()
    params = _params_of(m)
    extra = T.Tensor(np.ones((5,), np.float32), requires_grad=True)
    params = params + [extra]
    ids, labels = _batch()

    base_sess, _ = _make_session(params, zero_grad=None)
    st = base_sess.finish()                                # 没 attach：应返回 inactive
    assert st["active"] is False

    sess, tp = _make_session(params, zero_grad=lambda p: setattr(
        p, "grad", np.zeros(tuple(p.shape), np.float32)))
    _, st = _run_step(m, params, ids, labels, session=sess)
    assert st["active"] is True
    assert st["zero_filled"] == 1, "应恰好有 1 个参数没梯度：%s" % (st,)
    assert st["pending_zeroed"] is True
    assert extra.grad is not None
    np.testing.assert_array_equal(np.asarray(extra.grad, np.float32),
                                  np.zeros((5,), np.float32))


def test_frozen_param_is_not_counted_as_pending():
    """`requires_grad=False` 的参数不会拿到 .grad ⇒ 不能算进 pending（否则永远等不到）。"""
    m = _tied_model()
    frozen = T.Tensor(np.ones((3,), np.float32), requires_grad=False)
    params = _params_of(m) + [frozen]
    ids, labels = _batch()
    sess, _ = _make_session(params, zero_grad=_zero_grad)
    _, st = _run_step(m, params, ids, labels, session=sess)
    assert st["active"] is True
    assert st["zero_filled"] == 1
    assert frozen.grad is None or float(np.abs(np.asarray(frozen.grad, np.float32)).max()) == 0.0


# ══════════════════════════════════════════════════════════════════════
# 4. 安全阀
# ══════════════════════════════════════════════════════════════════════
def test_grad_accumulation_falls_back():
    """反向前已有 .grad（梯度累积）⇒ 必须整步退回不重叠，不许提前归约半成品。"""
    m = _tied_model()
    params = _params_of(m)
    ids, labels = _batch()
    for p in params:
        p.grad = np.zeros(tuple(p.shape), np.float32)      # 假装累积了上一步
    sess, _ = _make_session(params, zero_grad=_zero_grad)
    # ★ clear=False：必须保留上面那批「累积中的」.grad，否则测的就不是累积场景
    _, st = _run_step(m, params, ids, labels, session=sess, clear=False)
    assert st["active"] is False
    assert st["reason"].startswith("grad_accumulation")


def test_checkpointed_graph_falls_back():
    """图里有 `T.checkpoint` ⇒ 嵌套反传会各自写 .grad，必须退回不重叠。"""
    def seg(x, w):
        return T.checkpoint(lambda z: T.exp(z) @ w, x)

    rng = np.random.default_rng(0)
    w = T.Tensor(rng.normal(0, 0.1, (8, 8)).astype(np.float32), requires_grad=True)
    x0 = T.Tensor(rng.normal(0, 0.1, (4, 8)).astype(np.float32), requires_grad=True)
    params = [w, x0]
    sess, _ = _make_session(params, zero_grad=_zero_grad)
    T._set_bwd_overlap(sess)
    try:
        loss = seg(x0, w).sum()
        loss.backward()
        st = sess.finish()
    finally:
        T._set_bwd_overlap(None)
    assert st["active"] is False
    assert st["reason"] == "grad_checkpoint"
    assert w.grad is not None and x0.grad is not None, "退回路径不该影响梯度"


def test_double_backward_raises():
    """同一次会话里第二次 backward 必须炸（否则第二次的反向会写到已发出的桶里）。"""
    m = _tied_model()
    params = _params_of(m)
    ids, labels = _batch()
    sess, _ = _make_session(params, zero_grad=_zero_grad)
    T._set_bwd_overlap(sess)
    try:
        m(ids, labels).backward()
        with pytest.raises(RuntimeError, match="backward"):
            m(ids, labels).backward()
    finally:
        T._set_bwd_overlap(None)
        sess.abort()


def test_missing_hook_is_loud():
    """该到而没到、但 .grad 已经被写了 ⇒ 必须抛（这就是"静默漏归约"）。"""
    m = _tied_model()
    params = _params_of(m)
    ids, labels = _batch()
    sess, _ = _make_session(params, zero_grad=_zero_grad)
    # 把最后一个参数的就绪钩子吞掉：模拟"钩子漏了"
    victim = params[-1]
    orig = sess.leaf_ready

    def patched(node):
        if node is victim:
            return                     # 吞掉：.grad 写了，但会话不知道
        return orig(node)

    sess.leaf_ready = patched
    with pytest.raises(RuntimeError, match="漏掉|漏了"):
        _run_step(m, params, ids, labels, session=sess)


# ══════════════════════════════════════════════════════════════════════
# 5. 语义：平均 / 缺 0 参与 / 顺序
# ══════════════════════════════════════════════════════════════════════
def test_two_rank_average_matches_numpy():
    """模拟 2 rank：peer 的桶 = own * 2 ⇒ 结果应为 (own + 2*own)/2，逐位。"""
    m = _tied_model()
    params = _params_of(m)
    ids, labels = _batch()
    _run_step(m, params, ids, labels)
    own = _grads(params)

    seen = {"n": 0}

    def reduce_fn(buf, info, sess):
        # 对端数据 = 本端 × 2（确定性，便于独立算期望值）
        buf += buf * np.float32(2.0)
        seen["n"] += 1

    tp = O.NumpyTransport(reduce_fn=reduce_fn)
    sess, _ = _make_session(params, transport=tp, world=2,
                            zero_grad=_zero_grad)
    _, st = _run_step(m, params, ids, labels, session=sess)
    assert st["active"] is True
    assert seen["n"] == st["buckets"]
    for i, p in enumerate(params):
        want = (own[i] + own[i] * np.float32(2.0)) * np.float32(0.5)
        np.testing.assert_array_equal(
            np.asarray(p.grad, np.float32), want,
            err_msg="参数 #%d 的 2-rank 平均结果不逐位相同" % (i,))


def test_missing_region_is_zero_in_collective():
    """缺失梯度参与集合通信时必须是 0（不是跳过）—— 各 rank 的桶大小才会一致。"""
    m = _tied_model()
    params = _params_of(m)
    extra = T.Tensor(np.ones((6,), np.float32), requires_grad=True)
    params = params[:2] + [extra] + params[2:]      # 塞进中间，确保它和别的参数同桶
    ids, labels = _batch()
    _run_step(m, params, ids, labels)
    own = {i: _grads([p])[0] for i, p in enumerate(params)}

    def reduce_fn(buf, info, sess):
        buf += buf * np.float32(2.0)                # 对端 = 本端×2

    tp = O.NumpyTransport(reduce_fn=reduce_fn)
    sess, _ = _make_session(params, transport=tp, world=2, bucket_bytes=1 << 20,
                            zero_grad=_zero_grad)
    _, st = _run_step(m, params, ids, labels, session=sess)
    assert st["zero_filled"] == 1
    np.testing.assert_array_equal(np.asarray(extra.grad, np.float32),
                                  np.zeros((6,), np.float32))
    for i in (0, 1, 3):
        p = params[i]
        want = (own[i] + own[i] * np.float32(2.0)) * np.float32(0.5)
        np.testing.assert_array_equal(np.asarray(p.grad, np.float32), want)


def test_late_buckets_are_counted():
    """被「永远不到的成员」卡在 FIFO 里的桶必须在 finish 里补发并计数。

    「永远不到」= 建图时是可达叶子（pending=1）但反向里没有梯度路径到达（soft）。
    ORV 现有算子都会给每个 requires_grad 的父节点累加，构造不出这种图，
    所以这里**直接驱动会话状态**（open + leaf_ready），只验证记账/补发路径本身。
    """
    params = [T.Tensor(np.ones((4,), np.float32), requires_grad=True) for _ in range(4)]
    tp = O.NumpyTransport()
    plan = O.build_plan([4, 4, 4, 4], bucket_bytes=16)     # 一参数一桶
    assert plan.nbuckets == 4
    sess = O.OverlapSession(params, plan, tp, world=1, scale=1.0,
                            zero_grad=_zero_grad)
    # 模拟 attach_graph 已完成、反向正在跑
    sess._attached = True
    sess._active = True
    sess._t_begin = 0.0
    sess._pending = [1, 1, 1, 1]
    tp.open(sess)
    # 桶顺序是 [3]、[2]、[1]、[0]；让 #3/#2/#1 就绪，#0 永远不到
    for i in (3, 2, 1):
        params[i].grad = np.ones((4,), np.float32) * np.float32(i)
        sess.leaf_ready(params[i])
    assert sess._submitted == 3, "前 3 个桶应在反向中发出"
    st = sess.finish()
    assert st["active"] is True
    assert st["late_buckets"] == 1, "被缺失成员卡住的桶必须被统计到：%s" % (st,)
    assert st["pending_zeroed"] is True
    assert st["zero_filled"] == 1
    np.testing.assert_array_equal(np.asarray(params[0].grad, np.float32),
                                  np.zeros((4,), np.float32))


# ══════════════════════════════════════════════════════════════════════
# 6. world=1 / 开关 / 环境变量
# ══════════════════════════════════════════════════════════════════════
def test_world_one_is_noop():
    """`world=1` 必须零行为变化：`begin_overlap` 直接返回 None，钩子不安装。"""
    from tgai_go.train import dist as D

    dp = D.DataParallel(world=1, rank=0, job="pytest_ov_noop")
    x = T.Tensor(np.ones((2, 2), np.float32), requires_grad=True)
    assert dp.begin_overlap([x]) is None
    assert T._bwd_overlap_session() is None
    assert dp.finish_overlap() is None
    dp.close()


def test_overlap_env_switch(monkeypatch):
    monkeypatch.delenv("TG_DP_OVERLAP", raising=False)
    assert O.overlap_enabled() is True
    for off in ("0", "false", "OFF", "no"):
        monkeypatch.setenv("TG_DP_OVERLAP", off)
        assert O.overlap_enabled() is False
    monkeypatch.setenv("TG_DP_OVERLAP", "nope")
    with pytest.raises(ValueError, match="TG_DP_OVERLAP"):
        O.overlap_enabled()


def test_bucket_env(monkeypatch):
    monkeypatch.delenv("TG_DP_BUCKET_MB", raising=False)
    assert O.resolve_bucket_bytes() == 256 << 20
    monkeypatch.setenv("TG_DP_BUCKET_MB", "64")
    assert O.resolve_bucket_bytes() == 64 << 20
    monkeypatch.setenv("TG_DP_BUCKET_MB", "0")
    assert O.resolve_bucket_bytes() == 0
    monkeypatch.setenv("TG_DP_BUCKET_MB", "-1")
    with pytest.raises(ValueError):
        O.resolve_bucket_bytes()
    monkeypatch.setenv("TG_DP_OVERLAP_SLOTS", "0")
    with pytest.raises(ValueError):
        O.resolve_slots()


def test_duplicate_params_rejected():
    x = T.Tensor(np.ones((2,), np.float32), requires_grad=True)
    plan = O.build_plan([2, 2], bucket_bytes=1 << 20)
    with pytest.raises(ValueError, match="重复"):
        O.OverlapSession([x, x], plan, O.NumpyTransport(), zero_grad=None)


def test_session_stats_have_required_fields(monkeypatch):
    m = _tied_model()
    params = _params_of(m)
    ids, labels = _batch()
    sess, _ = _make_session(params, zero_grad=_zero_grad)
    _, st = _run_step(m, params, ids, labels, session=sess)
    for key in ("active", "mode", "buckets", "slots", "params", "zero_filled",
                "late_buckets", "exposed_ms", "comm_ms", "max_bucket_mb"):
        assert key in st, "stats() 缺字段 %s：%s" % (key, sorted(st))
    assert st["params"] == len(params)
    # 已经 finish 过的会话再取 stats 必须**返回缓存**（事件/缓冲可能已经销毁）
    assert sess.stats() is st
