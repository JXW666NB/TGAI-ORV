# -*- coding: utf-8 -*-
"""ZeRO-1/2 的**无卡**正确性测试（分片规划 + 打包/写回 + 分布式范数的结合顺序）。

分三类，越往后越接近真实：

1. `plan_ranges` 的纯函数性质 —— 全覆盖、不重叠、确定性、均衡；
2. `pack_shard` / `unpack_shard` 的偏移数学 —— 每个元素恰好被搬一次；
3. **分布式 gnorm 的结合顺序** —— ZeRO-2 把梯度切开以后，gnorm 还要和
   `clip_grad_norm_` **逐位相同**。这一条靠"每 rank 只填自己的槽位、其余清零，
   再按原张量顺序做同一次树归约"实现（`x + 0.0 == x` 在 fp32 下精确）。
   这里就用内核语义的 numpy 复刻把它钉住。

真跨进程的那部分（共享内存 all_gather / reduce_scatter）在
`tests/test_zero_transport.py`（两个进程、真 `dist.DataParallel`）。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tgai_go.train import zero as Z            # noqa: E402
from zero_fake import FakeRT, NParam, NBuf, reference_ssq_total   # noqa: E402


# ═════════════════════════════════════════════════════════════════════════
# 1) 分片规划
# ═════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("sizes,world", [
    ([10, 20, 30], 2),
    ([10, 20, 30], 3),
    ([1, 1, 1, 1, 1, 1, 1], 3),
    ([0, 5, 0, 7, 0], 2),
    ([1000], 4),
    ([3, 1, 4, 1, 5, 9, 2, 6], 4),
])
def test_plan_ranges_partitions_every_index_once(sizes, world):
    """所有下标必须被**恰好一次**覆盖，且每段是连续区间。"""
    rs = Z.plan_ranges(sizes, world)
    assert len(rs) == world
    assert rs[0][0] == 0
    assert rs[-1][1] == len(sizes)
    for k in range(world):
        s, e = rs[k]
        assert 0 <= s <= e <= len(sizes), "第 %d 段非法：%r" % (k, (s, e))
        if k + 1 < world:
            assert e == rs[k + 1][0], "第 %d 段与下一段不连续：%d != %d" % (
                k, e, rs[k + 1][0])
    total = 0
    for s, e in rs:
        total += sum(sizes[s:e])
    assert total == sum(sizes), "分片没有覆盖全部元素"


@pytest.mark.parametrize("world", [1, 2, 3, 5, 8])
def test_plan_ranges_is_deterministic(world):
    """各 rank 必须**独立算出完全相同**的分配（否则会把 A 的切片写到 B 上）。"""
    sizes = [98_304_000, 1024] + [2_883_584] * 8
    a = Z.plan_ranges(sizes, world)
    b = Z.plan_ranges(list(sizes), world)
    assert a == b


@pytest.mark.parametrize("world", [2, 4])
def test_plan_ranges_balances_element_counts(world):
    """均衡度：最大段不超过理想值的 1.6 倍（张量粒度切分，允许一个张量的偏差）。"""
    sizes = [98_304_000] + [2_883_584] * 64
    rs = Z.plan_ranges(sizes, world)
    loads = [sum(sizes[s:e]) for s, e in rs]
    ideal = sum(sizes) / float(world)
    assert sum(loads) == sum(sizes)
    assert max(loads) <= ideal * 1.6, "分配太不均衡：%r（ideal=%.0f）" % (loads, ideal)


def test_plan_ranges_more_ranks_than_params():
    """卡比参数多时允许空段，但不能丢下标。"""
    rs = Z.plan_ranges([5], 4)
    assert rs[0][0] == 0 and rs[-1][1] == 1
    assert [x for x in rs] == sorted(rs)


def test_plan_ranges_empty_model_is_all_empty():
    assert Z.plan_ranges([], 2) == [(0, 0), (0, 0)]


def test_plan_ranges_rejects_bad_world():
    with pytest.raises(ValueError) as ei:
        Z.plan_ranges([1, 2], 0)
    assert "world" in str(ei.value) and "修法" in str(ei.value)


# ═════════════════════════════════════════════════════════════════════════
# 2) ShardPlan / segments / 打包写回
# ═════════════════════════════════════════════════════════════════════════
def _params(sizes, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i, n in enumerate(sizes):
        out.append(NParam(rng.normal(0, 1, (max(1, n),)).astype(np.float32)))
    return out


@pytest.mark.parametrize("chunk", [1, 3, 7, 16, 10_000])
def test_segments_cover_shard_exactly_once(chunk):
    """把 shard 的 [lo,hi) 拆成段时，**每个元素恰好被覆盖一次**。

    漏一个 ⇒ 参数永远收不回来（静默算错）；重一个 ⇒ 被写两次（也是静默算错）。
    两种都不会报错，只能靠这条测出来。
    """
    sizes = [5, 0, 9, 2, 13]
    params = _params(sizes)
    for world in (2, 3):
        for rank in range(world):
            plan = Z.ShardPlan(params, world, rank)
            have = plan.shard_elems[rank]
            got = np.zeros(max(1, have), np.int32)
            lo = 0
            while lo < max(1, plan.cap):
                hi = min(plan.cap, lo + chunk)
                for _i, _so, do, n in plan.segments(rank, lo, hi):
                    got[lo + do:lo + do + n] += 1
                lo = hi
            if have == 0:
                continue
            assert np.all(got[:have] == 1), (
                "world=%d rank=%d 覆盖次数不是 1：%r" % (world, rank, got[:have].tolist()))
            assert not np.any(got[have:have + 1] if have < got.size else [])


@pytest.mark.parametrize("chunk", [4, 1024])
def test_pack_unpack_roundtrip(chunk):
    """打包再写回，参数必须**逐位还原**（含跨块的张量边界）。"""
    sizes = [7, 3, 5, 11]
    rt = FakeRT()
    orig = _params(sizes, seed=1)
    for world in (2, 3):
        # ① 每个 rank 把自己的切片打包出来（模拟 all_gather 的 send 侧）
        sends = []
        for rank in range(world):
            plan = Z.ShardPlan(orig, world, rank)
            buf = NBuf(np.zeros(max(1, plan.cap), np.float32))
            lo = 0
            while lo < plan.cap:
                hi = min(plan.cap, lo + chunk)
                # ★ 整段缓冲按 [lo,hi) 偏移切片：这样打包结果落在 buf[lo:hi]，
                #   与 `gather_params` 里"每块复用同一小块缓冲"的两种用法都对齐。
                Z.pack_shard(rt, plan, rank, lo, hi, rt.sub_view(buf, (hi - lo,), lo))
                lo = hi
            sends.append((plan, buf))
        # ② 每个 rank 收齐所有人的切片并写回自己那份（先填 -999 以便抓错写）
        for rank in range(world):
            dst = [NParam(np.full(max(1, n), -999.0, np.float32)) for n in sizes]
            plan = Z.ShardPlan(dst, world, rank)
            for r in range(world):
                p_r, buf_r = sends[r]
                lo = 0
                while lo < p_r.cap:
                    hi = min(p_r.cap, lo + chunk)
                    Z.unpack_shard(rt, plan, r, lo, hi,
                                   rt.sub_view(buf_r, (hi - lo,), lo))
                    lo = hi
            for i in range(len(sizes)):
                assert np.array_equal(dst[i].numpy(), orig[i].numpy()), (
                    "world=%d rank=%d 第 %d 个张量没被正确写回：%r vs %r"
                    % (world, rank, i, dst[i].numpy()[:6], orig[i].numpy()[:6]))


def test_pack_shard_zeroes_the_padding_tail():
    """``cap`` 对齐的 padding 必须清零（收端不读它，但线上不该出现垃圾）。"""
    rt = FakeRT()
    # 4+4+4+3=15 个元素、切 2 段 ⇒ 8 / 7，cap=8 ⇒ 第 1 个 rank 有 1 个 padding
    params = _params([4, 4, 4, 3], seed=2)
    plan = Z.ShardPlan(params, 2, 1)
    assert plan.shard_elems == [8, 7], plan.shard_elems
    assert plan.cap == 8
    buf = NBuf(np.full(plan.cap, 12345.0, np.float32))
    Z.pack_shard(rt, plan, 1, 0, plan.cap, buf)
    want = np.concatenate([params[2].numpy(), params[3].numpy()])
    assert np.array_equal(buf.np[:7], want)
    assert np.all(buf.np[7:8] == 0.0), "padding 尾巴必须清零：%r" % buf.np[7:8]


def test_pack_grads_treats_missing_grad_as_zero():
    """缺梯度按 0 参与（与数据并行一致），**不能**被跳过（否则字节数对不上）。"""
    rt = FakeRT()
    params = _params([4, 4], seed=3)
    params[0].set_grad(np.full(4, 2.0, np.float32))
    # params[1] 故意不给梯度
    plan = Z.ShardPlan(params, 1, 0)
    buf = NBuf(np.full(8, 7.0, np.float32))
    Z.pack_grads(rt, plan, 0, 0, 8, buf)
    assert np.array_equal(buf.np, np.array([2, 2, 2, 2, 0, 0, 0, 0], np.float32))


# ═════════════════════════════════════════════════════════════════════════
# 3) 分布式 gnorm 的结合顺序（ZeRO-2 逐位一致的核心）
# ═════════════════════════════════════════════════════════════════════════
class _DP1(object):
    """单 rank 的 DataParallel 替身：只提供 ZeRO 用到的两个入口。"""

    world = 1

    def __init__(self, rt):
        self._rt = rt

    def _nccl_rt(self, rt=None):
        return self._rt if rt is None else rt

    def all_reduce_array(self, a, rt=None):
        return np.asarray(a, np.float32)


def _five_grads(seed=7):
    rng = np.random.default_rng(seed)
    bufs = []
    for n in (3, 501, 1027, 64, 9):
        bufs.append(NParam(rng.normal(0, 1, (n,)).astype(np.float32)))
    return bufs


def test_ssq_part_slots_are_disjoint_and_zero_elsewhere():
    """每个 rank 只写自己的槽位；其余槽位必须**保持 0**（这是逐位一致的前提）。"""
    rt = FakeRT()
    params = _five_grads()
    for p in params:
        p.set_grad(p.numpy())
    plan = Z.ShardPlan(params, 3, 1)               # 中间的 rank：i0 > 0 且有右邻居
    bufs = [p.grad._gpu for p in params]
    _nb, off = rt._shard_offsets(bufs)
    part = NBuf(np.zeros(int(off[-1]), np.float32))
    owned = bufs[plan.i0:plan.i1]
    rt.sumsq_multi_shard_into(owned, part, int(off[plan.i0]))
    lo, hi = int(off[plan.i0]), int(off[plan.i1])
    assert np.all(part.np[:lo] == 0.0), "左边界之前的槽位被写了"
    assert np.all(part.np[hi:] == 0.0), "右边界之后的槽位被写了"
    assert np.any(part.np[lo:hi] != 0.0), "自己的槽位没写进去"


def test_zero_slot_tensors_do_not_change_the_total_at_all():
    """**关键性质**：槽位清零的张量贡献精确的 0.0 ⇒ 分布式与不分片逐位相同。

    做法：把 5 个张量按 2 段切开，两段分别写进各自的 part（互不重叠），
    再逐元素相加（等价于跨 rank 求和：另一边是 0），跑同一个 final ——
    结果必须与"一次写全部"**逐位相等**。
    """
    rt = FakeRT()
    params = _five_grads(seed=11)
    for p in params:
        p.set_grad(p.numpy())
    bufs = [p.grad._gpu for p in params]
    ref = reference_ssq_total(bufs)

    _nb, off = rt._shard_offsets(bufs)
    total_blocks = int(off[-1])
    pa = NBuf(np.zeros(total_blocks, np.float32))
    pb = NBuf(np.zeros(total_blocks, np.float32))
    cut = 2
    rt.sumsq_multi_shard_into(bufs[:cut], pa, int(off[0]))
    rt.sumsq_multi_shard_into(bufs[cut:], pb, int(off[cut]))
    merged = NBuf(pa.np + pb.np)                    # 跨 rank 求和的等价物
    got = rt.sumsq_blocks_total(merged, off, len(bufs))
    assert got == ref, "分布式平方和与参考路径不逐位相等：%.17g vs %.17g" % (got, ref)


def test_grad_ssq_total_owned_only_matches_reference_for_owned_prefix():
    """单 rank（world=1）时 `grad_ssq_total` 必须等于参考路径（覆盖全部张量）。"""
    rt = FakeRT()
    params = _five_grads(seed=13)
    for p in params:
        p.set_grad(p.numpy())
    plan = Z.ShardPlan(params, 1, 0)
    dp = _DP1(rt)
    got = Z.grad_ssq_total(plan, dp, rt=rt)
    ref = reference_ssq_total([p.grad._gpu for p in params])
    assert got == ref, "%.17g vs %.17g" % (got, ref)


def test_grad_ssq_total_single_tensor_matches():
    """只有一个张量时走 `sumsq_parallel` 特例（与 `sumsq_multi_total_sharded` 一致）。"""
    rt = FakeRT()
    p = NParam(np.arange(1, 101, dtype=np.float32))
    p.set_grad(p.numpy())
    plan = Z.ShardPlan([p], 1, 0)
    got = Z.grad_ssq_total(plan, _DP1(rt), rt=rt)
    want = float(np.float32(np.sum(np.float32(p.numpy() ** 2))))
    # 树归约与 numpy 的成对求和同量级：只要求 1e-5 相对误差
    assert abs(got - want) <= 1e-5 * max(1.0, want)


def test_grad_ssq_total_skips_rank_local_missing_grads_consistently():
    """某 rank 缺梯度时，参考列表里也不该有它（与 `clip_grad_norm_` 的过滤一致）。"""
    rt = FakeRT()
    params = _five_grads(seed=17)
    for i, p in enumerate(params):
        if i != 2:
            p.set_grad(p.numpy())
    plan = Z.ShardPlan(params, 1, 0)
    got = Z.grad_ssq_total(plan, _DP1(rt), rt=rt)
    ref = reference_ssq_total([p.grad._gpu for p in params if p.grad is not None])
    assert got == ref


# ═════════════════════════════════════════════════════════════════════════
# 4) 优化器外壳 / 通信量模型
# ═════════════════════════════════════════════════════════════════════════
class _FakeOpt(object):
    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr = float(lr)
        self.t = 0
        self.stepped = 0

    def step(self, **kw):
        self.stepped += 1
        self.t += 1
        return kw

    def bytes_per_param(self):
        return 3.0625


def test_sharded_optimizer_zero_grad_clears_all_params():
    """★ 必须清**全部**参数：只清自己那 1/w 会让下一轮梯度累加到残值上（静默算错）。"""
    params = _params([4, 4, 4, 4])
    plan = Z.ShardPlan(params, 2, 0)
    inner = _FakeOpt(plan.owned)
    opt = Z.make_optimizer(inner, params, plan)
    for p in params:
        p.grad = NParam(np.ones(4, np.float32))
    opt.zero_grad()
    assert all(p.grad is None for p in params), "有参数没被清梯度"


def test_sharded_optimizer_proxies_lr_and_step():
    params = _params([4, 4])
    plan = Z.ShardPlan(params, 2, 1)
    inner = _FakeOpt(plan.owned)
    opt = Z.make_optimizer(inner, params, plan)
    opt.lr = 0.25
    assert inner.lr == 0.25 and opt.lr == 0.25
    opt.step(grad_scale=2.0)
    assert inner.stepped == 1
    assert opt.t == 1, "t 必须透传（bias correction 要用）"
    assert opt.bytes_per_param() == 3.0625, "bytes_per_param 必须透传"


@pytest.mark.parametrize("world", [2, 4, 8])
def test_comm_model_matches_the_table_in_the_docstring(world):
    """通信量口径：ZeRO-1 比 DP 多 50%，ZeRO-2 与 DP 持平。这条别被"优化"掉。"""
    n = 848_000_000
    f = (world - 1.0) / world
    m0 = Z.comm_model(n, world, 0)
    m1 = Z.comm_model(n, world, 1)
    m2 = Z.comm_model(n, world, 2)
    assert m0["one_way_bytes"] == pytest.approx(2 * n * 4 * f)
    assert m1["one_way_bytes"] == pytest.approx(3 * n * 4 * f)
    assert m2["one_way_bytes"] == pytest.approx(2 * n * 4 * f)
    assert m1["ratio_vs_dp"] == pytest.approx(1.5)
    assert m2["ratio_vs_dp"] == pytest.approx(1.0)
    assert m1["all_gather_bytes"] == pytest.approx(n * 4 * f)
    assert m2["reduce_scatter_bytes"] == pytest.approx(n * 4 * f)


def test_world_one_comm_is_zero():
    for z in Z.MODES:
        assert Z.comm_model(1000, 1, z)["one_way_bytes"] == 0.0
        assert Z.expected_comm_bytes(1000, 1, z) == 0


def test_chunk_from_env_is_capped_by_cap(monkeypatch):
    monkeypatch.delenv("TG_ZERO_CHUNK_MB", raising=False)
    assert Z.chunk_from_env(10) == 10, "cap 小于块大小时必须取 cap"
    big = Z.chunk_from_env(10 ** 12)
    assert big == (128 << 20) // 4, "默认块大小必须是 128 MiB/rank"
    monkeypatch.setenv("TG_ZERO_CHUNK_MB", "8")
    assert Z.chunk_from_env(10 ** 12) == (8 << 20) // 4
    monkeypatch.setenv("TG_ZERO_CHUNK_MB", "junk")
    assert Z.chunk_from_env(10 ** 12) == (128 << 20) // 4


def test_supported_opts_covers_the_default_engine_optimizer():
    """`--opt int8gpu` 是 ORV 的默认省内存档，ZeRO 必须支持它。"""
    assert "int8gpu" in Z.SUPPORTED_OPTS
    assert set(Z.SUPPORTED_OPTS) <= {"adamw", "fp16mom", "int8gpu", "muon", "adafactor"}


# ═════════════════════════════════════════════════════════════════════════
# 5) tied 参数去重 / 缺梯度补齐 / 训练会话外壳
# ═════════════════════════════════════════════════════════════════════════
def test_plan_dedups_tied_parameters_by_identity():
    """tied embedding（同一张量在列表里出现两次）必须只分给一个 rank。

    不去重的后果：同一块存储被两个 rank 各自更新，参数收回时互相覆盖 —— 静默算错。
    这不是假想：``P["head"] is P["emb"]`` 时 ``params_of`` 依赖去重（它也去重，
    这里再兜一层，防止别的模型/调用方忘了去重）。
    """
    shared = NParam(np.arange(6, dtype=np.float32))
    other = NParam(np.arange(4, dtype=np.float32))
    params = [shared, other, shared]            # 同一个对象出现两次
    plan = Z.ShardPlan(params, 2, 0)
    assert plan.n_params == 2, "参数没有被按 id() 去重：%r" % (plan.sizes,)
    assert plan.n_dup == 1
    assert plan.sizes == [6, 4]
    assert plan.total_elems == 10
    assert sum(plan.shard_elems) == 10
    # 所有 rank 的切片加起来恰好是去重后的参数表
    for r in range(2):
        p = Z.ShardPlan(params, 2, r)
        assert p.params is not p.all_params or p.n_dup == 0
        assert p.total_elems == 10


def test_ensure_grads_fills_missing_grads_with_zeros():
    """缺梯度必须补 0，不能跳过：DP 归约后每个参数都有梯度，gnorm 的张量表要一致。"""
    rt = FakeRT()
    params = _params([3, 5, 7], seed=21)
    for i, p in enumerate(params):
        if i != 1:
            p.set_grad(np.ones(params[i].numel(), np.float32))
    plan = Z.ShardPlan(params, 1, 0)
    n = Z.ensure_grads(plan, rt)
    assert n == 1, "只该补 1 个（第 1 个参数），实际补了 %d 个" % n
    for p in params:
        assert p.grad is not None
        assert p.grad._gpu.n == p.numel()
    assert np.all(params[1].grad._gpu.np == 0.0)
    assert np.all(params[0].grad._gpu.np == 1.0), "已有梯度不能被覆盖"


def test_zero_trainer_zero0_is_a_pure_noop():
    """``--zero 0`` 必须**零行为变化**：同一份参数、同一个优化器对象、动作全空。"""
    params = _params([4, 4], seed=22)
    zt = Z.ZeroTrainer(params, zero=0, world=2, rank=1)
    assert zt.enabled is False
    assert zt.shard_params is params, "zero=0 时不能返回分片列表"
    inner = _FakeOpt(params)
    assert zt.wrap(inner) is inner, "zero=0 时不能套外壳"
    assert zt.reduce_grads() is None
    assert zt.after_step() is None
    assert zt.describe() == "zero=0（不分片）"


def test_zero_trainer_rejects_single_process():
    with pytest.raises(ValueError) as ei:
        Z.ZeroTrainer(_params([4]), zero=1, world=1, rank=0)
    assert "需要多进程" in str(ei.value) and "修法" in str(ei.value)


def test_zero_trainer_rejects_unsupported_optimizer():
    with pytest.raises(ValueError) as ei:
        Z.ZeroTrainer(_params([4]), zero=2, world=2, rank=0, opt_name="muon")
    assert "不支持" in str(ei.value) and "修法" in str(ei.value)


def test_zero_trainer_rejects_bad_mode():
    with pytest.raises(ValueError) as ei:
        Z.ZeroTrainer(_params([4]), zero=3, world=2, rank=0)
    assert "不认识" in str(ei.value) and "修法" in str(ei.value)


def test_zero_trainer_shard_params_and_attach_check():
    params = _params([8, 8], seed=23)
    zt = Z.ZeroTrainer(params, zero=1, world=2, rank=1)
    assert len(zt.shard_params) == len(zt.plan.owned)
    assert zt.plan.total_elems == 16
    # rank 对不上的 DataParallel 必须当场报错（否则通信会错位，且不会自曝）
    with pytest.raises(ValueError) as ei:
        zt.attach(_FakeDP(2, 0))
    assert "不一致" in str(ei.value) and "修法" in str(ei.value)
    zt.attach(_FakeDP(2, 1))          # 对得上就通过
    assert zt.dp is not None


class _FakeDP(object):
    def __init__(self, world, rank):
        self.world = world
        self.rank = rank
        self.job = "fake"


def test_zero_trainer_describe_reports_the_honest_comm_ratio():
    """ZeRO-1 的通信量必须写「DP 的 1.50x」，ZeRO-2 是 1.00x —— 不许写成更省。"""
    params = _params([1000, 1000], seed=24)
    z1 = Z.ZeroTrainer(params, zero=1, world=2, rank=0).describe()
    z2 = Z.ZeroTrainer(params, zero=2, world=2, rank=0).describe()
    assert "1.50x" in z1, z1
    assert "1.00x" in z2, z2
    assert "int8" not in z1 or "状态" in z1


# ═════════════════════════════════════════════════════════════════════════
# 6) 分片集合的互补性 / 显存与通信算术（与 plan.py 对拍）/ 纯 numpy 的 all_gather
# ═════════════════════════════════════════════════════════════════════════
def test_shard_partition_is_complementary_and_covers_every_param_once():
    """**各 rank 拿到的参数集合必须互补且不重叠**，并集 = 去重后的参数表。

    这条比「区间合法」更强：区间合法仍可能两个 rank 分到同一块存储
    （tied 参数没去重时就会），那会让同一个张量被更新两次、收回时互相覆盖，
    而且**不会报错**。这里按对象身份逐个核对。
    """
    shared = NParam(np.arange(8, dtype=np.float32))       # tied：出现两次
    other = [NParam(np.arange(n, dtype=np.float32)) for n in (3, 5, 2)]
    params = [shared, other[0], shared, other[1], other[2]]
    dedup = [shared, other[0], other[1], other[2]]
    for world in (2, 3, 4):
        seen = {}
        for rank in range(world):
            plan = Z.ShardPlan(params, world, rank)
            for p in plan.owned:
                key = id(p)
                assert key not in seen, (
                    "world=%d：参数被 rank%d 与 rank%d **同时**拥有（tied 没去重？）"
                    % (world, seen[key], rank))
                seen[key] = rank
        assert len(seen) == len(dedup), (
            "world=%d：分片并集少了参数：%d != %d" % (world, len(seen), len(dedup)))
        assert set(seen) == set(id(p) for p in dedup), "分片里出现了不该有的参数"
        # 每个 rank 的切片 = 参数表里一段**连续**下标
        for rank in range(world):
            plan = Z.ShardPlan(params, world, rank)
            assert plan.owned == plan.params[plan.i0:plan.i1]


def test_estimate_memory_matches_plan_module():
    """每卡字节数必须与 `plan.estimate_memory` 的算术**对得上**（防止两边漂）。

    `plan.py` 是仓库里的独立规划器（DRY：两边都别自己发明公式）。对齐关系：
      * zero=0 ↔ plan(默认)
      * zero=1 ↔ plan(shard_opt=True)
      * zero=2 的反向结束态 ↔ plan(shard_opt=True)
      * zero=2 的稳态 ↔ plan(shard_opt=True, shard_grad=True)

    ⚠️ `int8gpu` 两边刻意留了 0.1% 的口径差：`plan.BYTES_PER_PARAM` 把它记成
    **3.06**（规划器用的近似），而真实值是 `1 + 2 + 4/64 = 3.0625`（见
    `int8_adamw_gpu.bytes_per_param`）。ze**ro.py 用精确值**，所以这里按 rel≤1e-3
    对拍并额外断言精确公式，而不是把 3.0625 改成 3.06 去凑相等。
    """
    from tgai_go.train import plan as P
    n = 2_548_000_000
    GIB = float(1 << 30)
    for opt in ("int8gpu", "adamw"):
        rel = 1e-3 if opt == "int8gpu" else 1e-12
        for w in (1, 2, 4):
            a = Z.estimate_memory(n, w, 0, opt)
            b = P.estimate_memory(n, opt, w)
            assert a == pytest.approx(b, rel=rel), (opt, w, a, b)
            a1 = Z.estimate_memory(n, w, 1, opt)
            b1 = P.estimate_memory(n, opt, w, shard_opt=True)
            assert a1 == pytest.approx(b1, rel=rel), (opt, w, a1, b1)
            a2p = Z.estimate_memory(n, w, 2, opt)
            # 结构性判据（同口径内逐位相等）：ZeRO-2 的峰值 == ZeRO-1 的峰值
            assert a2p == pytest.approx(a1), "ZeRO-2 的峰值必须等于 ZeRO-1 的峰值"
            assert a2p == pytest.approx(b1, rel=rel), (opt, w, a2p, b1)
            a2s = Z.estimate_memory(n, w, 2, opt, shard_grad=True)
            b2 = P.estimate_memory(n, opt, w, shard_opt=True, shard_grad=True)
            assert a2s == pytest.approx(b2, rel=rel), (opt, w, a2s, b2)
    # 精确公式（与 int8_adamw_gpu.bytes_per_param() = 1+2+4/64 = 3.0625 同源）
    assert Z.BYTES_PER_PARAM["int8gpu"] == 1.0 + 2.0 + 4.0 / 64.0
    assert Z.estimate_memory(n, 1, 0, "int8gpu")[0] == \
        pytest.approx(n * (4.0 + 4.0 + 3.0625) / GIB)
    # fp16mom 不在 plan.BYTES_PER_PARAM 里，只验内部分式（bf16 动量 = 4 B/参数）
    assert Z.estimate_memory(n, 1, 0, "fp16mom")[0] == \
        pytest.approx(n * (4.0 + 4.0 + 4.0) / GIB)


def test_estimate_memory_rejects_unknown_optimizer_and_mode():
    with pytest.raises(ValueError) as ei:
        Z.estimate_memory(1000, 2, 0, "sgd_magic")
    assert "修法" in str(ei.value)
    with pytest.raises(ValueError) as ei2:
        Z.estimate_memory(1000, 2, 3)
    assert "修法" in str(ei2.value)


def test_bus_comm_bytes_is_the_pcie_side_and_never_claims_sharding_is_cheaper():
    """总线口径（d2h+h2d）：ZeRO-1 比 DP **多**，ZeRO-2 比 DP 多（不是更省）。

    这条是给"别把分片写成省通信"上锁的：三个档位的相对关系必须钉死。
    """
    n = 848_645_257
    w = 2
    dp = Z.bus_comm_bytes(n, w, 0)
    z1 = Z.bus_comm_bytes(n, w, 1)
    z2 = Z.bus_comm_bytes(n, w, 2)
    N = n * 4
    assert dp == pytest.approx(2 * N)
    assert z1 == pytest.approx(3 * N), "ZeRO-1 = allreduce 2N + gather N"
    assert z2 == pytest.approx(2.5 * N), "ZeRO-2 = rs 1.5N + gather N"
    assert z1 > dp and z2 > dp, "分片不省通信"
    assert Z.bus_comm_bytes(n, 1, 1) == 0, "world=1 不该有任何通信"


def test_all_gather_numpy_simulation_restores_parameters_bitwise():
    """纯 numpy 模拟 all_gather：每 rank 只发自己那 1/w，收回后参数**逐位相同**。

    这是「all_gather 不改数值」的最小证明：把三个 rank 的切片依次 pack 出来、再解包
    回一张**全新的**参数表，每个元素必须与分片前逐位相等（任何偏移/覆盖错误都会抓到）。
    """
    sizes = [6, 9, 4, 7]
    base = _params(sizes, seed=31)
    rt = FakeRT()
    for world in (2, 3):
        sends = []
        for rank in range(world):
            plan = Z.ShardPlan(base, world, rank)
            buf = NBuf(np.zeros(max(1, plan.cap), np.float32))
            Z.pack_shard(rt, plan, rank, 0, plan.cap, buf)
            sends.append((plan, buf))
        restored = [NParam(np.zeros(n, np.float32)) for n in sizes]
        p_dst = Z.ShardPlan(restored, world, 0)
        for r, (p_src, buf) in enumerate(sends):
            Z.unpack_shard(rt, p_dst, r, 0, p_src.cap,
                           rt.sub_view(buf, (p_src.cap,), 0))
        for i in range(len(sizes)):
            assert np.array_equal(restored[i].numpy(), base[i].numpy()), (
                "world=%d 第 %d 个张量 after all_gather 不再逐位相同" % (world, i))


