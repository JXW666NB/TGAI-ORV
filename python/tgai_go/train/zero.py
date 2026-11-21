# -*- coding: utf-8 -*-
"""ZeRO-1 / ZeRO-2：把**优化器状态**（ZeRO-1）和**梯度**（ZeRO-2）按 rank 分片。

参数仍然每卡一份（复制）—— 这是 ZeRO-1/2 的定义。参数分片（ZeRO-3）不在这里，
见 ``docs/ZERO_PLAN.md``。

## 为什么要它（实测约束，不是理论）

RTX 5090 31.4 GiB / 卡，``--opt int8gpu`` 是 3.0625 字节/参数、fp32 参数与梯度各
4 字节/参数 ⇒ 全参训练每卡要 ``4 + 4 + 3.0625 = 11.0625`` 字节/参数。实测 20 层
V3-MoE（1.82B）峰值 22.51 GiB；28 层（2.55B）在 **28.04 GiB 处 OOM**。
数据并行是「每卡一份完整模型 + 状态」，加卡**不减少**每卡占用 ⇒ 必须分片。

## 设计口径（先正确，再快）

1. **按「整张量」切，且每个 rank 的切片是参数表里的连续区间。**
   为什么必须整张量：``Int8AdamWGpu`` 的 m/v 量化块是 ``block=64`` **按张量**分的
   （``nb = ceil(n/block)``）。若把一个张量劈成两半，块边界就变了 ⇒ 量化 scale 变了
   ⇒ **不可能逐位一致**。整张量分配则每个张量的 m/v/scale 与不分片时逐位相同。
   为什么必须连续：ZeRO-2 的 gnorm 要复现「按张量下标顺序累加」，而
   ``reduce_scatter`` 的分段是连续的；两者一致才能同时做到「通信按段切」和
   「范数按原顺序」。
2. **参数分片收回用 ``all_gather``；梯度归约 ZeRO-1 用 ``all_reduce``、ZeRO-2 用
   ``reduce_scatter``。**
3. **逐位一致是验收判据，不是"差不多"。** ZeRO-1：梯度在每卡上都是完整归约结果，
   范数与裁剪和原来一模一样；优化器更新是逐元素算术，状态按整张量分片 ⇒ 每个参数
   的更新值与不分片时逐位相同。ZeRO-2：梯度被切开了，但 gnorm 用
   「逐张量分块平方和 + 按原顺序归约」（见 :func:`grad_ssq_total`）复现原值。
4. **按 ``id()`` 去重后分配。** tied embedding（``P["head"] is P["emb"]``）在
   ``params`` 里可能出现两次；同一块存储若被分给两个 rank，参数收回时会把一个 rank
   的更新覆盖到另一个上（静默算错）。``v3_train_cli.params_of`` 默认已去重，这里
   再做一次是防御（排序与内容对默认路径零变化）。

## 通信量（**如实**，单位是"单向字节数 / rank"，N = 梯度字节数 = 参数量×4）

| 模式 | 归约 | 参数回收 | 合计 | w=2 |
| --- | --- | --- | --- | --- |
| ``--zero 0``（数据并行） | all_reduce ``2N(w-1)/w`` | — | ``2N(w-1)/w`` | **N** |
| ``--zero 1`` | all_reduce ``2N(w-1)/w`` | all_gather ``N(w-1)/w`` | ``3N(w-1)/w`` | **1.5N** |
| ``--zero 2`` | reduce_scatter ``N(w-1)/w`` | all_gather ``N(w-1)/w`` | ``2N(w-1)/w`` | **N** |

⇒ **ZeRO-1 的通信量比数据并行多 50%**（这是它的代价，不是 bug）；ZeRO-2 与数据并行
**持平**。别指望分片更省通信 —— 它省的是显存。实测 3.16 GiB 梯度在这台 2×5090
（``p2p=False``）上 all_reduce ≈ 140.8~148.0 ms ⇒ 有效单向 ~24 GB/s。
ZeRO-1 于 2.55B（**按 DP 实测带宽推算，未实测**）：all_reduce 9.48 GiB ≈ 0.42~0.45 s
+ all_gather 4.74 GiB ≈ 0.21~0.23 s，合计 ≈ **0.63~0.68 s/步通信** —— 比 DP 多 50%；
而每卡算的还是一份 batch ⇒ **本机上 ZeRO 是"装得下"而不是"更快"**（`plan.judge`
口径下 2 卡吞吐比 ≈0.5×，净亏）。

## 显存账（每卡）

* ZeRO-1 省 ``(w-1)/w × 优化器状态``（``int8gpu`` 下 3.0625 B/参数）；
* ZeRO-2 在 ZeRO-1 之上再省 ``(w-1)/w × 梯度``，**但只省在反向之后**：
  峰值出现在 ``backward()`` 刚结束的那一刻（此时所有梯度都已存在），
  所以**当前步结构下 ZeRO-2 的峰值与 ZeRO-1 相同**。要真正把梯度峰值压下来必须
  反向过程中分桶归约（DDP 式 hook），那会改变归约顺序、破坏逐位一致，
  不在本轮范围内 —— ``docs/ZERO_PLAN.md`` 里写清了代价。

## 传输：**本模块自带，不改 dist.py / dist_nccl.py**

``all_gather`` / ``reduce_scatter`` / 小数组归约都走**主机共享内存**（与 ORV DP 的
host 路径同一套算法：pinned 缓冲 → 共享内存 → 各 rank 按 rank 轴求和/拼接）。
为什么不自带一份 ctypes NCCL：这台机器 ``p2p=False``，实测 NCCL 与本机 host 路径
只差 **5%**（3.16 GiB: 140.78 vs 148.03 ms），而 ZeRO 的通信量本来就更大；
多维护一条 NCCL 代码路径的收益抵不过验证成本。需要 NCCL 时可把
:class:`_Xchg` 的两个方法换成 `ncclAllGather` / `ncclReduceScatter`，调用点不变。

## 用法（v3_pretrain 接线见 ``docs/ZERO_PLAN.md`` 的 snippet）

```python
from tgai_go.train import zero as Z

zt = Z.ZeroTrainer(params, zero=2)        # world/rank 从 TGAI_DP_WORLD/RANK 读
opt = Int8AdamWGpu(zt.shard_params, ...)  # ★ 用分片后的参数建优化器
opt = zt.wrap(opt)                        # 外壳：zero_grad 清全部参数 + lr 透传
...                                       # 建 DataParallel 之后：
zt.attach(dp)
...
loss.backward()
zt.reduce_grads()                         # 归约（ZeRO-1 all_reduce / ZeRO-2 reduce_scatter）
gnorm, gscale = zt.grad_norm_scale(a.grad_clip, defer=True)
opt.step(grad_scale=gscale)
zt.after_step()                           # all_gather 参数
```
"""
import ctypes
import os

import numpy as np

__all__ = [
    "MODES", "SUPPORTED_OPTS", "BYTES_PER_PARAM", "ShardPlan",
    "ShardedOptimizer", "ZeroStats", "ZeroTrainer", "plan_ranges",
    "make_optimizer", "broadcast_start", "gather_params",
    "reduce_scatter_grads", "free_unowned_grads", "grad_ssq_total",
    "grad_norm_scale", "comm_model", "bus_comm_bytes", "estimate_memory",
    "chunk_from_env", "ensure_grads",
]

#: 支持的 ZeRO 档位。0 = 现状（不分片），1 = 分片优化器状态，2 = 再分片梯度。
MODES = (0, 1, 2)

#: ZeRO 只对**逐元素、状态按张量存放**的优化器开放。
#: 为什么：分片后的更新必须与不分片时逐位相同，前提是「每个参数的更新只依赖它自己的
#: 梯度和自己的状态」。Muon 的 Newton-Schulz 作用在整张矩阵上（整张量切仍然成立，
#: 但尚未实测）；Adafactor 的相对步长依赖全局量。没验过的不开放 —— 宁可少支持，
#: 不能静默算错。
SUPPORTED_OPTS = ("adamw", "fp16mom", "int8gpu")

#: 分片规划里「参数表被去重」的标记（测试用）。
_DEDUP_NOTE = "ShardPlan 按 id() 去重 tied 参数"


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _rt():
    import tgai_go.train as T
    return T._cuda()


def _now():
    import time
    return time.perf_counter()


def _dedup(params):
    """按 ``id()`` 去重（tied embedding 在 ``parameters()`` 里可能出现两次）。"""
    out, seen = [], set()
    for p in params:
        if p is None:
            continue
        k = id(p)
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


# ═════════════════════════════════════════════════════════════════════════
# 1) 分片规划（纯函数，可在 CPU 上单测）
# ═════════════════════════════════════════════════════════════════════════
def plan_ranges(sizes, world):
    """把参数下标 ``0..n-1`` 切成 ``world`` 段**连续**区间，按元素数尽量均衡。

    返回 ``[(start, end), ...]``（``end`` 独占）。同一个 ``sizes`` 在任何 rank 上
    都给出**完全相同**的结果 —— 这是整件事的地基：各 rank 对「谁拥有哪些参数」
    必须零歧义，否则收回参数时会把 A 的切片写到 B 上，而且**不会报错**。

    切点取「累计元素数最接近 ``k*total/world``」的张量边界；允许空段
    （参数个数比卡还少时）。
    """
    sizes = [int(s) for s in sizes]
    n = len(sizes)
    world = int(world)
    if world < 1:
        raise ValueError("world 必须 >= 1（当前 %d）" % world +
                         "。修法：--dp 至少为 1")
    if n == 0:
        return [(0, 0)] * world
    total = int(sum(sizes))
    cum = [0] * (n + 1)
    for i, s in enumerate(sizes):
        cum[i + 1] = cum[i] + s
    if total <= 0:
        # 全是空张量：只按个数切，保证每 rank 一段
        cuts = [min(n, (n * k) // world) for k in range(world)] + [n]
        return [(cuts[k], cuts[k + 1]) for k in range(world)]
    cuts = [0]
    for k in range(1, world):
        want = total * k // world
        lo = cuts[-1]
        best = lo
        bestd = abs(cum[lo] - want)
        for i in range(lo, n + 1):
            d = abs(cum[i] - want)
            if d < bestd:
                best, bestd = i, d
        cuts.append(best)
    cuts.append(n)
    return [(int(cuts[k]), int(cuts[k + 1])) for k in range(world)]


class ShardPlan(object):
    """一个 rank 的分片视图：谁拥有哪些参数、怎么在「平铺元素轴」上定位它们。

    属性全是**只读派生量**，构造后不再变化，因此可以安全地在每个 rank 上独立重建
    （只要 ``params`` 的顺序与 ``numel()`` 一致）。
    """

    def __init__(self, params, world, rank):
        self.all_params = list(params)          # 调用方原始列表（可能含 tied 重复）
        self.params = _dedup(params)            # ★ 分片按去重后的参数分配
        self.n_dup = len(self.all_params) - len(self.params)
        self.world = int(world)
        self.rank = int(rank)
        if not (0 <= self.rank < self.world):
            raise ValueError("rank=%d 越界（world=%d）" % (self.rank, self.world) +
                             "。修法：TGAI_DP_RANK 必须落在 [0, world)")
        self.sizes = [int(p.numel()) for p in self.params]
        self.n_params = len(self.sizes)
        self.total_elems = int(sum(self.sizes))
        self.cum = [0] * (self.n_params + 1)
        for i, s in enumerate(self.sizes):
            self.cum[i + 1] = self.cum[i] + s
        self.ranges = plan_ranges(self.sizes, self.world)
        self.shard_elems = [self.cum[e] - self.cum[s] for (s, e) in self.ranges]
        #: 每个 rank 的切片元素数必须对齐到这个上界（集合通信要求等长）
        self.cap = int(max(self.shard_elems)) if self.shard_elems else 0
        self.i0, self.i1 = self.ranges[self.rank]
        self.owned = self.params[self.i0:self.i1]
        self.owned_elems = int(self.shard_elems[self.rank])
        self.owned_bytes = self.owned_elems * 4

    # ---- 报告用 ----
    def describe(self):
        parts = []
        for r, (s, e) in enumerate(self.ranges):
            parts.append("r%d=[%d,%d)%d" % (r, s, e, self.shard_elems[r]))
        return ("world=%d rank=%d tensors=%d(dup=%d) total=%.3fB owned=%.3fB "
                "(%.1f%%) cap=%.3fB | %s"
                % (self.world, self.rank, self.n_params, self.n_dup,
                   self.total_elems / 1e9, self.owned_elems / 1e9,
                   100.0 * self.owned_elems / max(1, self.total_elems),
                   self.cap / 1e9, " ".join(parts)))

    # ---- 分片内元素区间 -> (参数, 段) 的映射 ----
    def segments(self, shard, lo, hi):
        """产出 ``(param_index, src_off, dst_off, n)``，把 shard 的 ``[lo, hi)`` 拆成段。

        偏移都以**元素**为单位；``src_off`` 是参数内部的偏移，``dst_off`` 是相对
        ``lo`` 的偏移。没有任何一个元素会被覆盖两次或漏掉（单测守着这一条）。
        """
        i0, i1 = self.ranges[int(shard)]
        base = self.cum[i0]
        for i in range(i0, i1):
            n = self.sizes[i]
            if n <= 0:
                continue
            a = self.cum[i] - base
            b = a + n
            if b <= lo or a >= hi:
                continue
            s = max(a, lo)
            e = min(b, hi)
            yield i, (s - a), (s - lo), (e - s)


# ═════════════════════════════════════════════════════════════════════════
# 2) 打包 / 写回（设备内拷贝；`rt` 可注入，CPU 单测传假运行时）
# ═════════════════════════════════════════════════════════════════════════
def _buf_of(t):
    """取张量的设备缓冲。ZeRO 只支持 CUDA 张量（``_gpu`` 必须存在）。"""
    g = getattr(t, "_gpu", None)
    if g is None:
        raise RuntimeError(
            "ZeRO 需要 CUDA 张量，但拿到了没有 _gpu 的（device=%r）"
            % (getattr(t, "device", "?"),) +
            "。修法：ZeRO 只能在 CUDA 上开（--zero 0 跑 CPU/其它后端）")
    return g


def _take(rt, src, n_elems, src_off, dst, dst_off):
    if n_elems <= 0:
        return
    rt.copy_d2d(dst, src, n_elems * 4, dst_off=int(dst_off) * 4,
                src_off=int(src_off) * 4)


def _zero_range(rt, buf, n_elems, off=0):
    """把 buf[off : off+n_elems) 清零（走 `sub_view` + `_memset32`，假运行时也支持）。"""
    n = int(n_elems)
    if n <= 0:
        return
    rt._memset32(rt.sub_view(buf, (n,), int(off)), 0)


def pack_shard(rt, plan, shard, lo, hi, dst):
    """把 shard 的 ``[lo, hi)`` 元素打包到 ``dst[0 : hi-lo]``（**参数**）。

    超出该 rank 实际切片长度的尾巴会被**清零**：它是 ``cap`` 对齐用的 padding，
    收端不读它，但清零能让「线上看到 nan」这类排查噪音消失。
    """
    for i, so, do, n in plan.segments(shard, lo, hi):
        _take(rt, _buf_of(plan.params[i]), n, so, dst, do)
    have = plan.shard_elems[int(shard)]
    if hi > have:
        _zero_range(rt, dst, hi - max(lo, have), max(lo, have) - lo)


def unpack_shard(rt, plan, shard, lo, hi, src):
    """把 ``src[0 : hi-lo]`` 写回 shard 的 ``[lo, hi)`` 元素（**参数**）。"""
    for i, so, do, n in plan.segments(shard, lo, hi):
        _take(rt, src, n, do, _buf_of(plan.params[i]), so)


def pack_grads(rt, plan, shard, lo, hi, dst):
    """同 :func:`pack_shard`，但取的是**梯度**。缺梯度按 0 参与（与 DP 语义一致）。"""
    for i, so, do, n in plan.segments(shard, lo, hi):
        p = plan.params[i]
        g = getattr(p, "grad", None)
        if g is None:
            _zero_range(rt, dst, n, do)
            continue
        _take(rt, _buf_of(g), n, so, dst, do)
    have = plan.shard_elems[int(shard)]
    if hi > have:
        _zero_range(rt, dst, hi - max(lo, have), max(lo, have) - lo)


def unpack_grads(rt, plan, shard, lo, hi, src):
    """把 ``src[0 : hi-lo]`` **覆盖写**进 shard 的梯度（缺梯度就新建一块 0 张量）。"""
    for i, so, do, n in plan.segments(shard, lo, hi):
        p = plan.params[i]
        g = _ensure_grad(p, rt)
        _take(rt, src, n, do, _buf_of(g), so)


def _new_grad(p, rt):
    """给参数新建一块**零**梯度张量并挂上 ``p.grad``。"""
    import tgai_go.train as T
    shape = tuple(int(s) for s in getattr(p, "shape", ()) or ())
    n = int(np.prod(shape)) if shape else int(p.numel())
    buf = rt.zeros((max(1, n),), dtype=np.float32)
    val = T.Tensor._from_gpu(buf, requires_grad=False)
    val = val.reshape(shape) if len(shape) > 1 else val
    p.grad = val
    return val


def _ensure_grad(p, rt):
    """保证参数有一块梯度缓冲（缺失时分配一块并挂上 ``p.grad``）。

    为什么要这一步：数据并行里「某个参数没梯度」会按 0 参与归约，然后优化器
    **仍然会对它做 weight decay**（`dist._all_reduce_host` / `_all_reduce_nccl`
    都会把它补成 0 梯度）。ZeRO-2 若让它保持 ``None``，优化器会跳过它 ⇒ 与不分片
    版本分叉。所以必须建 0 梯度。
    """
    g = getattr(p, "grad", None)
    if g is not None:
        return g
    return _new_grad(p, rt)


def ensure_grads(plan, rt=None):
    """把**全部**参数的缺失梯度补成 0（ZeRO-2 归约前必须做）。

    与数据并行路径保持一致：归约之后每个参数都有梯度（缺的补 0），所以
    ``clip_grad_norm_`` 枚举到的张量集合、顺序、分块表完全相同 —— 这是 ZeRO-2 的
    gnorm 能与 ZeRO-0 逐位相等的**前提之一**（另一个是 :func:`grad_ssq_total` 的
    槽位技巧）。返回新建的梯度个数（空张量不补：参考路径也把它过滤掉）。
    """
    if rt is None:
        rt = _rt()
    n = 0
    for p in plan.params:
        if getattr(p, "grad", None) is not None:
            continue
        if int(p.numel()) <= 0:
            continue
        _new_grad(p, rt)
        n += 1
    return n


# ═════════════════════════════════════════════════════════════════════════
# 3) 传输：主机共享内存（自包含；不依赖 dist.py 的新增方法）
# ═════════════════════════════════════════════════════════════════════════
def chunk_from_env(cap, chunk=None):
    """分块大小（元素）。默认 ``TG_ZERO_CHUNK_MB=128`` 的每 rank 分片块。

    为什么必须分块：``all_gather`` 的收包缓冲是 ``world × chunk``，不分块就是
    ``world × cap ≈ 整个参数表``（3B 上 11 GiB），比不分片还费显存。
    默认 128 MiB/rank 分片块 ⇒ w=2 时设备侧收包缓冲 256 MiB、主机侧共享内存 256 MiB。
    （再大就省 barrier 次数，但显存/主机内存也线性涨；256 MiB 档实测更快的机器
    可以把 ``TG_ZERO_CHUNK_MB`` 调到 256。）
    """
    if chunk is None:
        mb = max(1, _env_int("TG_ZERO_CHUNK_MB", 128))
        chunk = (mb << 20) // 4
    return max(1, min(int(cap), int(chunk)))


def _shm_tag(dp):
    job = str(getattr(dp, "job", "default") or "default")
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in job)[:64]


class _Xchg(object):
    """两台/多台进程之间的「主机中转」集合通信（每 rank 一个实例，绑定同一个 dp）。

    ## 为什么不直接用 dist.py 的方法

    任务约束要求 ``zero.py`` **自包含**、不改 ``dist.py`` / ``dist_nccl.py``，所以
    这里只依赖 :class:`dist.DataParallel` 的**既有**基础设施：``world``/``rank``/
    ``job``、``_start()``（rendezvous）、``_barrier(tag)``（文件屏障）、``_shm``
    （统一回收共享内存）。集合通信本体在本模块实现。

    ## 为什么主机中转而不是 NCCL

    这台 2×5090 ``p2p=False``（消费卡无 P2P），梯度只能过主机内存；实测同 payload
    下 NCCL 只比 host 路径快 5%（140.78 vs 148.03 ms / 3.16 GiB）。多一条 NCCL 代码
    路径的验证成本 > 5% 收益。要换 NCCL 只需重写本类的两个方法，调用点不变。
    """

    def __init__(self, dp):
        self.dp = dp
        self._slots = {}          # key -> (SharedMemory, per_rank_bytes)
        self._regs = {}           # key -> _PinnedHost（把 shm 注册成页锁定后的句柄）
        self._own = []            # dp 没有 _shm 时自己兜底回收

    # ---- 把共享内存注册成页锁定（d2h 可直接写进去，省一次整块主机拷贝）----
    def _reg(self, key, rt):
        """``rt.host_register(slot.buf)``：成功则返回 ``_PinnedHost``，失败返回 None。

        为什么重要：不注册就得「d2h 到 pinned → copyto 进 shm」多一趟主机拷贝
        （HEAD 的 DP 路径实测这一趟 3.16 GiB 约 0.3 s）。注册失败退回老路（正确性不变）。
        """
        if key in self._regs:
            return self._regs[key]
        ent = self._slots.get(key)
        if ent is None or not hasattr(rt, "host_register"):
            return None
        slot, need = ent
        try:
            reg = rt.host_register(slot.buf, int(need) * int(self.dp.world))
        except Exception:                            # noqa: BLE001
            reg = None
        self._regs[key] = reg
        return reg

    # ---- 共享内存槽（按用途名复用，只在变大时重建）----
    def _slot(self, key, per_rank_bytes):
        w = int(self.dp.world)
        need = int(per_rank_bytes)
        ent = self._slots.get(key)
        if ent is not None and int(ent[1]) >= need:
            return ent[0]
        if ent is not None:
            try:
                ent[0].close()
            except Exception:                        # noqa: BLE001
                pass
        from multiprocessing import shared_memory
        name = "tgai_zero_%s_%s_%d" % (_shm_tag(self.dp), key, need)
        if int(self.dp.rank) == 0:
            # 清掉上一轮崩溃留下的同名残留（否则 create=True 抛 FileExistsError）
            try:
                old = shared_memory.SharedMemory(name=name)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
            slot = shared_memory.SharedMemory(name=name, create=True,
                                              size=need * w)
        else:
            t0 = _now()
            while True:
                try:
                    slot = shared_memory.SharedMemory(name=name, create=False)
                    break
                except FileNotFoundError:
                    if _now() - t0 > float(getattr(self.dp, "timeout", 600.0)):
                        raise TimeoutError(
                            "共享内存 %s 未创建（ZeRO 的 %s 缓冲）" % (name, key) +
                            "。修法：确认所有 rank 都启动了、TGAI_DP_JOB 一致；"
                            "若有 rank 提前退出，其余 rank 会在这里超时")
                    import time
                    time.sleep(0.002)
        try:
            self.dp._shm.append(slot)
        except Exception:                            # noqa: BLE001 - 假 dp
            self._own.append(slot)
        self._slots[key] = (slot, need)
        return slot

    def close(self):
        for slot, _ in list(self._slots.values()):
            try:
                slot.close()
            except Exception:                        # noqa: BLE001
                pass
            try:
                if int(self.dp.rank) == 0:
                    slot.unlink()
            except Exception:                        # noqa: BLE001
                pass
        for slot in self._own:
            try:
                slot.close()
            except Exception:                        # noqa: BLE001
                pass
            try:
                slot.unlink()
            except Exception:                        # noqa: BLE001
                pass
        self._slots = {}
        self._regs = {}
        self._own = []

    # ---- 三个集合通信 ----
    def all_gather(self, send, recv, count, rt):
        """``recv[r*count : (r+1)*count] = 第 r 个 rank 的 send[0:count]``（fp32）。"""
        count = int(count)
        if count <= 0:
            return
        dp = self.dp
        w = int(dp.world)
        r = int(dp.rank)
        dp._start()
        span = w * count
        slot = self._slot("g", count * 4)             # 总大小 = w*count*4
        shm = np.ndarray((span,), np.float32, buffer=slot.buf)
        reg = self._reg("g", rt)
        if reg is not None:
            # ★ 直接 d2h 进共享内存自己那一段（省一次 pinned→shm 拷贝）
            rt.d2h_into(send, ctypes.c_void_p(reg.ptr.value + r * count * 4),
                        nbytes=count * 4)
        else:
            hp = rt.host_pinned(count * 4, slot="zero_gather")
            arrs = hp.array(np.float32, count)
            rt.d2h_into(send, ctypes.c_void_p(hp.ptr.value), nbytes=count * 4)
            np.copyto(shm[r * count:(r + 1) * count], arrs)   # ★ 只写自己那一段
        dp._barrier("zg")
        # ★ 只 h2d「别人那 w-1 段」：自己那段已经在参数表里、且下一步前向不需要
        #   从 stage 取回（省掉 1/w 的回传带宽 —— w=2 时 gather 回传减半）。
        if reg is not None:
            for rr in range(w):
                if rr == r:
                    continue
                rt.h2d_into(rt.sub_view(recv, (count,), rr * count),
                            ctypes.c_void_p(reg.ptr.value + rr * count * 4),
                            nbytes=count * 4)
        else:
            hp2 = rt.host_pinned(span * 4, slot="zero_gather2")
            arr = hp2.array(np.float32, span)
            np.copyto(arr, shm)
            for rr in range(w):
                if rr == r:
                    continue
                rt.h2d_into(rt.sub_view(recv, (count,), rr * count),
                            ctypes.c_void_p(hp2.ptr.value + rr * count * 4),
                            nbytes=count * 4)
        # ★ 第二个 barrier 不能省：快的 rank 若先进入下一块的写入，
        #   会把慢 rank 正在读的那一段覆盖掉（静默算错，不是卡死）。
        dp._barrier("zg2")

    def reduce_scatter(self, send, recv, count, scale, rt):
        """``recv[i] = sum_r send[r*count + i] * scale``（fp32，默认乘 ``1/world``）。"""
        count = int(count)
        if count <= 0:
            return
        dp = self.dp
        w = int(dp.world)
        r = int(dp.rank)
        dp._start()
        span = w * count
        slot = self._slot("r", span * 4)              # 总大小 = w*span*4
        shm = np.ndarray((w, span), np.float32, buffer=slot.buf)
        reg = self._reg("r", rt)
        if reg is not None:
            rt.d2h_into(send, ctypes.c_void_p(reg.ptr.value + r * span * 4),
                        nbytes=span * 4)
        else:
            hp = rt.host_pinned(span * 4, slot="zero_rs")
            arrs = hp.array(np.float32, span)
            rt.d2h_into(send, ctypes.c_void_p(hp.ptr.value), nbytes=span * 4)
            np.copyto(shm[r], arrs)                   # ★ 只写自己那一行
        dp._barrier("zr")
        acc = np.empty(count, np.float32)
        np.sum(shm[:, r * count:(r + 1) * count], axis=0, out=acc)
        if float(scale) != 1.0:
            acc *= np.float32(float(scale))
        oslot = self._slot("o", count * 4)            # 只放 count 个元素的出口
        oreg = self._reg("o", rt)
        if oreg is not None:
            out_view = np.ndarray((w, count), np.float32, buffer=oslot.buf)[r]
            np.copyto(out_view, acc)
            src = ctypes.c_void_p(oreg.ptr.value + r * count * 4)
        else:
            hop = rt.host_pinned(count * 4, slot="zero_rs_out")
            aop = hop.array(np.float32, count)
            np.copyto(aop, acc)
            src = ctypes.c_void_p(hop.ptr.value)
        rt.h2d_into(recv, src, nbytes=count * 4)
        dp._barrier("zr2")

    def sum_array(self, arr):
        """小 float32 主机数组在所有 rank 上**求和**（控制量/统计量专用，MB 级）。"""
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32)).reshape(-1)
        n = int(a.size)
        dp = self.dp
        w = int(dp.world)
        if n <= 0 or w <= 1:
            return a.copy()
        dp._start()
        slot = self._slot("a", n * 4)                 # 总大小 = w*n*4
        shm = np.ndarray((w, n), np.float32, buffer=slot.buf)
        shm[int(dp.rank)] = a
        dp._barrier("za")
        out = np.empty(n, np.float32)
        np.sum(shm, axis=0, out=out)
        dp._barrier("za2")
        return out


def _xchg_of(dp):
    x = getattr(dp, "_zero_xchg", None)
    if x is None:
        x = _Xchg(dp)
        try:
            dp._zero_xchg = x
        except Exception:                            # noqa: BLE001
            pass
    return x


class ZeroStats(object):
    """统计：**单向**收发元素数（与 `comm_model` 的口径一致，便于和实测对表）。"""

    def __init__(self):
        self.gather_send = 0
        self.gather_recv = 0
        self.rs_send = 0
        self.rs_recv = 0
        self.ar_send = 0
        self.ar_recv = 0
        self.gather_sec = 0.0
        self.rs_sec = 0.0
        self.small_sec = 0.0

    def one_way_elems(self):
        return int(max(self.gather_send + self.rs_send + self.ar_send,
                       self.gather_recv + self.rs_recv + self.ar_recv))

    def as_dict(self):
        return {"gather_send_elems": int(self.gather_send),
                "gather_recv_elems": int(self.gather_recv),
                "reduce_scatter_send_elems": int(self.rs_send),
                "reduce_scatter_recv_elems": int(self.rs_recv),
                "small_allreduce_send_elems": int(self.ar_send),
                "one_way_elems": int(self.one_way_elems()),
                "one_way_bytes": int(self.one_way_elems() * 4),
                "gather_sec": round(float(self.gather_sec), 6),
                "reduce_scatter_sec": round(float(self.rs_sec), 6),
                "small_allreduce_sec": round(float(self.small_sec), 6)}


# ═════════════════════════════════════════════════════════════════════════
# 4) 三个动作：归约梯度 / 收回参数 / 释放非本 rank 梯度
# ═════════════════════════════════════════════════════════════════════════
def gather_params(plan, dp, rt=None, chunk=None, stats=None):
    """``all_gather`` 每 rank 更新过的那 1/w 参数，写回所有 rank 的参数表。

    ★ 显存只多占 ``world × chunk``（默认 128 MiB/rank ⇒ w=2 时 256 MiB），
      不是一整份参数表。
    ★ 只在 ``opt.step()`` 之后调用一次；调用后每张卡上的参数**逐位相同**，
      所以下一步的前向/loss 也逐位相同。
    """
    w = int(dp.world)
    if w <= 1 or plan.cap <= 0:
        return
    rtk = rt if rt is not None else _rt()
    x = _xchg_of(dp)
    cc = chunk_from_env(plan.cap, chunk)
    mine = rtk.empty((cc,), dtype=np.float32)
    stage = rtk.empty((w * cc,), dtype=np.float32)
    t0 = _now()
    for lo in range(0, plan.cap, cc):
        hi = min(plan.cap, lo + cc)
        n = hi - lo
        pack_shard(rtk, plan, plan.rank, lo, hi, mine)
        x.all_gather(mine, stage, n, rtk)
        for r in range(w):
            if r == plan.rank:
                continue
            unpack_shard(rtk, plan, r, lo, hi, rtk.sub_view(stage, (n,), r * n))
        if stats is not None:
            stats.gather_send += (w - 1) * n
            stats.gather_recv += (w - 1) * n
    if stats is not None:
        stats.gather_sec += _now() - t0


def reduce_scatter_grads(plan, dp, rt=None, chunk=None, scale=None, stats=None):
    """ZeRO-2 的梯度归约：``reduce_scatter`` 后每 rank 只留自己那 1/w 梯度。

    ``send`` 里按 rank 排成 world 段，第 r 段最终归第 r 个 rank ⇒ 每 rank 只收 ``count``。
    归约结果默认乘 ``1/world``（与数据并行的「求平均」语义一致）。

    ★ 调用前必须 :func:`ensure_grads`（缺梯度补 0），否则
      ``reduce_scatter`` 的 ``send`` 各 rank 字节数不同 ⇒ 共享内存名不同 ⇒ 死等。
    ★ 完成后：本 rank 拥有参数的梯度是完整归约值，其余参数的梯度仍是本地值
      （调用方随后用 :func:`free_unowned_grads` 清掉；清之前它们要参与 gnorm 的
      **张量表枚举**，所以这里不动）。
    """
    w = int(dp.world)
    if w <= 1 or plan.cap <= 0:
        return
    rtk = rt if rt is not None else _rt()
    x = _xchg_of(dp)
    cc = chunk_from_env(plan.cap, chunk)
    sendbuf = rtk.empty((w * cc,), dtype=np.float32)
    out = rtk.empty((cc,), dtype=np.float32)
    sc = float(scale) if scale is not None else 1.0 / float(w)
    t0 = _now()
    for lo in range(0, plan.cap, cc):
        hi = min(plan.cap, lo + cc)
        n = hi - lo
        for r in range(w):
            pack_grads(rtk, plan, r, lo, hi,
                       rtk.sub_view(sendbuf, (n,), r * n))
        x.reduce_scatter(sendbuf, out, n, sc, rtk)
        unpack_grads(rtk, plan, plan.rank, lo, hi, rtk.sub_view(out, (n,), 0))
        if stats is not None:
            stats.rs_send += (w - 1) * n
            stats.rs_recv += (w - 1) * n
    if stats is not None:
        stats.rs_sec += _now() - t0


def free_unowned_grads(plan):
    """释放不属于本 rank 的梯度（ZeRO-2 真正的显存收益在这里兑现）。"""
    n = 0
    for i, p in enumerate(plan.params):
        if plan.i0 <= i < plan.i1:
            continue
        if getattr(p, "grad", None) is not None:
            p.grad = None
            n += 1
    return n


def owned_grad_bufs(plan):
    """本 rank 拥有的、且确实有梯度的参数梯度缓冲（按参数顺序）。"""
    out = []
    for i in range(plan.i0, plan.i1):
        g = getattr(plan.params[i], "grad", None)
        if g is not None:
            out.append(_buf_of(g))
    return out


# ═════════════════════════════════════════════════════════════════════════
# 5) 分布式梯度范数（ZeRO-2 逐位一致的核心）
# ═════════════════════════════════════════════════════════════════════════
def _keep_alive(rt, objs):
    """把异步内核仍在用的临时缓冲挂住（滑动窗口，同 cuda.py 的 `_zero_keep_alive`）。"""
    keep = getattr(rt, "_zero_keep", None)
    if keep is None:
        keep = rt._zero_keep = []
    keep.append(objs)
    if len(keep) > 8:
        del keep[:-8]


def _ssq_part_into(rt, bufs, part, elem_offset=0):
    """把一组张量的**分块平方和**写进 ``part[elem_offset + ...]``（kernel A）。

    优先用运行时的同名入口（新版 cuda.py 有）；没有就用本模块自带的一份
    （同样是 ``sumsq_multi_shard`` 内核 + ``_shard_offsets`` 分块表 ⇒ 逐位同值）。
    """
    bufs = [b for b in bufs if b is not None and int(b.n) > 0]
    if not bufs:
        return 0
    fn = getattr(rt, "sumsq_multi_shard_into", None)
    if fn is not None:
        return fn(bufs, part, int(elem_offset))
    nb, off = rt._shard_offsets(bufs)
    total_blocks = int(off[-1])
    ptrs, sizes = rt._ptrs_and_sizes(bufs)
    offs = rt.h2d(np.ascontiguousarray(off).view(np.float32))
    dst = part if int(elem_offset) == 0 else \
        rt.sub_view(part, (total_blocks,), int(elem_offset))
    rt.launch("sumsq_multi_shard", (total_blocks, 1, 1), (256, 1, 1), [
        rt._ptr(ptrs), rt._ptr(sizes), rt._ptr(offs),
        (len(bufs), ctypes.c_int), rt._ptr(dst)])
    _keep_alive(rt, (ptrs, sizes, offs, dst))
    return total_blocks


def _ssq_part_total(rt, part, off_np, n_tensors):
    """``sumsq_multi_final``：按张量下标顺序把 ``part`` 的块和累加成总平方和。"""
    fn = getattr(rt, "sumsq_blocks_total", None)
    if fn is not None:
        return float(fn(part, off_np, int(n_tensors)))
    offs = rt.h2d(np.ascontiguousarray(
        np.asarray(off_np, dtype=np.int32)).view(np.float32))
    out = rt.empty((), dtype=np.float32)
    rt.launch("sumsq_multi_final", (1, 1, 1), (256, 1, 1), [
        rt._ptr(part), rt._ptr(offs), (int(n_tensors), ctypes.c_int),
        rt._ptr(out)])
    _keep_alive(rt, (part, offs, out))
    return float(rt.read_scalar(out))


def grad_ssq_total(plan, dp, rt=None, stats=None):
    """ZeRO-2 的**分布式**梯度平方和，要求与 ``clip_grad_norm_`` 的结果**逐位相同**。

    ## 为什么不能"各 rank 算自己那块的范数再相加"

    ``sumsq_multi_total_sharded`` 的算法是两步：kernel A 每张量写一组「分块部分和」
    到 ``part``，kernel B 对 ``part`` 做**固定顺序**的树归约 + **按张量下标顺序累加**。
    把张量切成两半分别归约再相加，结合顺序变了 ⇒ 结果差最后几位 ⇒ 裁剪系数差 1 ulp
    ⇒ 参数差 1 ulp ⇒ 20 步后 loss 不再逐位相同。所以必须"复现同一套结合顺序"。

    ## 怎么做到（关键）

    kernel B 对某个张量 t 的贡献是 ``total = __fadd_rn(total, s_t)``，而
    **``x + 0.0 == x`` 在 fp32 下是逐位精确的**。于是：
      1. 每 rank 只把自己那几个张量（在参数表里是**连续**的）的分块部分和写进
         ``part`` 的对应槽位，其余槽位清零；
      2. 对整份 ``part`` 做一次跨 rank 求和（槽位互不重叠，加 0 不改变任何值）；
      3. 用**整份**的 ``off`` 表跑 kernel B。
    这样每个 ``s_t`` 与不分片时逐位相同，累加顺序也相同 ⇒ 总平方和逐位相同。

    ★ 必须在 :func:`free_unowned_grads` **之前**调用（要按参考顺序枚举全部梯度）。
    ★ 调用前必须 :func:`ensure_grads`：数据并行归约后每个参数都有梯度，
      这里的张量表必须与 ``clip_grad_norm_(params)`` 枚举到的完全一致。
    """
    rtk = rt if rt is not None else _rt()
    t0 = _now()
    bufs = []
    pos = []
    for i, p in enumerate(plan.params):
        g = getattr(p, "grad", None)
        if g is None:
            continue
        b = _buf_of(g)
        if int(b.n) <= 0:                       # 与 sumsq_multi_total_sharded 的过滤一致
            continue
        bufs.append(b)
        pos.append(i)
    n_all = len(bufs)
    if n_all == 0:
        return 0.0
    if n_all == 1:
        # 单张量的特例：只有 owner 的那份是**归约后**的值，其余 rank 贡献 0.0
        total = 0.0
        if plan.i0 <= pos[0] < plan.i1:
            total = float(rtk.read_scalar(rtk.sumsq_parallel(bufs[0])))
        acc = _xchg_of(dp).sum_array(np.asarray([total], np.float32))
        if stats is not None:
            stats.ar_send += 1
            stats.ar_recv += 1
            stats.small_sec += _now() - t0
        return float(acc.reshape(-1)[0])
    _nb, off = rtk._shard_offsets(bufs)
    total_blocks = int(off[-1])
    part = rtk.zeros((total_blocks,), dtype=np.float32)
    # 本 rank 拥有的参数在 bufs 里也是连续的一段
    a = None
    b = None
    for k, i in enumerate(pos):
        if plan.i0 <= i < plan.i1:
            if a is None:
                a = k
            b = k + 1
    if a is not None:
        _ssq_part_into(rtk, bufs[a:b], part, int(off[a]))
    if int(dp.world) > 1:
        host = np.ascontiguousarray(np.asarray(rtk.d2h(part), dtype=np.float32))
        host = np.ascontiguousarray(_xchg_of(dp).sum_array(host), dtype=np.float32)
        rtk.copy_h2d_into(part, host)
        if stats is not None:
            stats.ar_send += int(host.size)
            stats.ar_recv += int(host.size)
    total = _ssq_part_total(rtk, part, off, n_all)
    if stats is not None:
        stats.small_sec += _now() - t0
    return float(total)


def grad_norm_scale(plan, dp, max_norm, zero=2, defer=True, rt=None, stats=None):
    """ZeRO 的梯度范数 + 裁剪缩放系数（与 ``clip_grad_norm_`` 逐位同口径）。

    * ZeRO-0/1：梯度是完整的 ⇒ 直接调 ``T.clip_grad_norm_``（与不分片完全同一路径）。
    * ZeRO-2：梯度被切开了 ⇒ 用 :func:`grad_ssq_total` 复现同一个平方和，
      然后**按 ``clip_grad_norm_`` 的同一公式**算 scale：
      ``scale = max_norm / (norm + 1e-6)``（仅当 ``norm > max_norm``）。
      ``defer=True`` 时不做物理缩放（交给优化器内核），返回 ``(norm, scale)``；
      ``defer=False`` 时只对本 rank 拥有的梯度做物理缩放（其余 rank 的对应切片由它们
      自己缩放；不属于本 rank 的梯度随后被释放）⇒ 更新值与不分片逐位相同。
      ZeRO-2 还会在算完范数后**释放非本 rank 的梯度**（这时才真正兑现显存收益）。

    返回 ``(gnorm, gscale)``；不裁剪时 ``gscale == 1.0``。
    """
    if int(zero) != 2:
        import tgai_go.train as T
        r = T.clip_grad_norm_(plan.params, max_norm, defer_scale=bool(defer))
        if isinstance(r, tuple):               # defer 路径返回 (norm, scale)
            return float(r[0]), float(r[1])
        return float(r), 1.0                   # 物理缩放路径只返回 norm
    rtk = rt if rt is not None else _rt()
    total = grad_ssq_total(plan, dp, rt=rtk, stats=stats)
    norm = float(np.sqrt(max(total, 0.0)))
    scale = 1.0
    if max_norm is not None and float(max_norm) > 0 and norm > float(max_norm):
        scale = float(float(max_norm) / (norm + 1e-6))
    if defer:
        free_unowned_grads(plan)
        return norm, scale
    if scale != 1.0:
        bufs = owned_grad_bufs(plan)
        fn = getattr(rtk, "scale_multi_sharded", None)
        if fn is not None:
            fn(bufs, float(scale))
        else:
            for gpu in bufs:
                rtk.scale_inplace(gpu, float(scale))
    free_unowned_grads(plan)
    return norm, 1.0


# ═════════════════════════════════════════════════════════════════════════
# 6) 优化器外壳
# ═════════════════════════════════════════════════════════════════════════
class ShardedOptimizer(object):
    """只持有 1/w 优化器状态的优化器外壳。

    为什么用外壳而不是改每个优化器类：**状态分配在 ``__init__`` 里**。直接拿
    ``params[i0:i1]`` 构造原优化器，状态自然就只有 1/w —— 一行都不用改优化器。
    外壳只补两件原优化器做不到的事：

    1. ``zero_grad()`` 必须清**全部**参数的梯度（否则下一轮梯度会累加到上一步的
       残值上，静默算错）；
    2. ``lr`` 透传（``CosineLR`` 每步写 ``opt.lr``）。
    """

    def __init__(self, inner, params, plan=None):
        self.inner = inner
        self.params = list(params)
        self.plan = plan
        self.sharded = True

    # ---- 必须显式代理的两个 ----
    @property
    def lr(self):
        return self.inner.lr

    @lr.setter
    def lr(self, v):
        self.inner.lr = float(v)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self, *a, **k):
        return self.inner.step(*a, **k)

    # ---- 其余一律透传（t / bytes_per_param / m16 / ...）----
    def __getattr__(self, name):
        if name in ("inner", "params", "plan", "sharded"):
            raise AttributeError(name)
        return getattr(self.inner, name)


def make_optimizer(inner, params, plan):
    return ShardedOptimizer(inner, params, plan)


# ═════════════════════════════════════════════════════════════════════════
# 7) 通信量模型（给报告用；与实测同口径）
# ═════════════════════════════════════════════════════════════════════════
def comm_model(total_elems, world, zero):
    """返回 ``{"grad_bytes": ..., "one_way_bytes": ..., "ratio_vs_dp": ...}``。

    口径：**单向**字节数 / rank。N = 梯度字节数，w = world。

    * ``zero=0``：all_reduce ``2N(w-1)/w``
    * ``zero=1``：all_reduce ``2N(w-1)/w`` + all_gather ``N(w-1)/w``
    * ``zero=2``：reduce_scatter ``N(w-1)/w`` + all_gather ``N(w-1)/w``
    """
    n = float(total_elems) * 4.0
    w = max(1, int(world))
    f = float(w - 1) / float(w)
    base = 2.0 * n * f
    if int(zero) == 0:
        one = base
    elif int(zero) == 1:
        one = base + n * f
    else:
        one = n * f + n * f
    return {"grad_bytes": n,
            "one_way_bytes": one,
            "ratio_vs_dp": (1.0 if base <= 0 else one / base),
            "all_reduce_bytes": (base if int(zero) in (0, 1) else 0.0),
            "reduce_scatter_bytes": (n * f if int(zero) == 2 else 0.0),
            "all_gather_bytes": (n * f if int(zero) in (1, 2) else 0.0)}


def expected_comm_bytes(total_elems, world, zero):
    return int(round(comm_model(total_elems, world, zero)["one_way_bytes"]))


#: 每参数固定字节数（与 `plan.BYTES_PER_PARAM` 同口径：fp32 参数/梯度 + 各优化器状态）。
BYTES_PER_PARAM = {"param": 4.0, "grad": 4.0,
                   "adamw": 8.0, "fp16mom": 4.0, "int8gpu": 3.0625}

_GIB = float(1 << 30)


def estimate_memory(total_elems, world=1, zero=0, optim="int8gpu",
                    shard_grad=False):
    """每卡「参数 + 梯度 + 优化器状态」的**确定值**（GiB），**不含激活**。

    与 :func:`plan.estimate_memory` 完全同一套算术（`tests/test_zero_shard.py`
    会逐档对拍，防止两边漂）。口径：

    | zero | 参数 | 梯度 | 状态 | 何时成立 |
    | --- | --- | --- | --- | --- |
    | 0 | 全量 | 全量 | 全量 | 数据并行 |
    | 1 | 全量 | 全量 | `1/w` | ZeRO-1（**也是 ZeRO-2 的峰值时刻**）|
    | 2 | 全量 | `1/w` | `1/w` | ZeRO-2 **反向结束后**（峰值仍在反向末，和 zero=1 相同）|

    ★ 默认 `shard_grad=False` ⇒ 报的是**峰值**（保守、也是验收口径）：
    ZeRO-2 的峰值与 ZeRO-1 相同，因为峰值在 `backward()` 刚结束、梯度还是全量的那一刻。
    要看 ZeRO-2 反向结束后的稳态，传 `shard_grad=True`（等价于
    ``plan.estimate_memory(..., shard_grad=True)``）。
    """
    w = max(1, int(world))
    z = int(zero)
    if z not in MODES:
        raise ValueError("zero=%r 不认识（可选 0/1/2）" % (zero,) +
                         "。修法：--zero 0/1/2")
    opt = str(optim).lower()
    if opt not in BYTES_PER_PARAM:
        raise ValueError("不认识的优化器 %r" % (optim,) +
                         "。修法：用 %s 之一" % ("/".join(sorted(
                             k for k in BYTES_PER_PARAM
                             if k not in ("param", "grad"))),))
    p = float(total_elems)
    sg = bool(shard_grad)
    so = z >= 1
    gib = (p * BYTES_PER_PARAM["param"]
           + (p / w if sg else p) * BYTES_PER_PARAM["grad"]
           + (p / w if so else p) * BYTES_PER_PARAM[opt]) / _GIB
    return [gib] * w


def bus_comm_bytes(total_elems, world, zero):
    """**收发合计**的总线字节 / rank（用于和 `plan.estimate_comm` 的 `gb` 口径对表）。

    与 :func:`comm_model` 的区别：那个是「单向有效数据量」，这个是 host 路径上
    真正过 PCIe 的 d2h+h2d 字节数（DP 的实测耗时就是用这个口径标定的）。

    * ``zero=0``：all_reduce = d2h `N` + h2d `N` = ``2N``
    * ``zero=1``：``2N``（all_reduce）+ ``N``（all_gather：发自己 1/w + 收别人 1/w）
    * ``zero=2``：``N + N/w``（reduce_scatter：d2h 全量 N、h2d 只收 1/w）
      + ``N``（all_gather）= ``N(2 + 1/w)``
    """
    n = float(total_elems) * 4.0
    w = max(1, int(world))
    if w <= 1:
        return 0
    if int(zero) == 0:
        return 2.0 * n
    if int(zero) == 1:
        return 2.0 * n + n
    return n * (1.0 + 1.0 / w) + n



def broadcast_start(plan, dp, rt=None):
    """训练开始前把权重从 rank0 分发一次（ZeRO 不改变这一步，只是包装）。"""
    if dp is None or dp.world <= 1:
        return
    dp.broadcast(plan.all_params)


# ═════════════════════════════════════════════════════════════════════════
# 8) 训练循环外壳（把「什么时候调哪个」收在一处，减少接线接错的机会）
# ═════════════════════════════════════════════════════════════════════════
class ZeroTrainer(object):
    """一次 ZeRO 训练会话：分片规划 + 优化器外壳 + 三个步内动作。

    ``world`` / ``rank`` 缺省从 ``TGAI_DP_WORLD`` / ``TGAI_DP_RANK`` 读（与
    :class:`dist.DataParallel` 同一套环境变量）。构造时机是**建优化器之前**：
    ``shard_params`` 交给优化器，``wrap`` 包一层外壳。``DataParallel`` 建好后
    ``attach`` 一次即可。

    ``zero=0`` 时本类不产生任何行为（``shard_params is params``、三个动作全是空操作）。
    """

    def __init__(self, params, zero=None, world=None, rank=None, opt_name=None,
                 rt=None):
        # ★ zero=0 时连「列表身份」都要保持原样（纯 no-op，便于逐位对比）
        self.all_params = params if isinstance(params, list) else list(params)
        z = _env_int("TG_ZERO", 0) if zero is None else int(zero)
        if z not in MODES:
            raise ValueError("zero=%r 不认识（可选 0/1/2）" % (zero,) +
                             "。修法：--zero 0/1/2")
        self.zero = z
        if opt_name is not None and z > 0 and str(opt_name) not in SUPPORTED_OPTS:
            raise ValueError(
                "ZeRO 不支持 --opt=%s（支持 %s）"
                % (opt_name, "/".join(SUPPORTED_OPTS)) +
                "。修法：用 --opt int8gpu/adamw/fp16mom，或 --zero 0（不省状态）")
        self.opt_name = opt_name
        w = _env_int("TGAI_DP_WORLD", 1) if world is None else int(world)
        r = _env_int("TGAI_DP_RANK", 0) if rank is None else int(rank)
        if z > 0 and w <= 1:
            raise ValueError(
                "zero=%d 需要多进程（当前 world=%d）" % (z, w) +
                "。修法：--dp 2 且两个进程分别设 TGAI_DP_RANK=0/1；"
                "单卡请用 --zero 0")
        self.plan = ShardPlan(self.all_params, w, r) if z > 0 else None
        self.dp = None
        self.stats = ZeroStats()
        self._rt = rt          # 显式注入的运行时（CPU 单测传假运行时；生产传 None）

    def _rt_now(self):
        return _rt() if self._rt is None else self._rt

    # ---- 建图期 ----
    @property
    def enabled(self):
        return self.zero > 0

    @property
    def shard_params(self):
        """交给优化器构造用的参数列表（zero=0 时就是原列表）。"""
        return self.all_params if self.plan is None else self.plan.owned

    def wrap(self, opt):
        if self.plan is None:
            return opt
        return ShardedOptimizer(opt, self.all_params, self.plan)

    def attach(self, dp):
        """``DataParallel`` 建好、``broadcast`` 之后调用一次（校验 world/rank 一致）。"""
        if self.plan is None:
            return
        if int(dp.world) != self.plan.world or int(dp.rank) != self.plan.rank:
            raise ValueError(
                "ZeRO 的 world/rank（%d/%d，来自 TGAI_DP_*）与 DataParallel"
                "（%d/%d）不一致" % (self.plan.world, self.plan.rank,
                                    int(dp.world), int(dp.rank)) +
                "。修法：--zero 与 --dp 必须用同一组 TGAI_DP_WORLD/TGAI_DP_RANK")
        self.dp = dp

    def describe(self):
        if self.plan is None:
            return "zero=0（不分片）"
        c = comm_model(self.plan.total_elems, self.plan.world, self.zero)
        return ("zero=%d %s | comm/step one-way=%.3f GiB (DP 的 %.2fx) | "
                "状态≈%.2f B/param（int8gpu 档，w 分片后） | %s"
                % (self.zero,
                   "ZeRO-1（状态分片）" if self.zero == 1 else "ZeRO-2（状态+梯度分片）",
                   c["one_way_bytes"] / float(1 << 30), c["ratio_vs_dp"],
                   3.0625 / self.plan.world,
                   self.plan.describe()))

    # ---- 步内动作 ----
    def reduce_grads(self):
        """反向之后、裁剪之前调用。ZeRO-1 = 全量 all_reduce；ZeRO-2 = reduce_scatter。"""
        if self.plan is None:
            return None
        if self.dp is None:
            raise RuntimeError("ZeroTrainer 还没 attach(DataParallel)" +
                               "。修法：建好 DataParallel 后立即调用 attach")
        t0 = _now()
        rtk = self._rt_now()
        if self.zero == 1:
            # 与 clip_grad_norm_ 枚举同一份（去重后的）参数表；v3_pretrain 里两者相同
            self.dp.all_reduce_grads(self.plan.params)
        else:
            ensure_grads(self.plan, rtk)
            reduce_scatter_grads(self.plan, self.dp, rt=rtk, stats=self.stats)
        return _now() - t0

    def grad_norm_scale(self, max_norm, defer=True):
        if self.plan is None:
            return 0.0, 1.0
        if self.dp is None:
            raise RuntimeError("ZeroTrainer 还没 attach(DataParallel)" +
                               "。修法：建好 DataParallel 后立即调用 attach")
        return grad_norm_scale(self.plan, self.dp, max_norm, zero=self.zero,
                               defer=defer, rt=self._rt_now(), stats=self.stats)

    def after_step(self):
        """优化器更新之后调用：all_gather 参数（每卡恢复成完整且逐位相同的参数）。"""
        if self.plan is None:
            return None
        if self.dp is None:
            raise RuntimeError("ZeroTrainer 还没 attach(DataParallel)" +
                               "。修法：建好 DataParallel 后立即调用 attach")
        t0 = _now()
        gather_params(self.plan, self.dp, rt=self._rt_now(), stats=self.stats)
        return _now() - t0

    def close(self):
        x = getattr(self.dp, "_zero_xchg", None) if self.dp is not None else None
        if x is not None:
            try:
                x.close()
            except Exception:                        # noqa: BLE001
                pass
