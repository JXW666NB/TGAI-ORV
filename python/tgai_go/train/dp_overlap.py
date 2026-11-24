# -*- coding: utf-8 -*-
"""反向重叠的数据并行梯度归约（gradient bucketing + all-reduce overlap）。

## 为什么要它（实测，不是推断）

2×RTX 5090 / 拓扑 SYS / `p2p=False` / 848M 参数 / 3.16 GiB 梯度，同一份集合通信：

| 实现 | seq512 2-rank 吞吐比 | 归约暴露 |
| --- | --- | --- |
| ORV `dist.all_reduce_grads`（不重叠） | 0.87×（host）/ 0.84×（nccl） | **100%**（host 148 ms / NCCL 141 ms） |
| PyTorch DDP（反向中分桶启动） | 1.85–1.93× | 7–17 ms |

归约本身是物理成本（无 P2P 时 3.16 GiB 走 PCIe 就是 ~140 ms），差别只在**什么时候发**。
本模块把桶的 D2D 打包放进反向里做，桶一齐就立刻在**第二条流**上启动集合通信。

## 实测结果（2026-11-21，2×5090）与结论

完整留档见 `docs/DP_OVERLAP_2026-11-21.md`。一句话：**正确性全绿，性能只到 seq512
1.10×（基线 0.84–0.87×）/ seq2048 1.25×（基线 1.20–1.21×），没到验收线（1.35×/1.5×）。**

根因同样是实测的：NCCL 8 通道把 3.16 GiB fp32 all-reduce 做到 86 ms（默认 134 ms），
但**NCCL 内核与 ORV 计算内核在这台 PCIe-only 的 5090 上基本不共驻** —— 一开重叠，
反向从 70 ms 被拖到 183 ms（≈ 70 + 归约 111 ms），归约并没有真的被"藏"起来。
用 host 拷贝引擎的路径反向不膨胀（仍是 70 ms），但那条路径的搬运本身要 150–190 ms，
也藏不住。⇒ 在「3.16 GiB fp32 + 无 P2P」下，seq512 的 1.35× 是物理上达不到的。

## 三个必须同时成立的前提（每条都有对应的硬校验）

1. **"叶子只会被写一次 .grad"** —— 见下面「tied」一节。
2. **集合通信的调用序列在所有 rank 上逐字相同** —— 桶布局由 `params` 静态决定
   （各 rank 相同），提交严格 FIFO。**绝不**用"谁先就绪"动态组桶：MoE 路由是
   数据相关的，两个 rank 的就绪顺序可以不同，动态组桶会让 collective 序列错位
   ⇒ 死锁或把 A 的梯度加到 B 上。
3. **反向过程中不能有其它集合同步**（`broadcast` / 文件 barrier / `all_reduce_grads`）。

## tied / 共享参数：为什么这里数的是「写 .grad 的次数」而不是「反向边」

`tie_embedding` 时 `P["head"] is P["emb"]`：同一个张量既是 embedding 权重又是
lm_head 权重。前向里它被两条路径用到，所以反向里 `_acc()` 会**被调两次**
（embedding 反向一次、matmul 反向一次）。

但 ORV 的 `_acc` 只把两份贡献累加进 `grads[id(leaf)]` 这个**中转缓冲**，
真正写 `leaf.grad` 的只有拓扑序走到该叶子的那**一次**（`__init__.py` 的
`node.grad = ...` / `node.grad = node.grad + g`，CUDA 分支同构）。

⇒ 若照 PyTorch DDP 那样「数反向边」，tied 叶子的 pending 会是 2，而钩子只会到
1 次：桶永远等不到齐 ⇒ 要么卡死、要么漏归约。**这就是静默错的那条路。**
本模块数的 pending 是「本次反传会不会写这个叶子的 .grad」：在可达叶集合里 ⇒ 1，
不在 ⇒ 0（MoE 未路由到的专家、冻结参数）。0 的成员按 0 参与集合通信
（与不重叠路径 `dist.all_reduce_grads` 的语义逐位一致）。

`tests/test_dp_overlap.py` 用 CPU 对 tied 配置做了**逐位**对比，不靠「看起来对」。

## 为什么钩在 `.grad` 写入点，而不是 `_acc`

* `_acc` 是热路径（一次反向 ~1000 次调用），且它在**中转缓冲**上就地累加
  （`cuda.py:add_`），提前拷走就是半成品；等它计数归零再拷，与「等 .grad 写入」
  是同一个时刻，只是多付 1000 次分派。
* `.grad` 写入点是「这个叶子的梯度已经是最终值」的唯一权威时刻
  （拓扑序保证所有贡献都已 `_acc` 完）。

## 安全阀（任何一条不满足就**退回不重叠路径**，绝不静默错）

| 情形 | 为什么危险 | 处理 |
| --- | --- | --- |
| 图里有梯度检查点（`bw._is_checkpoint`） | 段内参数由**嵌套反传**各自写 `.grad`，「只写一次」失效；且部分桶可能已发出 | `reason=grad_checkpoint`，整步走老路径 |
| 进反向时已有 `.grad`（梯度累积 / 上步没清） | 提前归约的是「累积中的半成品」 | `reason=grad_accumulation` |
| CPU 后端 | 没有设备侧归约可重叠 | `reason=cpu` |
| NCCL/host 传输初始化失败 | 缺符号、显存不足、shm 注册失败 | `reason=transport:...` |
| `TG_DP_OVERLAP=0` | —— | `reason=disabled` |
| `world<=1` | —— | 返回 `None`（**零行为变化**） |

## 失败必须响

`finish()` 逐参数核对「该到的是否到了」：

* 该到而没到、**但 `.grad` 已经不是 None** ⇒ 说明钩子漏了（写了却没通知）
  ⇒ **抛 RuntimeError**（这正是「静默漏归约」的样子，绝不能放过）。
* 该到而没到、`.grad is None` ⇒ 这个叶子确实没有梯度（与不重叠路径一样按 0 补齐）
  ⇒ 计入 `stats()["zero_filled"]`，不抛。

## 显存

桶缓冲是**环形复用**的 `slots` 块（默认 3 × 256 MB = 768 MiB），不是「每个桶一块」
（那要 3.16 GiB）。slot 被复用前，主流程先等上一轮用它的桶完成
（`cuStreamWaitEvent`，只挡流不挡主机线程）。

## 低精度通信（`TG_DP_GRAD_DTYPE=bf16|fp16`，默认 fp32）

三条传输都支持把桶按低精度通信（字节数减半，见 `dist.py` 的同一节）：

* `NumpyTransport`：stage 时量化（`bf16/fp16` 收窄再升宽），缓冲里是「对端会收到的
  数值」，用于 CPU 上逐位对拍；
* `NcclBucketTransport`：桶缓冲本身是 2 字节，`ncclAllReduce(bf16/fp16)`；
* `HostBucketTransport`：共享内存里传 2 字节，对端 H2D 后 dequant 成 fp32 再累加。

量化语义：**先在 stage 时乘 `1/world` 再收窄**（`bf16` 下乘 2 的幂是精确的，
所以 world=2 时与不重叠路径的「先求和再乘」逐位一致），归约完后直接 dequant
覆盖写回 `p.grad`。fp32 路径**一个字节都没改**。
"""

def _dist_lowp():
    """取 `dist.py` 里的低精度 cast/清零内核封装（延迟 import：避免循环依赖）。"""
    from . import dist as _d
    return _d


def _nccl_dtype_for(dtype):
    """低精度 dtype → ``ncclDataType_t``。fp32 返回 None（调用方走无类型的老路径）。"""
    if dtype == "fp32":
        return None
    from . import dist_nccl as _N
    if dtype == "bf16":
        return int(_N.NCCL_BFLOAT16)
    if dtype == "fp16":
        return int(_N.NCCL_FLOAT16)
    raise ValueError(
        "未知的梯度通信 dtype=%r" % (dtype,) +
        "。修法：只支持 fp32/bf16/fp16；先用 resolve_grad_dtype() 归一化")

import os
import queue
import threading
import time

import numpy as np

__all__ = [
    "OverlapUnavailable",
    "BucketPlan",
    "build_plan",
    "plan_order_adapter",
    "OverlapSession",
    "TransportBase",
    "NumpyTransport",
    "NcclBucketTransport",
    "HostBucketTransport",
    "overlap_enabled",
    "resolve_bucket_bytes",
    "resolve_slots",
]

#: 桶大小的默认值（MB）。`TG_DP_BUCKET_MB` 可调；0 = 一个桶装下全部梯度
#: （用来做「桶大小=全部」的 A/B 对照）。
DEFAULT_BUCKET_MB = 256

#: 环形缓冲的槽数。3 槽 × 256 MB = 768 MiB，在「峰值显存涨幅 ≤1 GiB」的红线内。
DEFAULT_SLOTS = 3


class OverlapUnavailable(RuntimeError):
    """这条传输路径当前不可用（缺符号/显存不足/shm 注册失败……）。

    捕获方**必须**退回不重叠路径，并把 `reason` 记进 stats —— 不允许
    「静默降级成看起来开了重叠」。
    """


def _env_int(name, default, lo=None, hi=None):
    raw = os.environ.get(name, "")
    if str(raw).strip() == "":
        v = int(default)
    else:
        try:
            v = int(str(raw).strip())
        except (TypeError, ValueError):
            raise ValueError("%s=%r 不是整数" % (name, raw) +
                             "。修法：写一个整数；不想调就删掉这个环境变量")
    if lo is not None and v < lo:
        raise ValueError("%s=%d 太小（下限 %d）" % (name, v, lo) +
                         "。修法：按下限写，或删掉这个环境变量用默认值")
    if hi is not None and v > hi:
        raise ValueError("%s=%d 太大（上限 %d）" % (name, v, hi) +
                         "。修法：按上限写，或删掉这个环境变量用默认值")
    return v


def overlap_enabled():
    """`TG_DP_OVERLAP`：默认开；`0/false/off/no` 关。非法值**当场报错**。

    为什么不静默：写错一个字母就退回「100% 暴露」的老路径，日志里看不出任何
    异常 —— 这种「悄悄降级」是性能实验最难查的一类错（与 `dp_backend_requested`
    同一条纪律）。
    """
    raw = os.environ.get("TG_DP_OVERLAP", "")
    s = str(raw).strip().lower()
    if s in ("", "1", "true", "on", "yes"):
        return True
    if s in ("0", "false", "off", "no"):
        return False
    raise ValueError(
        "TG_DP_OVERLAP=%r 不认识" % (raw,) +
        "。修法：设 1 开、0 关；不想用就删掉这个环境变量")


def resolve_bucket_bytes():
    """桶上限（字节）。`TG_DP_BUCKET_MB` 默认 256 MB；0 ⇒ 不设上限。"""
    mb = _env_int("TG_DP_BUCKET_MB", DEFAULT_BUCKET_MB, lo=0)
    return 0 if mb == 0 else int(mb) << 20


def resolve_slots():
    """环形缓冲槽数（`TG_DP_OVERLAP_SLOTS`，默认 3）。"""
    return _env_int("TG_DP_OVERLAP_SLOTS", DEFAULT_SLOTS, lo=1, hi=16)


def resolve_grad_dtype(dtype=None):
    """归一化梯度通信精度。``None`` ⇒ 读 `TG_DP_GRAD_DTYPE`（默认 fp32）。

    非法值**当场报错**（与 dist 里同一条纪律：写错字母悄悄退回 2 倍字节，
    日志里看不出来）。
    """
    if dtype is None:
        return _dist_lowp().grad_dtype_requested()
    v = str(dtype).strip().lower()
    if v in ("fp32", "float32"):
        return "fp32"
    if v in ("bf16", "bfloat16"):
        return "bf16"
    if v in ("fp16", "half", "float16"):
        return "fp16"
    raise ValueError(
        "grad_dtype=%r 不认识（可选 fp32/bf16/fp16）" % (dtype,) +
        "。修法：传 fp32、bf16 或 fp16；想保持历史行为就传 fp32 / 不传")


# ══════════════════════════════════════════════════════════════════════════
# 静态桶布局
# ══════════════════════════════════════════════════════════════════════════
class BucketPlan(object):
    """一次 step 内固定的桶布局。**必须各 rank 完全一致**（只由 `params` 决定）。"""

    __slots__ = ("groups", "param_off", "param_bucket", "param_elems",
                 "bucket_elems", "bucket_off", "cap_elems")

    def __init__(self, groups, param_off, param_bucket, param_elems,
                 bucket_elems, bucket_off, cap_elems):
        self.groups = groups              # [[param_idx, ...], ...]
        self.param_off = param_off        # 每个参数在它那个桶内的元素偏移
        self.param_bucket = param_bucket  # 每个参数属于哪个桶
        self.param_elems = param_elems
        self.bucket_elems = bucket_elems
        self.bucket_off = bucket_off      # 每个桶在「整条梯度」里的元素偏移
        self.cap_elems = cap_elems

    @property
    def nbuckets(self):
        return len(self.groups)

    def nbytes(self, b):
        return int(self.bucket_elems[b]) * 4

    @property
    def max_bucket_bytes(self):
        return max(self.bucket_elems) * 4 if self.bucket_elems else 0

    @property
    def total_bytes(self):
        return int(sum(self.bucket_elems)) * 4

    def summary(self):
        return {
            "buckets": self.nbuckets,
            "bytes": self.total_bytes,
            "max_bucket_mb": round(self.max_bucket_bytes / float(1 << 20), 2),
            "cap_mb": round(self.cap_elems * 4 / float(1 << 20), 2),
        }

    def describe(self, n_show=6):
        """给日志用的一行摘要：每桶多少参数 / 多大。"""
        parts = []
        for b, g in enumerate(self.groups):
            parts.append("%d:%d/%.0fMB" % (b, len(g), self.bucket_elems[b] * 4 / 1e6))
            if len(parts) >= n_show:
                parts.append("...")
                break
        return " ".join(parts)


def build_plan(elems, bucket_bytes=None, order=None):
    """按**就绪顺序**把参数切成桶。

    `elems`：每个参数的元素个数（`params` 顺序，各 rank 必须相同）。
    `order`：就绪顺序（参数下标的一个排列）。`None` ⇒ 参数列表的**倒序**
             —— 反向是前向的逆序，这是 PyTorch DDP 的做法，也是不依赖模型结构的
             通用近似。调用方知道得更清楚时可以显式给（例如 `v3_pretrain` 把
             `head` 放最前、`emb` 放最后：head 的梯度反向一开始就有，emb 的要等到最后）。
    `bucket_bytes`：桶上限；单个参数大于上限时自成一块（**不切参数**）。

    ★ 这个函数**只**看 `elems`/`order`，不看任何运行时状态 —— 各 rank 的结果
      逐字节相同是集合通信正确性的前提。
    """
    elems = [int(e) for e in elems]
    n = len(elems)
    if n == 0:
        raise ValueError("build_plan: 参数列表是空的" +
                         "。修法：确认 params 非空（反向已经跑过、参数没被过滤掉）")
    if any(e < 0 for e in elems):
        raise ValueError("build_plan: 参数元素个数不能为负" +
                         "。修法：检查是否有 0 维/异常张量混进了 params")
    cap = resolve_bucket_bytes() if bucket_bytes is None else int(bucket_bytes)
    cap_elems = 0 if cap <= 0 else max(1, cap // 4)
    if order is None:
        order = list(range(n - 1, -1, -1))
    else:
        order = [int(i) for i in order]
        if len(order) != n or sorted(order) != list(range(n)):
            raise ValueError(
                "build_plan: order 必须是 0..%d 的一个排列（收到 %d 个）"
                % (n - 1, len(order)) +
                "。修法：调用方给的顺序漏了/重了参数；不给 order 就用默认倒序")
    groups, bucket_elems, bucket_off = [], [], []
    param_bucket = [0] * n
    param_off = [0] * n
    cur, cur_elems, flat_off = [], 0, 0
    for i in order:
        e = elems[i]
        if cur and cap_elems and cur_elems + e > cap_elems:
            groups.append(cur)
            bucket_elems.append(cur_elems)
            bucket_off.append(flat_off)
            flat_off += cur_elems
            cur, cur_elems = [], 0
        param_off[i] = cur_elems
        cur.append(i)
        cur_elems += e
    if cur:
        groups.append(cur)
        bucket_elems.append(cur_elems)
        bucket_off.append(flat_off)
    for b, g in enumerate(groups):
        for i in g:
            param_bucket[i] = b
    return BucketPlan(groups, param_off, param_bucket, elems,
                      bucket_elems, bucket_off, cap_elems)


def plan_order_adapter(params, first=(), last=(), reverse=True):
    """给「已知结构」的调用方用的顺序适配器：`first` 最前、`last` 最后、其余倒序。

    `first` / `last` 是**张量对象**（不是下标）——调用方直接给 `[P["head"]]` /
    `[P["emb"]]`。找不到的张量会被忽略（tied 时 `head is emb`，同一个张量只出现一次）。
    """
    idx = {}
    for i, p in enumerate(params):
        idx.setdefault(id(p), i)                   # 重复张量只认第一个（与去重口径一致）
    head, tail, used = [], [], set()
    for t in first:
        i = idx.get(id(t))
        if i is not None and i not in used:
            head.append(i)
            used.add(i)
    for t in last:
        i = idx.get(id(t))
        if i is not None and i not in used:
            tail.append(i)
            used.add(i)
    mid = [i for i in range(len(params)) if i not in used]
    if reverse:
        mid = list(reversed(mid))
    return head + mid + tail


# ══════════════════════════════════════════════════════════════════════════
# 会话
# ══════════════════════════════════════════════════════════════════════════
class _BucketInfo(object):
    """一个已提交桶的账面：区域、缺失成员、槽号。"""

    __slots__ = ("index", "slot", "elems", "regions", "zero_regions", "missing")

    def __init__(self, index, slot, elems):
        self.index = int(index)
        self.slot = int(slot)
        self.elems = int(elems)
        self.regions = []          # [(off_bytes, param_idx, nbytes)] 有梯度
        self.zero_regions = []     # [(off_bytes, param_idx, nbytes)] 按 0 参与
        self.missing = []          # [(off_bytes, param_idx, nbytes)] 最终要 .grad=0

    def __repr__(self):
        return "BucketInfo(#%d slot=%d elems=%d regions=%d missing=%d)" % (
            self.index, self.slot, self.elems, len(self.regions), len(self.missing))


class OverlapSession(object):
    """一次 step 的重叠归约会话。

    生命周期严格是 `begin_overlap()` → 一次 `backward()` → `finish()`。
    中间不能有第二次 backward（会抛，见 `attach_graph`）。
    """

    def __init__(self, params, plan, transport, world=1, scale=None,
                 zero_grad=None, deduped=0, tag=""):
        self.params = list(params)
        self.plan = plan
        self.transport = transport
        self.world = int(world)
        self.scale = (1.0 / float(self.world)) if scale is None else float(scale)
        self.tag = tag
        self._zero_grad = zero_grad
        self.deduped = int(deduped)
        n = len(self.params)
        self.slots = max(1, int(getattr(transport, "slots", 1)))
        self._index = {}
        for i, p in enumerate(self.params):
            if id(p) in self._index:
                raise ValueError(
                    "OverlapSession 的 params 里有重复张量（下标 %d 与 %d）"
                    % (self._index[id(p)], i) +
                    "。修法：调用方先按 id() 去重（tied 权重会被 parameters() 列两次）")
            self._index[id(p)] = i
        self._pending = [0] * n
        self._done = [False] * n
        #: 建图时就不在本次反向图里的参数（pending==0）——与不重叠路径一样**补 0 梯度**，
        #: 否则优化器会跳过它（与历史行为不一致），而且各 rank 的 .grad 形状集合也会不同。
        self._zero_members = []
        self._bucket_left = [len(g) for g in plan.groups]
        self._slot_ready = [False] * plan.nbuckets
        self._submitted = 0
        self._infos = []
        self._attached = False
        self._finished = False
        self._active = False
        self.reason = None
        self._t_begin = None
        self._exposed_sec = None
        self._comm_sec = None
        self._zero_filled = 0
        self._late_buckets = 0
        #: `finish()` 是否把**所有**参数的就绪记账都结清了（`True` 才允许继续训练）
        self._pending_zeroed = None
        self._open_sec = None
        self._finish_sec = None
        self._stats_cache = None

    # ---------- 生命周期 ----------

    def fallback(self, reason):
        """本次整步退回不重叠路径（安全阀）。"""
        self.reason = reason
        self._active = False
        return None

    def attach_graph(self, topo, cuda=True):
        """建图后调用一次：核对前提、算 pending、开传输。返回 self 或 None。

        `topo` 是 `Tensor.backward` 里那份拓扑序（后序）列表。
        """
        if self._finished:
            raise RuntimeError("OverlapSession 已经 finish()，不能再用" +
                               "。修法：每步都要 begin_overlap() 建新会话")
        if self._attached:
            raise RuntimeError(
                "同一次 begin_overlap 里 backward() 被调用了两次"
                "。修法：梯度累积/多 loss 反传请设 TG_DP_OVERLAP=0"
                "（重叠只支持「一步一次反向」）")
        if self.reason is not None:
            return None
        if not cuda and not getattr(self.transport, "supports_cpu", False):
            # 生产环境的 CPU 数据并行没有"设备侧归约"可重叠；只有测试用的
            # NumpyTransport 声明 supports_cpu=True，用来验钩子/桶/写回的逐位正确性。
            return self.fallback("cpu")
        # ① 梯度检查点：段内参数由嵌套反传各自写 .grad，「叶子只写一次」不成立
        for node in topo:
            bw = getattr(node, "_backward", None)
            if bw is not None and getattr(bw, "_is_checkpoint", False):
                return self.fallback("grad_checkpoint")
        # ② 已有 .grad ⇒ 梯度累积（或上一步没 zero_grad）：提前归约会归约半成品
        for p in self.params:
            if getattr(p, "grad", None) is not None:
                return self.fallback("grad_accumulation")
        # ③ 叶集合 ⇒ pending（1 = 会写 .grad；0 = 本步不参与，按 0 归约）
        leaf_ids = set()
        for node in topo:
            if (getattr(node, "_backward", None) is None
                    and getattr(node, "requires_grad", False)):
                leaf_ids.add(id(node))
        for i, p in enumerate(self.params):
            if id(p) in leaf_ids:
                self._pending[i] = 1
            else:
                # 本步没有反向路径写这个叶子的 .grad（MoE 未路由到的专家、冻结参数、
                # 或调用方多传的参数）。按**0 参与集合通信 + 最后补 0 梯度**处理，
                # 与不重叠路径 `dist.all_reduce_grads` 的 missing 语义逐位一致。
                self._bucket_left[self.plan.param_bucket[i]] -= 1
                self._zero_members.append(i)
        _t_open = time.perf_counter()
        try:
            self.transport.open(self)
        except OverlapUnavailable as exc:
            return self.fallback("transport:%s" % (exc,))
        except Exception as exc:                       # noqa: BLE001
            return self.fallback("transport_error:%r" % (exc,))
        self._open_sec = time.perf_counter() - _t_open
        self._attached = True
        self._active = True
        self._t_begin = time.perf_counter()
        # 头部桶可能整个都是「本步不参与」的参数（极端情况）⇒ 现在就发
        self._try_submit()
        return self

    def leaf_ready(self, node):
        """叶子 `.grad` 写完后的钩子（**只读 .grad**，不许改）。"""
        if not self._active:
            return
        i = self._index.get(id(node))
        if i is None:
            return
        if self._done[i]:
            raise RuntimeError(
                "参数 #%d（shape=%s）的梯度在同一次反向里被写了两次"
                % (i, tuple(getattr(self.params[i], "shape", ()))) +
                "。修法：这是重叠归约的 usage 计数错误（tied 参数最容易踩），"
                "先设 TG_DP_OVERLAP=0 确认不重叠版本正确，再反馈")
        if self._pending[i] != 1:
            raise RuntimeError(
                "参数 #%d（shape=%s）的梯度就绪了，但它不在建图时的可达叶集合里"
                % (i, tuple(getattr(self.params[i], "shape", ()))) +
                "。修法：说明图结构与反向实际执行不一致（检查点/条件分支？），"
                "先设 TG_DP_OVERLAP=0")
        b = self.plan.param_bucket[i]
        if not self._slot_ready[b]:
            self.transport.await_slot(b % self.slots, b - self.slots)
            self._slot_ready[b] = True
        off = int(self.plan.param_off[i]) * 4
        nb = int(self.plan.param_elems[i]) * 4
        self.transport.stage(b % self.slots, off, i, nb)
        self._done[i] = True
        self._bucket_left[b] -= 1
        self._try_submit()

    def finish(self):
        """反向结束：校验、补齐、等通信、写回、出 stats。"""
        if self._stats_cache is not None:
            return self._stats_cache
        if not self._active:
            self._stats_cache = {"active": False,
                                 "reason": self.reason or "not_attached",
                                 "pending_zeroed": None,
                                 "mode": self.transport.kind}
            return self._stats_cache
        n = len(self.params)
        fired = [i for i in range(n) if self._pending[i] and not self._done[i]]
        hard = [i for i in fired if getattr(self.params[i], "grad", None) is not None]
        if hard:
            raise RuntimeError(
                "重叠归约漏掉了 %d 个参数的梯度就绪通知（下标 %s，形状 %s）"
                % (len(hard), hard[:8],
                   [tuple(getattr(self.params[i], "shape", ())) for i in hard[:8]]) +
                "。修法：这些参数**已经有 .grad** 却没触发钩子 ⇒ 归约会漏掉它们，"
                "绝不能继续。请设 TG_DP_OVERLAP=0 并反馈（附完整栈）")
        # 该到没到、且 .grad 仍为 None ⇒ 这个叶子本步确实没有梯度（与不重叠路径
        # 一样按 0 参与）。把 pending 改判成 0，让桶在提交时把它们当「缺失」处理。
        soft = [i for i in fired if getattr(self.params[i], "grad", None) is None]
        for i in soft:
            self._pending[i] = 0
            self._bucket_left[self.plan.param_bucket[i]] -= 1
        # 两处来源合并：① 建图时就不在反向图里的（pending 0）；② 在图里但本步没有
        # 梯度到达的（soft）。两者与不重叠路径一样，最终都要拿到**全 0** 的 .grad。
        zero_all = list(self._zero_members) + soft
        self._zero_filled = len(zero_all)
        for b, left in enumerate(self._bucket_left):
            if left != 0:
                raise RuntimeError(
                    "重叠归约的内部记账不一致：桶 %d 还有 %d 个成员没结论" % (b, left) +
                    "。修法：这是 bug，设 TG_DP_OVERLAP=0 并反馈")
        # 被慢成员卡住 FIFO、没能在反向中发出的桶，在这里补发
        while self._submitted < self.plan.nbuckets:
            self._late_buckets += 1
            self._submit(self._submitted)
            self._submitted += 1
        win = self.transport.begin_window()
        for info in self._infos:
            self.transport.wait(info)
        for info in self._infos:
            self.transport.writeback(info)
        if zero_all:
            self._write_zero_grads(zero_all)
        self._exposed_sec = self.transport.end_window(win)
        self._comm_sec = self.transport.comm_seconds()
        self.transport.finish()
        self._finished = True
        self._pending_zeroed = True
        self._stats_cache = self._build_stats()
        return self._stats_cache

    def abort(self):
        """异常路径：拆掉钩子、关传输（调用方会重新跑老路径或退出）。"""
        self._active = False
        try:
            self.transport.close()
        except Exception:                              # noqa: BLE001 - 清理不抛
            pass
        self._finished = True

    # ---------- 内部 ----------

    def _try_submit(self):
        while (self._submitted < self.plan.nbuckets
               and self._bucket_left[self._submitted] == 0):
            self._submit(self._submitted)
            self._submitted += 1

    def _submit(self, b):
        slot = b % self.slots
        if not self._slot_ready[b]:
            self.transport.await_slot(slot, b - self.slots)
            self._slot_ready[b] = True
        info = _BucketInfo(b, slot, self.plan.bucket_elems[b])
        for i in self.plan.groups[b]:
            off = int(self.plan.param_off[i]) * 4
            nb = int(self.plan.param_elems[i]) * 4
            if self._done[i]:
                info.regions.append((off, i, nb))
            elif self._pending[i] == 0:
                info.zero_regions.append((off, i, nb))
                info.missing.append((off, i, nb))
            else:
                raise RuntimeError(
                    "桶 %d 的成员 #%d 还没就绪却被提交" % (b, i) +
                    "。修法：这是重叠归约的内部 bug，设 TG_DP_OVERLAP=0 并反馈")
        self._infos.append(info)
        self.transport.submit(info)

    def _write_zero_grads(self, idxs):
        if self._zero_grad is None:
            raise RuntimeError(
                "有 %d 个参数本步没有梯度，但会话没拿到「补 0 梯度」的回调"
                % (len(idxs),) +
                "。修法：用 dist.begin_overlap() 建会话（它会注入 _set_grad）")
        for i in idxs:
            self._zero_grad(self.params[i])

    def _build_stats(self):
        out = {
            "active": bool(self._active),
            "mode": self.transport.kind,
            "params": len(self.params),
            "deduped": int(self.deduped),
            "zero_filled": int(self._zero_filled),
            "late_buckets": int(self._late_buckets),
            # ★ 任务要求的判据：pending 是否全部归零。`True` = 每个参数都拿到了结论
            #   （有梯度并已归约 / 确实没有梯度并按 0 补齐）；`None` = 会话没生效。
            "pending_zeroed": self._pending_zeroed,
        }
        out.update(self.plan.summary())
        out["slots"] = int(self.slots)
        if self._exposed_sec is not None:
            out["exposed_sec"] = round(float(self._exposed_sec), 6)
            out["exposed_ms"] = round(1000.0 * float(self._exposed_sec), 2)
        if self._open_sec is not None:
            out["open_ms"] = round(1000.0 * float(self._open_sec), 2)
        if self._comm_sec is not None:
            out["comm_sec"] = round(float(self._comm_sec), 6)
            out["comm_ms"] = round(1000.0 * float(self._comm_sec), 2)
            if self._exposed_sec is not None:
                out["overlapped_ms"] = round(max(
                    0.0, 1000.0 * (float(self._comm_sec) - float(self._exposed_sec))), 2)
        if self.reason:
            out["reason"] = self.reason
        extra = self.transport.stats()
        if extra:
            out.update(extra)
        return out

    def stats(self):
        """已 finish 就直接返回缓存（close 之后也能读）；否则给一份当前快照。"""
        if self._stats_cache is not None:
            return self._stats_cache
        return self._build_stats()


# ══════════════════════════════════════════════════════════════════════════
# 传输（集合通信）抽象
# ══════════════════════════════════════════════════════════════════════════
class TransportBase(object):
    """会话只依赖这几个方法。实现要么是设备侧集合通信，要么是（测试用的）直通。"""

    kind = "abstract"
    slots = 1
    available = True
    #: 这条传输能不能在 CPU 张量上工作。生产传输全是 False（CPU 数据并行没有
    #: 设备侧归约可重叠）；只有测试用的 NumpyTransport 是 True。
    supports_cpu = False

    def open(self, sess):
        raise NotImplementedError

    def stage(self, slot, off, i, nbytes):
        """主流程：把 `sess.params[i].grad` 的 `nbytes` 拷进 `slot` 的 `off` 处。"""
        raise NotImplementedError

    def await_slot(self, slot, prev_bucket):
        """主流程：写 `slot` 之前，确保上一轮用它的桶（`prev_bucket`，可为负）已完成。"""
        pass

    def submit(self, info):
        """异步启动这个桶的集合通信 + 写回（**不许阻塞主机线程**）。"""
        raise NotImplementedError

    def begin_window(self):
        """在主流上记「反向结束」时刻。"""
        return None

    def wait(self, info):
        """主流等这个桶完成（**只挡流、不挡主机线程**）。"""
        raise NotImplementedError

    def writeback(self, info):
        """桶内结果已写回各 grad；这里只补「缺失梯度」的 0 张量。"""
        pass

    def end_window(self, token):
        """所有 wait 之后：返回「反向结束 → 归约完成」的秒数（暴露时间）。"""
        return None

    def comm_seconds(self):
        return None

    def finish(self):
        pass

    def close(self):
        pass

    def stats(self):
        return {}


# ══════════════════════════════════════════════════════════════════════════
# ① CPU / 单进程：直通（测试 + 「桶拼拆逐位正确」的验证）
# ══════════════════════════════════════════════════════════════════════════
class NumpyTransport(TransportBase):
    """numpy 直通：`reduce_fn(bucket, info, sess)` 决定归约语义。

    默认不做集合通信（world=1 语义：桶里的值就是最终值）。测试里可以塞一个
    `reduce_fn` 模拟「各 rank 求和 / 取平均」，验证算术与写回是否逐位正确。

    低精度（`grad_dtype=bf16/fp16`）时：stage 先把梯度**量化**（收窄再升宽），
    缓冲里存的是「对端实际会收到的数值」（fp32 表示、但只有低精度的信息量），
    `reduce_fn` 与写回都作用在量化后的值上 —— 因此可以在 CPU 上**逐位**断言
    「cast 路径 + 均值语义」是否正确。`comm_bytes` 记下真实通信载荷字节。
    """

    kind = "numpy"
    slots = 1
    supports_cpu = True

    def __init__(self, reduce_fn=None, scale=None, grad_dtype="fp32"):
        self.reduce_fn = reduce_fn
        self.scale = scale
        self.grad_dtype = resolve_grad_dtype(grad_dtype)
        self.comm_bytes = 0
        self._bufs = {}
        self._sess = None
        self._comm = 0.0
        self.opened = False
        self.closed = False

    def open(self, sess):
        self._sess = sess
        self.opened = True
        self._bufs = {b: np.zeros(int(sess.plan.bucket_elems[b]), np.float32)
                      for b in range(sess.plan.nbuckets)}

    def stage(self, slot, off, i, nbytes):
        p = self._sess.params[i]
        b = self._sess.plan.param_bucket[i]
        n = nbytes // 4
        dst = self._bufs[b][off // 4: off // 4 + n]
        src = np.asarray(p.grad, np.float32).reshape(-1)
        if src.size != n:
            raise RuntimeError("numpy 传输：梯度大小 %d != 期望 %d" % (src.size, n) +
                               "。修法：这是测试用直通传输的内部校验；检查会话 params 是否被中途替换（真实训练请用 NCCL/host 传输）")
        if self.grad_dtype == "fp32":
            dst[...] = src
            return
        D = _dist_lowp()
        bits = D.quantize_to_bits(src, self.grad_dtype)
        dst[...] = D.dequantize_from_bits(bits, self.grad_dtype)
        self.comm_bytes = int(self.comm_bytes) + int(n) * 2

    def submit(self, info):
        t0 = time.perf_counter()
        buf = self._bufs[info.index]
        for (off, i, nb) in info.zero_regions:
            buf[off // 4: off // 4 + nb // 4] = 0.0
        s = self.scale if self.scale is not None else self._sess.scale
        if self.reduce_fn is not None:
            self.reduce_fn(buf, info, self._sess)
        if s != 1.0:
            buf *= np.float32(s)
        for (off, i, nb) in info.regions:
            # ★ 写回必须**按原 shape 赋值**，不能 `reshape(-1)[...] = ...`：
            #   ORV 里矩阵权重的 `.grad` 可能是**非连续**的 strided ndarray
            #   （transpose/permute 的反向会留下 F-order 视图）。对非连续数组
            #   `reshape(-1)` 返回的是**拷贝**，往拷贝里写会静默丢掉整个写回
            #   —— 实测表现就是「一半参数根本没被归约，却没有任何报错」。
            g = np.asarray(self._sess.params[i].grad, np.float32)
            g[...] = buf[off // 4: off // 4 + nb // 4].reshape(g.shape)
        self._comm += time.perf_counter() - t0

    def wait(self, info):
        pass

    def begin_window(self):
        return time.perf_counter()

    def end_window(self, token):
        return None if token is None else (time.perf_counter() - token)

    def comm_seconds(self):
        return self._comm

    def stats(self):
        out = {"grad_dtype": self.grad_dtype}
        if self.grad_dtype != "fp32":
            out["comm_bytes"] = int(self.comm_bytes)
        return out

    def close(self):
        self.closed = True
        self._bufs = {}


# ══════════════════════════════════════════════════════════════════════════
# ② NCCL：设备侧归约，跑在自己的流上
# ══════════════════════════════════════════════════════════════════════════
class NcclBucketTransport(TransportBase):
    """`ncclAllReduce` 直接作用在桶上，桶的打包/写回都是 D2D，**不碰主机内存**。

    流序：:

        main:   [stage 各 grad → 桶] --record ev_in[b]-->
        reduce: <--wait ev_in[b]-- [allReduce] [×1/world] [D2D 写回各 grad] --record ev_done[b]-->
        main:   ...（继续反向）... 最后 <--wait ev_done[b]--

    桶槽环形复用：写 `slot` 前先 `cuStreamWaitEvent(main, ev_done[b - slots])`。
    """

    kind = "nccl"

    def __init__(self, comm, stream, rt, scale, slots=None, grad_dtype="fp32"):
        self.comm = comm
        self.stream = stream
        self.rt = rt
        self.scale = float(scale)
        self.slots = int(slots if slots is not None else resolve_slots())
        self.grad_dtype = resolve_grad_dtype(grad_dtype)
        self._bufs = []
        self._ev_in = []
        self._ev_done = []
        self._sess = None
        self._comm = 0.0
        self._n_reduced = 0
        self._bucket_ms = {}
        self._dt = None
        self._zero1 = None
        self.comm_bytes = 0

    def open(self, sess):
        self._sess = sess
        rt = self.rt
        nb = int(sess.plan.max_bucket_bytes)
        if nb <= 0:
            raise OverlapUnavailable("桶总字节为 0（没有要归约的梯度）" +
                                 "。修法：确认反向前已把参数加入 begin_overlap(params)；若模型没有任何可训练参数，请设 TG_DP_OVERLAP=0")
        if self.grad_dtype == "fp32":
            self._bufs = [rt.empty((nb // 4,), dtype=np.float32) for _ in range(self.slots)]
        else:
            # ★ 低精度桶缓冲：**元素数不变、字节数减半**（bf16/fp16 都是 2 字节/元素）。
            #   用 uint16 存位模式，cast 走 dist 里那两个已存在的 CUDA 内核。
            if nb % 4:
                raise OverlapUnavailable(
                    "桶字节数 %d 不是 4 的倍数，低精度桶布局不成立" % (nb,) +
                    "。修法：设 TG_DP_GRAD_DTYPE=fp32，或把 TG_DP_BUCKET_MB 设为 4 的倍数（MB）")
            self._bufs = [rt.empty((nb // 4,), dtype=np.uint16) for _ in range(self.slots)]
            self._dt = _nccl_dtype_for(self.grad_dtype)
            self._zero1 = rt.zeros((1,), dtype=np.float32)
        self._ev_in = [rt.event_create() for _ in range(sess.plan.nbuckets)]
        self._ev_done = [rt.event_create() for _ in range(sess.plan.nbuckets)]
        #: 每个桶在**归约流**上真正开始/结束的 event（用于测「真实通信时长」；
        #  `_ev_in` 在主流量上、含排队时间，不能拿来算忙碌时长）
        self._ev_start = [rt.event_create() for _ in range(sess.plan.nbuckets)]

    def stage(self, slot, off, i, nbytes):
        g = getattr(self._sess.params[i], "grad", None)
        if g is None or getattr(g, "_gpu", None) is None:
            raise OverlapUnavailable(
                "参数 #%d 的梯度不在 GPU 上（NCCL 重叠归约只支持 CUDA）" % (i,) +
                "。修法：TG_DP_BACKEND=nccl 只能配 CUDA 训练；CPU 后端请设 TG_DP_OVERLAP=0")
        if self.grad_dtype == "fp32":
            # ★ 必须用**主流量上的异步** D2D：同步版 cuMemcpyDtoD 会挡住主机线程，
            #   把反向流水打断（实测 bwd 0.10s → 0.22s，重叠收益全丢）。异步只入队，
            #   顺序仍由主流量保证（grad 写完 → 拷进桶），ev_in 也在主流量上之后记录。
            self.rt.copy_d2d(self._bufs[slot], g._gpu, nbytes, dst_off=off, src_off=0,
                             stream=self.rt._stream)
            return
        # 低精度：先乘 1/world（stage 里做，省掉一次 fp32 归约后的缩放），再收窄进桶。
        # bf16 下乘 2 的幂是精确的 ⇒ world=2 时与不重叠路径「先求和再乘」逐位一致。
        n = int(nbytes) // 4
        self.rt.scale_inplace(g._gpu, self.scale, stream=self.rt._stream)
        _dist_lowp()._cast_f32_to_16(self.rt, self.grad_dtype, g._gpu, self._bufs[slot], n,
                                     dst_off=int(off) // 4, stream=self.rt._stream)
        self.comm_bytes = int(self.comm_bytes) + n * 2

    def await_slot(self, slot, prev_bucket):
        self._wait_bucket(prev_bucket)

    def _wait_bucket(self, b):
        if b is not None and 0 <= int(b) < len(self._ev_done):
            self.rt.cu.cuStreamWaitEvent(self.rt._stream, self._ev_done[int(b)], 0)

    def submit(self, info):
        rt = self.rt
        st = self.stream
        b = info.index
        slot = info.slot
        buf = self._bufs[slot]
        for (off, i, nb) in info.zero_regions:
            # ★ 不养一块 256MB 的常驻零缓冲（那是显存红线里的 1/4）：直接在桶上清零。
            #   缺失区很少，代价可忽略。低精度下 2 字节填 0 没有现成内核，
            #   用 dist._zero_16_region（D32 字 + 单元素 cast，绝不越界写）。
            if self.grad_dtype == "fp32":
                rt.scale_inplace(rt.sub_view(buf, (nb // 4,), off // 4), 0.0,
                                 stream=rt._stream)
            else:
                _dist_lowp()._zero_16_region(rt, self.grad_dtype, buf, off // 4,
                                             nb // 4, self._zero1, stream=rt._stream)
        rt.cu.cuEventRecord(self._ev_in[b], rt._stream)
        rt.cu.cuStreamWaitEvent(st, self._ev_in[b], 0)
        rt.cu.cuEventRecord(self._ev_start[b], st)
        # ★ 必须把 `st` 传给 all_reduce：否则 NCCL 会把内核发在**通信器自己的流**上，
        #   而 stage/writeback/ev_done 都在 `st` 上 —— 两条流之间没有依赖，
        #   writeback 可能先跑完（把旧桶内容写回梯度）⇒ 梯度静默算错。
        #   实测症状：gnorm 偏小、rank0 与 rank1 的 gnorm 不相等。
        self.comm.all_reduce_group([(int(buf.ptr), int(info.elems))], stream=st,
                                   dtype=self._dt)
        self._n_reduced += 1
        if self.grad_dtype == "fp32" and self.scale != 1.0:
            rt.scale_inplace(buf, self.scale, stream=st)
        for (off, i, nb) in info.regions:
            g = getattr(self._sess.params[i], "grad", None)
            if g is None or getattr(g, "_gpu", None) is None:
                raise OverlapUnavailable(
                    "参数 #%d 的梯度在归约期间丢了（写回阶段）" % (i,) +
                    "。修法：说明归约过程中 .grad 被外部清空了；别在反向/归约期间手动改 .grad，或设 TG_DP_OVERLAP=0")
            if self.grad_dtype == "fp32":
                rt.copy_d2d(g._gpu, buf, nb, dst_off=0, src_off=off, stream=st)
            else:
                # 低精度：桶里是「已乘过 1/world 的和」，dequant 后直接覆盖 p.grad
                _dist_lowp()._cast_16_to_f32(rt, self.grad_dtype, buf, g._gpu, nb // 4,
                                             src_off=off // 4, stream=st)
        rt.cu.cuEventRecord(self._ev_done[b], st)

    def begin_window(self):
        ev = self.rt.event_create()
        self.rt.cu.cuEventRecord(ev, self.rt._stream)
        return ev

    def wait(self, info):
        self._wait_bucket(info.index)

    def end_window(self, token):
        rt = self.rt
        n = self._sess.plan.nbuckets
        last = self._ev_done[n - 1]
        # 所有桶都在同一条流上 ⇒ 最后一个完成即全部完成
        rt.event_sync(last)
        if token is None:
            return None
        rt.event_sync(token)
        ms = rt.event_elapsed_ms(token, last)
        rt.event_destroy(token)
        # ★ 负数 = 最后一个桶在反向结束**之前**就跑完了（完全藏住）⇒ 暴露时间按 0 记。
        #   不 clamp 会在日志里出现负的 exposed_ms，看起来像 bug。
        return max(0.0, float(ms) / 1000.0)

    def comm_seconds(self):
        rt = self.rt
        tot = 0.0
        for info in self._sess._infos:
            b = info.index
            try:
                self._bucket_ms[b] = float(
                    rt.event_elapsed_ms(self._ev_start[b], self._ev_done[b]))
                tot += self._bucket_ms[b]
            except Exception:                          # noqa: BLE001
                pass
        return tot / 1000.0

    def close(self):
        rt = self.rt
        for ev in (list(getattr(self, "_ev_in", []))
                   + list(getattr(self, "_ev_done", []))
                   + list(getattr(self, "_ev_start", []))):
            rt.event_destroy(ev)
        self._ev_in, self._ev_done, self._ev_start = [], [], []
        self._bufs = []

    def stats(self):
        out = {"nccl_reduced": int(self._n_reduced), "nccl_stream": "own",
               "grad_dtype": self.grad_dtype}
        if self.grad_dtype != "fp32":
            out["comm_bytes"] = int(self.comm_bytes)
            out["nccl_dtype"] = int(self._dt) if self._dt is not None else None
        if os.environ.get("TG_DP_OV_DEBUG") and self._bucket_ms:
            out["bucket_comm_ms"] = {int(k): round(v, 3)
                                     for k, v in sorted(self._bucket_ms.items())}
        return out


# ══════════════════════════════════════════════════════════════════════════
# ③ 主机中转：后台线程 + pinned 双缓冲（**如实标注上限**）
# ══════════════════════════════════════════════════════════════════════════
class HostBucketTransport(TransportBase):
    """后台线程把桶 D2H → 共享内存 → barrier → 对端 H2D + 就地加 → 写回。

    ## 上限（别把它当 NCCL 用）

    * 物理成本与不重叠版**相同**：3.16 GiB D2H + 3.16 GiB H2D，无 P2P 时约
      140–150 ms。重叠只能把它藏进反向里，**藏不住的部分 ≈ 归约时长 − 反向时长**。
    * 每个桶要一次**文件 barrier**（`dist._barrier`，2 ms 轮询）：桶越多同步点越多；
      两边反向进度不一致时先到的 rank 会等 —— 这是这条路径的真实代价。
    * D2H/H2D 走 `cuMemcpy*Async`（要求目标/源是页锁定内存）。共享内存必须
      `cuMemHostRegister` 成功，否则 `open()` 直接抛 ⇒ 退回不重叠路径。
    """

    kind = "host"

    def __init__(self, dp, rt, scale, slots=None, grad_dtype="fp32"):
        self.dp = dp
        self.rt = rt
        self.scale = float(scale)
        self.slots = int(slots if slots is not None else max(2, resolve_slots()))
        self.grad_dtype = resolve_grad_dtype(grad_dtype)
        self._esz = 4 if self.grad_dtype == "fp32" else 2
        self._bufs = []
        self._peer = None
        self._peer16 = None      # 低精度：对端数据（2 字节）
        self._tmp32 = None       # 低精度：对端 dequant 后的 fp32 临时区
        self._zero1 = None
        self.comm_bytes = 0
        self._shm = None
        self._shm_total = 0
        self._shm_ptr = 0
        self._ev_in = []
        self._ev_done = []
        self._ev_d2h = []
        self._stream = None
        self._stream_d2h = None
        self._batch = 1
        self._sess = None
        self._q = None
        self._th = None
        self._stop = None
        self._err = None
        self._comm = 0.0
        self._n = 0

    def open(self, sess):
        rt = self.rt
        self._sess = sess
        self.dp._start()
        # 低精度下共享内存/桶缓冲都按 2 字节/元素：字节数减半，元素数不变。
        bucket_elems = int(sess.plan.max_bucket_bytes) // 4
        if bucket_elems <= 0:
            raise OverlapUnavailable("桶总字节为 0（没有要归约的梯度）" +
                                 "。修法：确认 begin_overlap(params) 的 params 非空且非全 0 元素；否则设 TG_DP_OVERLAP=0")
        nb = bucket_elems * self._esz            # 每块桶的**通信字节数**
        total = (int(sess.plan.total_bytes) // 4) * self._esz
        self._stream = self._new_stream()          # H2D/加法/写回
        self._stream_d2h = self._new_stream()      # D2H（与上面那条流真并发）
        # ★ 批大小必须 <= 槽数：同一批里若有两个桶共用一个 slot，后一个的 D2H 会在
        #   主流程还没 stage 它时就开跑（主流程要等前一个 ev_done 才能复用该 slot）。
        env_b = os.environ.get("TG_DP_HOST_BATCH", "")
        try:
            self._batch = int(env_b) if str(env_b).strip() != "" else self.slots
        except (TypeError, ValueError):
            self._batch = self.slots
        self._batch = max(1, min(int(self._batch), int(self.slots)))
        if self.grad_dtype == "fp32":
            self._bufs = [rt.empty((bucket_elems,), dtype=np.float32)
                          for _ in range(self.slots)]
            self._peer = rt.empty((bucket_elems,), dtype=np.float32)
        else:
            # ★ 元素数必须与 fp32 相同（bucket_elems），只是每元素 2 字节。
            self._bufs = [rt.empty((bucket_elems,), dtype=np.uint16)
                          for _ in range(self.slots)]
            self._peer16 = rt.empty((bucket_elems,), dtype=np.uint16)
            self._tmp32 = rt.empty((bucket_elems,), dtype=np.float32)
            self._zero1 = rt.zeros((1,), dtype=np.float32)
        self._shm_total = total
        self.dp._ensure_slot(total)
        self._shm = self.dp._slot
        reg = rt.host_register(self._shm.buf)
        if reg is None:
            raise OverlapUnavailable(
                "共享内存注册页锁定失败（cuMemHostRegister）" +
                "。修法：设 TG_DP_OVERLAP=0 用不重叠的 host 路径")
        self._shm_ptr = int(reg.ptr.value)
        self._ev_in = [rt.event_create() for _ in range(sess.plan.nbuckets)]
        self._ev_done = [rt.event_create() for _ in range(sess.plan.nbuckets)]
        self._ev_d2h = [rt.event_create() for _ in range(sess.plan.nbuckets)]
        self._q = queue.Queue()
        self._stop = object()
        self._th = threading.Thread(target=self._worker, name="tg-dp-overlap", daemon=True)
        self._th.start()

    def _new_stream(self):
        import ctypes

        rt = self.rt
        pvoid = ctypes.POINTER(ctypes.c_void_p)
        create = rt._bind_sym("cuStreamCreate", [pvoid, ctypes.c_uint])
        if create is None:
            raise OverlapUnavailable("驱动缺少 cuStreamCreate" +
                                 "。修法：升级显卡驱动；或设 TG_DP_OVERLAP=0 用不重叠路径")
        st = ctypes.c_void_p()
        # CU_STREAM_NON_BLOCKING = 1：**必须**。默认的 blocking 流会与 legacy
        # NULL 流隐式互相同步，重叠直接归零。
        rc = int(create(ctypes.byref(st), 1))
        if rc != 0 or not st.value:
            raise OverlapUnavailable("cuStreamCreate(非阻塞) 失败 rc=%d" % rc +
                                 "。修法：看 rc 对应的 CUDA 错误（显存/上下文）；急着训练就设 TG_DP_OVERLAP=0")
        return st

    def stage(self, slot, off, i, nbytes):
        g = getattr(self._sess.params[i], "grad", None)
        if g is None or getattr(g, "_gpu", None) is None:
            raise OverlapUnavailable("参数 #%d 的梯度不在 GPU 上" % (i,) +
                                 "。修法：host 重叠传输只支持 CUDA 张量；CPU 后端设 TG_DP_OVERLAP=0")
        # 同 NCCL 路径：必须异步（同步版会把主机线程挡住，打断反向流水）
        if self.grad_dtype == "fp32":
            self.rt.copy_d2d(self._bufs[slot], g._gpu, nbytes, dst_off=off, src_off=0,
                             stream=self.rt._stream)
            return
        # 低精度：先乘 1/world（bf16 下乘 2 的幂精确）再收窄进桶；写回时 dequant 即均值
        n = int(nbytes) // 4
        self.rt.scale_inplace(g._gpu, self.scale, stream=self.rt._stream)
        _dist_lowp()._cast_f32_to_16(self.rt, self.grad_dtype, g._gpu, self._bufs[slot], n,
                                     dst_off=int(off) // 4, stream=self.rt._stream)
        self.comm_bytes = int(self.comm_bytes) + n * 2

    def await_slot(self, slot, prev_bucket):
        if prev_bucket is not None and 0 <= int(prev_bucket) < len(self._ev_done):
            self.rt.cu.cuStreamWaitEvent(self.rt._stream,
                                         self._ev_done[int(prev_bucket)], 0)

    def submit(self, info):
        rt = self.rt
        slot = info.slot
        for (off, i, nb) in info.zero_regions:
            if self.grad_dtype == "fp32":
                rt.scale_inplace(rt.sub_view(self._bufs[slot], (nb // 4,), off // 4), 0.0)
            else:
                # 低精度：2 字节区填 0（D32 字 + 单元素 cast，不越界写相邻参数）
                _dist_lowp()._zero_16_region(rt, self.grad_dtype, self._bufs[slot],
                                             off // 4, nb // 4, self._zero1)
        rt.cu.cuEventRecord(self._ev_in[info.index], rt._stream)
        if self._err is not None:
            raise OverlapUnavailable("host 重叠线程已经失败：%s" % (self._err,) +
                                 "。修法：按上面线程里的原始异常修；急用可设 TG_DP_OVERLAP=0 退回不重叠路径")
        self._n += 1
        self._q.put(info)

    def _worker(self):
        import ctypes

        rt = self.rt
        st = self._stream
        st_d2h = self._stream_d2h
        total = int(self._shm_total)          # 共享内存里每 rank 的字节数（低精度时是 fp32 的一半）
        own_base = self.dp.rank * total
        plan = self._sess.plan
        n = int(plan.nbuckets)
        B = int(self._batch)
        # ★ 后台线程必须**自己**把 CUDA 上下文设为当前：`CudaRuntime._activate()`
        #   用一个进程级的 `_ctx_cur` 缓存（主线程早就置 True），新线程第一次调用会
        #   直接 return ⇒ 这个线程上所有 CUDA 调用都 rc=201 invalid device context
        #   （实测：cuEventCreate 直接失败，随后 ev_done 永不记录，主线程在
        #   cuEventElapsedTime 上拿到 rc=400，报成「invalid resource handle」）。
        try:
            rt._deactivate_hint()
            rt._activate()
        except Exception as exc:                       # noqa: BLE001
            self._err = exc
            return
        b0 = 0
        lowp = self.grad_dtype != "fp32"
        esz = int(self._esz)
        D = _dist_lowp() if lowp else None
        try:
            while b0 < n:
                b1 = min(n, b0 + B)
                infos = []
                # ① 收齐这一批（阻塞等；主流程按 FIFO 提交）
                for b in range(b0, b1):
                    item = self._q.get()
                    if item is self._stop:
                        return
                    infos.append(item)
                t0 = time.perf_counter()
                # ② 批量发起 D2H（异步，直接写进已注册共享内存自己那一段）
                for info in infos:
                    b = info.index
                    n_elems = int(info.elems)
                    off_b = int(plan.bucket_off[b]) * esz
                    rt.cu.cuStreamWaitEvent(st_d2h, self._ev_in[b], 0)
                    rt.d2h_into(self._bufs[info.slot],
                                ctypes.c_void_p(self._shm_ptr + own_base + off_b),
                                nbytes=n_elems * esz, stream=st_d2h)
                    rt.cu.cuEventRecord(self._ev_d2h[b], st_d2h)
                # ③ 这批 D2H 全完成 → **一次** barrier
                for info in infos:
                    rt.event_sync(self._ev_d2h[info.index])
                self.dp._barrier("ov")
                # ④ 对端：H2D 到 peer，再就地加进桶（rank 升序，与不重叠路径一致）
                for info in infos:
                    b = info.index
                    n_elems = int(info.elems)
                    off_b = int(plan.bucket_off[b]) * esz
                    if not lowp:
                        for r in range(self.dp.world):
                            if r == self.dp.rank:
                                continue
                            src = self._shm_ptr + r * total + off_b
                            rt.h2d_into(self._peer, ctypes.c_void_p(src),
                                        nbytes=n_elems * 4, stream=st)
                            rt.add_(self._bufs[info.slot], self._peer, stream=st)
                        if self.scale != 1.0:
                            rt.scale_inplace(self._bufs[info.slot], self.scale, stream=st)
                        # ⑤ 写回各 grad（同一条流，顺序在加/缩放之后）
                        for (off, i, nbb) in info.regions:
                            g = getattr(self._sess.params[i], "grad", None)
                            if g is None or getattr(g, "_gpu", None) is None:
                                raise OverlapUnavailable(
                                    "参数 #%d 的梯度在写回阶段丢了" % (i,) +
                                    "。修法：归约期间 .grad 被外部清空；"
                                    "不要在反向/归约期间手动改 .grad，或设 TG_DP_OVERLAP=0")
                            rt.copy_d2d(g._gpu, self._bufs[info.slot], nbb,
                                        dst_off=0, src_off=off, stream=st)
                    else:
                        # 低精度：本 rank 的 grads 在 stage 时已经被乘过 1/world，
                        # 这里只需把**对端**的量化值 dequant 成 fp32 后加进去。
                        # 自贡献 + Σ对端 ≈ mean(梯度)；写回步骤因此可以省掉。
                        for r in range(self.dp.world):
                            if r == self.dp.rank:
                                continue
                            src = self._shm_ptr + r * total + off_b
                            rt.h2d_into(self._peer16, ctypes.c_void_p(src),
                                        nbytes=n_elems * esz, stream=st)
                            D._cast_16_to_f32(rt, self.grad_dtype, self._peer16,
                                              self._tmp32, n_elems, stream=st)
                            for (off, i, nbb) in info.regions:
                                g = getattr(self._sess.params[i], "grad", None)
                                if g is None or getattr(g, "_gpu", None) is None:
                                    raise OverlapUnavailable(
                                        "参数 #%d 的梯度在写回阶段丢了" % (i,) +
                                        "。修法：归约期间 .grad 被外部清空；"
                                        "不要在反向/归约期间手动改 .grad，或设 TG_DP_OVERLAP=0")
                                rt.add_(g._gpu,
                                        rt.sub_view(self._tmp32, (nbb // 4,), off // 4),
                                        stream=st)
                    rt.cu.cuEventRecord(self._ev_done[b], st)
                self._comm += time.perf_counter() - t0
                b0 = b1
        except Exception as exc:                       # noqa: BLE001
            self._err = exc


    def begin_window(self):
        # host 路径的「暴露时间」就是主线程从反向结束到归约全部完成的**墙钟**：
        # 这条路径本来就要主机等文件 barrier，用 event 计时没有意义
        # （而且 worker 线程的 event 在 CUDA 上跨线程计时会踩 rc=400）。
        return time.perf_counter()

    def wait(self, info):
        if self._err is not None:
            raise OverlapUnavailable("host 重叠线程已经失败：%s" % (self._err,) +
                                 "。修法：按上面线程里的原始异常修；急用可设 TG_DP_OVERLAP=0 退回不重叠路径")
        self.rt.cu.cuStreamWaitEvent(self.rt._stream, self._ev_done[info.index], 0)

    def end_window(self, token):
        rt = self.rt
        last = self._ev_done[self._sess.plan.nbuckets - 1]
        rt.event_sync(last)
        if self._err is not None:
            # 走到这里说明部分桶已经归约/写回，**不能**再静默退回不重叠路径
            # （那会把「已平均过的梯度」再平均一次）。必须响亮地停。
            raise RuntimeError(
                "host 重叠归约线程失败：%s" % (self._err,) +
                "。修法：设 TG_DP_OVERLAP=0 重启训练（这条路径不完整，绝不能继续）")
        if token is None:
            return None
        return float(time.perf_counter() - token)

    def comm_seconds(self):
        return self._comm

    def close(self):
        rt = self.rt
        try:
            if self._th is not None and self._th.is_alive():
                self._q.put(self._stop)
                self._th.join(timeout=2.0)
        except Exception:                              # noqa: BLE001
            pass
        self._th = None
        for st in (self._stream, self._stream_d2h):
            if st is None:
                continue
            cu = getattr(rt, "cu", None)
            destroy = None
            if cu is not None:
                destroy = (getattr(cu, "cuStreamDestroy_v2", None)
                           or getattr(cu, "cuStreamDestroy", None))
            if destroy is not None:
                try:
                    destroy(st)
                except Exception:                      # noqa: BLE001
                    pass
        self._stream, self._stream_d2h = None, None
        for ev in list(self._ev_in) + list(self._ev_done) + list(self._ev_d2h):
            rt.event_destroy(ev)
        self._ev_in, self._ev_done, self._ev_d2h = [], [], []
        self._bufs = []
        self._peer = None
        self._peer16 = None
        self._tmp32 = None
        self._zero1 = None

    def stats(self):
        out = {"host_thread": "yes", "host_buckets": int(self._n),
               "host_batch": int(self._batch),
               "host_dual_stream": bool(self._stream_d2h is not None),
               "grad_dtype": self.grad_dtype}
        if self.grad_dtype != "fp32":
            out["comm_bytes"] = int(self.comm_bytes)
        return out
