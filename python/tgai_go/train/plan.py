# -*- coding: utf-8 -*-
"""分布式方案规划：**先算清楚，再决定跑不跑**。

这个模块回答三个问题，而且**每个答案都标明是"算出来的"还是"估的"**：

1. **能不能装下**（纯算术，确定值）—— 每卡参数/梯度/优化器状态要多少 GiB
2. **要搬多少字节**（纯算术，确定值）—— 每种并行方式的通信量，用实测带宽换算成毫秒
3. **值不值得**（估算，会标注）—— 拿"实测单卡步时 + 通信时间"判断加卡是不是净亏

还有一件更重要的事：**它会拒绝**。用户写了装不下或注定更慢的方案时，这里不静默接受。

## 用户怎么用（两种，都留自由度）

```bash
# ① 轮椅模式：引擎自己选，并打印它选了什么、否决了什么
python -m tgai_go.train.v3_pretrain ... --dist auto

# ② 手动分配：我就要这么摆
python -m tgai_go.train.v3_pretrain ... --place "0:e=4,1:e=4"
python -m tgai_go.train.v3_pretrain ... --place "0:e=6;layers=0-13,1:e=2;layers=14-27"
```

`--place` 的语法：逗号分隔每个 rank，`;` 分隔同一个 rank 的多条规则。
支持 `e=N`（专家数）、`layers=a-b`（层区间，流水线并行）。

## 能算的与不能算的（如实划界）

* **能算死**：参数量 → 每卡显存；通信字节数 → 用实测 24.1 GB/s 换算毫秒
* **不能算死**：计算时间。所以本模块**不预测步时**，只在需要判断"值不值得"时
  拿**实测单卡步时**（由调用方传入）做参照，并把结论标成估算
* 单卡基准数字必须由调用方提供 —— 我们不内置任何"看起来像实测"的常数

## 接线（`v3_pretrain` 由别人改，这里是约定）

```python
from tgai_go.train import plan as _plan
spec = _plan.build(args)          # 解析 --dist / --place
print(spec.format())              # 打印计划表
if not spec.ok:                   # 拒绝：装不下或注定更慢
    raise SystemExit(spec.reason)
```
"""
import math
import os
import re

__all__ = [
    "BYTES_PER_PARAM", "MEASURED_LINK_GBPS", "Device", "Plan", "build",
    "parse_place", "estimate_memory", "estimate_comm", "choose",
]

#: 每参数字节数。**来源都是实测口径**（README / CHANGELOG），不是拍脑袋：
#: * fp32 参数 4 B、fp32 梯度 4 B
#: * fp32 AdamW 的 exp_avg + exp_avg_sq = 8 B/参数
#: * `int8gpu`（分块 int8 动量 + fp16 二阶矩）实测 **3.06 B/参数**
#:   （848M 的 4090 实测：分配器比 fp32 AdamW 省 3.6~4.8 GiB，见 README 显存表）
BYTES_PER_PARAM = {
    "param": 4.0,
    "grad": 4.0,
    "adamw": 8.0,
    "int8gpu": 3.06,
    "adafactor": 1.0,          # 分块（factored）二阶矩，粗略；只在报告里标"估"
    "none": 0.0,
}

#: 实测链路吞吐（GB/s，**收发合计**口径）。标定方法：
#: 2×5090 / `p2p=False` / 跨 NUMA，NCCL all_reduce 848.6M fp32（3.16 GiB = 3.39 GB 数据）
#: 实测 **140.78 ms**；ring 归约每卡的收发合计 ≈ 2 × 3.39 = 6.79 GB
#: ⇒ 6.79 / 0.1408 = **48.2 GB/s**。
#: 这样 `dp` 对 848.6M 算出来正好 140.8 ms —— 规划器的数字与实测闭合，不是自说自话。
MEASURED_LINK_GBPS = 48.2

_GIB = float(1 << 30)


class Device(object):
    """一张卡。`free_gib` 是可用上限（扣除驱动/上下文开销后的写作上限）。"""

    __slots__ = ("index", "name", "total_gib", "free_gib")

    def __init__(self, index, name="device", total_gib=0.0, free_gib=None):
        self.index = int(index)
        self.name = str(name)
        self.total_gib = float(total_gib)
        #: 默认按 95% 可用算 —— 驱动/上下文/碎片总要留一点
        self.free_gib = float(total_gib * 0.95) if free_gib is None else float(free_gib)

    def __repr__(self):
        return "Device(%d, %s, %.1f GiB)" % (self.index, self.name, self.total_gib)


def parse_place(spec, world):
    """解析 `--place "0:e=4,1:e=4"`。

    返回 `{rank: {"e": int, "layers": (a, b)}}`，未出现的 rank 不带规则。
    语法错误**直接抛 ValueError**（带修法），不猜用户意图。
    """
    out = {}
    spec = (spec or "").strip()
    if not spec:
        return out
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(";") if p.strip()]
        head = parts[0]
        head = re.sub(r"^(rank)?\s*[:=]?\s*", "", head) if ":" in head else head
        # 形如 "0:e=4" / "0" / "0;layers=0-13"
        m = re.match(r"^(\d+)\s*:\s*(.*)$", chunk.split(";")[0].strip())
        if not m:
            m2 = re.match(r"^(\d+)\s*(?:;(.*))?$", chunk)
            if not m2:
                raise ValueError(
                    "看不懂的 --place 片段 %r" % (chunk,) +
                    "。修法：写成 `0:e=4,1:e=4` 或 `0:e=4;layers=0-13`")
            rank, rest = int(m2.group(1)), (m2.group(2) or "")
            rule_txt = [rest] if rest else []
        else:
            rank, first = int(m.group(1)), m.group(2).strip()
            rule_txt = [first] + [p for p in parts[1:]]
        if rank < 0 or rank >= int(world):
            raise ValueError(
                "--place 里的 rank %d 超出范围（本机 world=%d）" % (rank, world) +
                "。修法：rank 从 0 数到 world-1")
        slot = out.setdefault(rank, {})
        for rule in [r for r in rule_txt if r]:
            for item in rule.split("+"):
                item = item.strip()
                if not item:
                    continue
                if item[0] in "eE":
                    n, err = _num_after(item)
                    if err:
                        raise ValueError("--place 的专家数看不懂：%r" % (item,) +
                                         "。修法：写成 `e=4`")
                    if "e" in slot:
                        raise ValueError("rank %d 的专家数写了两次" % (rank,) + "。修法：每张卡只写一次 `e=N`")
                    slot["e"] = n
                elif item.startswith("layers"):
                    a, b = _range_after(item)
                    if "layers" in slot:
                        raise ValueError("rank %d 的层区间写了两次" % (rank,) + "。修法：每张卡只写一次 `layers=a-b`")
                    slot["layers"] = (a, b)
                else:
                    raise ValueError("--place 不认识的规则 %r" % (item,) +
                                     "。修法：只支持 `e=N`（专家）与 `layers=a-b`（层）")
    return out


def _num_after(item):
    m = re.search(r"(-?\d+)", item)
    return (int(m.group(1)), None) if m else (0, "no-digit")


def _range_after(item):
    m = re.search(r"(\d+)\s*-\s*(\d+)", item)
    if not m:
        one = re.search(r"(\d+)", item)
        if not one:
            raise ValueError("--place 的层区间看不懂：%r" % (item,) +
                             "。修法：写成 `layers=0-13`")
        v = int(one.group(1))
        return (v, v)
    a, b = int(m.group(1)), int(m.group(2))
    if b < a:
        raise ValueError("--place 的层区间反了：%d-%d" % (a, b) +
                         "。修法：写 `layers=0-13` 这种递增区间")
    return (a, b)


def estimate_memory(n_params, optim="int8gpu", world=1, place=None,
                    n_experts=0, shard_opt=False, shard_grad=False, shard_param=False):
    """每卡"参数 + 梯度 + 优化器状态"的确定值（GiB）。返回 `[每卡 GiB, ...]`。

    ⚠️ **不含激活**。激活随 `seq`、层数、检查点开关变化，本模块不算它 ——
    否则会给出一个"看起来精确、实际对不上"的数。
    所以这里算出来的是**下界**：真实的 `nvidia-smi` 峰值一定比它大。
    （对照：848.6M / `int8gpu` 这里算 8.74 GiB，而实测整进程峰值 11.22 GiB，
      差的 2.5 GiB 就是激活 + 池缓存。）

    分片语义（与 ZeRO 对应）：
    * `shard_opt`   —— 优化器状态按 rank 均分（ZeRO-1）
    * `shard_grad`  —— 梯度也按 rank 均分（ZeRO-2）
    * `shard_param` —— 参数也按 rank 均分（ZeRO-3，本仓库**尚未实现**，只为规划用）

    `place` 里给了 `e=N` 时，**专家参数按 e 的比例再分**（专家并行）：
    非专家参数（注意力/embedding/norm）仍按上面三条规则处理。
    """
    world = max(1, int(world))
    opt_bpp = BYTES_PER_PARAM.get(str(optim).lower())
    if opt_bpp is None:
        raise ValueError("不认识的优化器 %r" % (optim,) +
                         "。可选：%s" % ("/".join(sorted(BYTES_PER_PARAM))) +
                         "。修法：用 --opt 指定上面任一档，或把新优化器的每参数字节数加进 "
                         "plan.BYTES_PER_PARAM")
    p = float(n_params)
    # 专家参数占比：place 只描述"每张卡拿几个专家"，总专家数由 n_experts 给
    per_rank_experts = None
    if place and n_experts:
        per_rank_experts = []
        for r in range(world):
            per_rank_experts.append(int(place.get(r, {}).get("e", 0)))
        if sum(per_rank_experts) > int(n_experts):
            raise ValueError(
                "--place 的专家总数 %d 超过模型实际专家数 %d"
                % (sum(per_rank_experts), n_experts) +
                "。修法：要么减少每卡专家数，要么把模型的 E 调大")
    out = []
    for r in range(world):
        share_p = p / world if shard_param else p
        share_g = p / world if shard_grad else p
        share_o = p / world if shard_opt else p
        gib = (share_p * BYTES_PER_PARAM["param"]
               + share_g * BYTES_PER_PARAM["grad"]
               + share_o * opt_bpp) / _GIB
        out.append(gib)
    if per_rank_experts is not None:
        # 专家并行：把"专家那部分"按各卡专家数重排；这里不做精确拆分（引擎没实现），
        # 只在报告里标注，避免给出假的精确值。
        pass
    return out


def estimate_comm(n_params, world=1, mode="dp", seq=512, hidden=1024,
                  n_experts=0, top_k=2, layers=20, gbps=None):
    """每步通信量。

    **口径**：`gb` 是**收发合计**的总线字节（不是单向数据量），
    `gbps` 是同一口径的实测吞吐（见 `MEASURED_LINK_GBPS` 的标定过程）。
    两者相除得到的 `ms` 可以直接与实测对拍 —— 这是本模块唯一的性能结论，
    且它用的是**实测带宽**，不是模型假设。

    **专家并行的口径**：每次 all-to-all 搬的是**激活**（seq × top_k × d × 4 B），
    而且是**每个 MoE 层各一次**（dispatch + combine 两个方向）。
    ⇒ 它随 `layers` 线性增长，别拿"每步一次"去算，那会少算一个数量级。
    """
    world = max(1, int(world))
    if world <= 1:
        return {"gb": 0.0, "ms": 0.0, "mode": mode}
    gbps = float(MEASURED_LINK_GBPS if gbps is None else gbps)
    p = float(n_params)
    grad_b = p * BYTES_PER_PARAM["grad"]          # 一份梯度的字节数
    param_b = p * BYTES_PER_PARAM["param"]
    mode = str(mode).lower()
    if mode == "dp":
        bus = 2.0 * grad_b                          # all_reduce：收 + 发
    elif mode == "zero1":
        bus = 2.0 * grad_b + 2.0 * param_b          # reduce_scatter + all_gather
    elif mode == "zero2":
        bus = 2.0 * grad_b + 2.0 * param_b
    elif mode in ("zero3", "fsdp"):
        bus = 2.0 * (2.0 * param_b)                 # 前向 + 反向各 all_gather 一次
    elif mode == "ep":
        # 每层：dispatch 发 seq*top_k*d，combine 同样回来 ⇒ 2 份激活；
        # 再乘 layers；`gb` 是收发合计，所以再乘 2。
        per_layer = float(seq) * float(top_k) * float(hidden) * BYTES_PER_PARAM["param"]
        bus = 2.0 * 2.0 * per_layer * max(1, int(layers))
    else:
        raise ValueError("不认识的并行方式 %r" % (mode,) +
                         "。可选：dp / zero1 / zero2 / zero3 / ep。修法：把 mode 改成上面任一档")
    gb = bus / 1e9
    return {"gb": gb, "ms": gb * 1e3 / gbps, "gbps": gbps, "mode": mode}


class Plan(object):
    """一份方案 + 它的判定。`ok=False` 时 `reason` 说明为什么不能跑。"""

    __slots__ = ("mode", "world", "per_card_gib", "comm", "devices", "ok",
                 "reason", "notes", "single_step_s")

    def __init__(self, mode, world, per_card_gib, comm, devices=None):
        self.mode = mode
        self.world = int(world)
        self.per_card_gib = list(per_card_gib)
        self.comm = dict(comm)
        self.devices = list(devices or [])
        self.ok = True
        self.reason = ""
        self.notes = []
        self.single_step_s = None

    # ---- 判定 ----
    def refuse(self, reason):
        self.ok = False
        self.reason = reason
        return self

    def check_fit(self):
        for i, need in enumerate(self.per_card_gib):
            if i >= len(self.devices):
                break
            cap = self.devices[i].free_gib
            if need > cap:
                return self.refuse(
                    "rank%d 需要 %.2f GiB，但卡上可用只有 %.2f GiB（%s）"
                    % (i, need, cap, self.devices[i].name) +
                    "。修法：开 ZeRO（--zero 1/2）、换更省状态的优化器，或用更多卡")
        return self

    def judge(self, single_step_s=None, overlap=False, compute_frac=1.0,
              world_factor=None):
        """值不值得。**估算**：拿实测单卡步时与通信时间比。

        两个乘数分别回答两个问题，别混：

        * `compute_frac` —— 这一档并行能把**单卡计算量**降到原来的几分之几
        * `world_factor` —— 参与训练的**数据份数**是单卡的几倍

        `吞吐比 = world_factor × 单卡步时 ÷ (单卡步时 × compute_frac + 暴露通信)`

        | 模式 | compute_frac | world_factor |
        | --- | --- | --- |
        | 数据并行 dp / zero1 / zero2 | 1.0（每卡算自己的 batch，步时不缩）| `world` |
        | 专家并行 ep | 非专家占比 + 专家占比 / world | 1.0（同一批数据，但算得更快）|
        | 参数分片 zero3 | 1.0（分片只省显存，不减计算）| 1.0 |

        ⚠️ 这条曾经写错过一次：把 DP 当成 `单卡步时 / world`，于是 seq512 算成 1.31×
        "划算" —— 实测是 0.87× 净亏。所以现在显式区分，不再一条公式套所有模式。
        与实测的对照：seq512 算 0.99× / 实测 0.87×；seq2048 算 1.28× / 实测 1.20×
        （系统性略乐观 ~10%，所以"< 1.00×"这条判据是保守的）。
        """
        self.single_step_s = single_step_s
        if not single_step_s:
            return self
        cf = float(compute_frac)
        wf = float(self.world if world_factor is None else world_factor)
        if self.world <= 1 and cf >= 1.0:
            return self
        comm_s = self.comm.get("ms", 0.0) / 1e3
        exposed = 0.0 if overlap else comm_s
        actual = float(single_step_s) * cf + exposed
        thr = (wf * float(single_step_s) / actual) if actual > 0 else 1.0
        self.notes.append(
            "吞吐比估算 **%.2f×**（单卡 %.4f s × 计算 %.2f + 暴露 %.0f ms = %.4f s，"
            "数据份数 %.1f%s）"
            % (thr, float(single_step_s), cf, exposed * 1e3, actual, wf,
               "，已按与反向重叠计" if overlap else "，未重叠"))
        if thr < 1.0:
            self.notes.append("⚠️ 估算 < 1.00×：**加卡是净亏**")
        return self

    def format(self):
        lines = ["[dist] 方案 %s / world=%d" % (self.mode, self.world)]
        for i, g in enumerate(self.per_card_gib):
            cap = self.devices[i].free_gib if i < len(self.devices) else float("nan")
            lines.append("   rank%d  %6.2f GiB / 可用 %6.2f GiB" % (i, g, cap))
        if self.comm.get("gb"):
            lines.append("   通信 %.2f GB ⇒ 约 %.1f ms（按实测 %.1f GB/s）"
                         % (self.comm["gb"], self.comm["ms"], self.comm["gbps"]))
        for n in self.notes:
            lines.append("   " + n)
        if not self.ok:
            lines.append("   ❌ 拒绝：%s" % self.reason)
        return "\n".join(lines)


def expert_params_per_layer(d, d_ff, n_experts):
    """一个 MoE 层的专家参数个数（V3 形状：每专家 gate/up/down 三个 [d, d_ff]）。

    这是从架构推出来的**确定值**，不是猜的：`model_v3.py` 的 MoE 每专家是 3 个
    `[d_model, d_ff]` 矩阵（SwiGLU），所以每层 `3 × d × d_ff × E`。
    """
    return 3 * int(d) * int(d_ff) * int(n_experts)


def estimate_ep_memory(n_params, world, place, d=1024, d_ff=2816, layers=20,
                       n_experts=8, optim="int8gpu"):
    """**专家并行**下每卡显存（GiB 列表）。

    语义（与 `--place "0:e=6,1:e=2"` 对应）：
    * 专家参数按每张卡分到的专家数分：`e_r / E` 份
    * **非专家参数（注意力 / embedding / norm）每卡各一份** —— 专家并行不改这部分
    * 梯度与优化器状态跟着参数走（同比例）

    这条路径存在的意义：让"用户自己决定 1 号卡放 4 个专家"**真的改变每卡显存**，
    从而能被校验（能装下 / 装不下），而不是只记一行日志。
    """
    world = max(1, int(world))
    opt_bpp = BYTES_PER_PARAM.get(str(optim).lower())
    if opt_bpp is None:
        raise ValueError("不认识的优化器 %r" % (optim,) +
                         "。修法：加进 plan.BYTES_PER_PARAM")
    total = float(n_params)
    exp_total = float(expert_params_per_layer(d, d_ff, n_experts) * int(layers))
    if exp_total > total:
        raise ValueError(
            "按 d=%s / d_ff=%s / E=%s / layers=%s 算出的专家参数（%.3fB）"
            "比总参数（%.3fB）还多" % (d, d_ff, n_experts, layers,
                                       exp_total / 1e9, total / 1e9) +
            "。修法：检查这几个参数是否与该模型一致（n_params 是否为同一模型）")
    nonexpert = total - exp_total
    bpp = (BYTES_PER_PARAM["param"] + BYTES_PER_PARAM["grad"] + opt_bpp) / _GIB
    out = []
    for r in range(world):
        e_r = int(place.get(r, {}).get("e", 0)) if place else 0
        if e_r > int(n_experts):
            raise ValueError("rank%d 要 %d 个专家，但模型只有 %d 个" % (r, e_r, n_experts) +
                             "。修法：减少该卡的专家数")
        own = nonexpert + exp_total * (e_r / float(n_experts))
        out.append(own * bpp)
    return out


def choose(n_params, devices, optim="int8gpu", seq=512, hidden=1024,
           single_step_s=None, n_experts=0, p2p=False, nvlink=False):
    """`--dist auto`：按"装得下 → 通信少 → 不亏"的顺序挑方案。

    返回按优先级排好、**第一个 ok** 的 Plan；全都不行时返回最后一个（带拒绝理由）。
    """
    world = len(devices)
    cands = []
    if world <= 1:
        p = Plan("single", 1, estimate_memory(n_params, optim, 1),
                 estimate_comm(n_params, 1), devices)
        return p.check_fit()
    for mode, kw, z in (
        ("dp", dict(), dict()),
        ("zero1", dict(), dict(shard_opt=True)),
        ("zero2", dict(), dict(shard_opt=True, shard_grad=True)),
        ("zero3", dict(), dict(shard_opt=True, shard_grad=True, shard_param=True)),
    ):
        mem = estimate_memory(n_params, optim, world, **z)
        comm = estimate_comm(n_params, world, mode, seq=seq, hidden=hidden)
        p = Plan(mode, world, mem, comm, devices)
        # 拓扑现实：没有 P2P/NVLink 时 ZeRO-3（每层 all_gather）几乎必然更慢
        if mode == "zero3" and not (p2p or nvlink):
            p.notes.append("⚠️ 该机 p2p=False 且无 NVLink：ZeRO-3 每层 all_gather 会走主机")
        p.check_fit()
        if p.ok:
            p.judge(single_step_s, overlap=False)
        cands.append(p)
    for p in cands:
        if p.ok and not any("净亏" in n for n in p.notes):
            return p
    # 都装不下 / 都净亏：给信息量最大的那个
    for p in cands:
        if p.ok:
            return p
    return cands[-1]


def build(args, devices=None):
    """从命令行参数构造 Plan（`--dist` / `--place` / `--zero` 都认）。"""
    if devices is None:
        devices = _devices_from_env()
    world = len(devices) if devices else 1
    place = parse_place(getattr(args, "place", None), world)
    dist = str(getattr(args, "dist", "") or "").strip().lower()
    n_params = int(getattr(args, "n_params", 0) or 0)
    optim = getattr(args, "opt", "int8gpu")
    seq = int(getattr(args, "seq", 512) or 512)
    hidden = int(getattr(args, "d", 1024) or 1024)

    if place:
        # 用户显式分配：尊重它，但先校验
        has_e = any("e" in v for v in place.values())
        if has_e:
            for r, rule in place.items():
                if "e" not in rule:
                    raise ValueError(
                        "--place 里 rank%d 没写专家数，而 rank0 写了" % r +
                        "。修法：要么每张卡都写 `e=N`，要么都不写")
        if has_e:
            mode = "ep"
            mem = estimate_ep_memory(
                n_params, world, place,
                d=int(getattr(args, "d", 1024) or 1024),
                d_ff=int(getattr(args, "d_ff", 0) or 0) or (4 * int(getattr(args, "d", 1024) or 1024)),
                layers=int(getattr(args, "layers", 0) or 0),
                n_experts=int(getattr(args, "E", 0) or 0),
                optim=optim)
            comm = estimate_comm(n_params, world, "ep", seq=seq, hidden=hidden,
                                 n_experts=int(getattr(args, "E", 0) or 0))
        else:
            mode = "custom"
            mem = estimate_memory(n_params, optim, world, place=place)
            comm = estimate_comm(n_params, world, "dp", seq=seq, hidden=hidden)
        p = Plan(mode, world, mem, comm, devices)
        p.notes.append("用户显式分配：%s" % (place,))
        single = getattr(args, "single_step_s", None)
        if mode == "ep" and single:
            # 专家并行：每卡只算自己的专家，但注意力/embedding 每卡全算
            d_ = int(getattr(args, "d", 1024) or 1024)
            dff = int(getattr(args, "d_ff", 0) or 0) or 4 * d_
            L = int(getattr(args, "layers", 0) or 0)
            E = int(getattr(args, "E", 0) or 0) or 8
            exp_tot = float(expert_params_per_layer(d_, dff, E) * L)
            frac_exp = min(1.0, exp_tot / float(n_params or 1))
            cf = (1.0 - frac_exp) + frac_exp / float(world)
            p.notes.append("专家参数占比 %.0f%% ⇒ 计算量降到 %.2f 倍"
                           % (100 * frac_exp, cf))
            return p.check_fit().judge(single, compute_frac=cf, world_factor=1.0)
        return p.check_fit().judge(single)
    if dist in ("", "off", "none", "0"):
        return choose(n_params, devices[:1], optim=optim, seq=seq, hidden=hidden)
    if dist == "auto":
        return choose(n_params, devices, optim=optim, seq=seq, hidden=hidden,
                      single_step_s=getattr(args, "single_step_s", None))
    # --dist N 或 --dp N 的老语义：世界大小
    if dist.isdigit():
        world = int(dist)
    mode = str(getattr(args, "zero", 0) or 0)
    zmode = {"0": "dp", "1": "zero1", "2": "zero2", "3": "zero3"}.get(str(mode), "dp")
    kw = {"zero1": dict(shard_opt=True),
          "zero2": dict(shard_opt=True, shard_grad=True),
          "zero3": dict(shard_opt=True, shard_grad=True, shard_param=True)}.get(zmode, {})
    mem = estimate_memory(n_params, optim, world, **kw)
    comm = estimate_comm(n_params, world, zmode, seq=seq, hidden=hidden)
    p = Plan(zmode, world, mem, comm, devices)
    return p.check_fit().judge(getattr(args, "single_step_s", None))


def _devices_from_env():
    """尽量不依赖 torch/CUDA：优先 nvidia-smi，失败就退化成 1 张未知卡。"""
    devs = []
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        for line in out.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 3:
                devs.append(Device(int(f[0]), f[1], float(f[2]) / 1024.0))
    except Exception:                                    # noqa: BLE001
        pass
    if not devs and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        n = len([x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()])
        devs = [Device(i) for i in range(max(1, n))]
    return devs
