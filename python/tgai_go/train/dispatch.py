# -*- coding: utf-8 -*-
"""能力模型与算子分派 —— 为多后端（CUDA / Vulkan / ROCm / CPU）铺路。

## 为什么需要它

插件「单文件适配全部后端」的前提是：**插件针对能力写，不针对后端写**。

```python
plugin.op("my_silu", fast_impl, caps={"async_copy": True})   # 有 async copy 才用
plugin.op("my_silu", slow_impl)                               # 通用回退（必须存在）
```

分派时按**当前后端的真实能力**选择实现，而不是问"你是不是 CUDA"。

## 性能纪律（硬约束）

封装层不能拖慢热路径。本模块的三条保证：

1. **单次 dict 查表**：键是 `(op, backend, dtype, arch)` 元组，一次命中；
   不做多级 fallback 链、不做 `inspect` / `getattr` 反射。
2. **结果缓存**：同一组键只解析一次，之后是纯 dict 查找。
3. **可量化**：`benchmark_overhead()` 直接测"经过分派 vs 直接调用"的额外耗时，
   实测超 1% 就应该改成静态绑定（见 docs/PLUGIN_DISPATCH.md）。

## 能力不足必须报错

若某算子在该后端上**没有任何可用实现**，分派抛 `DispatchError` 并列出缺什么能力 ——
绝不静默选一个跑不了的分支。这与插件系统"能力不足当场报错"的纪律一致。
"""

from __future__ import annotations

import time

#: 已登记的实现：(op_name, backend) -> [{"fn", "caps", "min_cc", "max_cc", "dtype", "prio"}]
_IMPLS = {}
#: 解析结果缓存：(op, backend, dtype, cc_major, cc_minor) -> fn 或 None
_CACHE = {}


class DispatchError(Exception):
    """分派失败（没有可用实现 / 能力不足）。"""


def _arch_tuple(arch):
    """把 `(7, 0)` / `"7.0"` / `70` 统一成 `(major, minor)`；None 表示不限。"""
    if arch is None:
        return (None, None)
    if isinstance(arch, str):
        parts = arch.replace("_", ".").split(".")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    if isinstance(arch, (tuple, list)):
        a = list(arch) + [0]
        return (int(a[0]), int(a[1]))
    v = int(arch)
    return (v // 10, v % 10)


def register_impl(op_name, backend, fn, *, caps=None, min_cc=None, max_cc=None,
                  dtype=None, prio=0):
    """为一个算子在某后端上登记一个实现。

    参数
    ----
    caps : dict
        该实现**要求**后端具备的能力（值可为 True，或 `(min, max)` 区间）。
        后端缺任何一条 ⇒ 该实现不参与分派。
    min_cc / max_cc : tuple|str|int|None
        允许的计算能力范围（如 `min_cc=(7, 0)` 表示 Volta 及以上）。
    dtype : str|None
        限定数据类型（"fp32"/"fp16"）。None = 通用。
    prio : int
        同一条件下多个候选时的优先级，**大的优先**。同优先级按登记顺序。
    """
    if not callable(fn):
        raise DispatchError("算子 %r 在 %r 上的实现不可调用" % (op_name, backend) +
                            "。修法：传一个函数（lambda 或 def）；"
                            "组合式算子用 plugin.op(name, fn) 即可")
    lo = _arch_tuple(min_cc)
    hi = _arch_tuple(max_cc)
    _IMPLS.setdefault((str(op_name), str(backend)), []).append({
        "fn": fn,
        "caps": dict(caps or {}),
        "min_cc": lo,
        "max_cc": hi,
        "dtype": None if dtype is None else str(dtype),
        "prio": int(prio),
    })
    _CACHE.clear()          # 登记变化后缓存失效
    return fn


def _caps_ok(need, have):
    """检查后端能力是否满足实现要求。缺一条就整体不满足。"""
    for k, want in need.items():
        if k not in have:
            return False
        got = have[k]
        if isinstance(want, bool):
            if bool(got) != want:
                return False
        elif isinstance(want, (tuple, list)) and len(want) == 2:
            try:
                if not (want[0] <= got <= want[1]):
                    return False
            except TypeError:
                return False
        else:
            if got != want:
                return False
    return True


def _cc_ok(arch, lo, hi):
    if lo[0] is None and hi[0] is None:
        return True
    maj, mino = arch
    if maj is None:
        return False                      # 实现要求 cc，但调用方没给
    cur = maj * 100 + (mino or 0)
    if lo[0] is not None and cur < lo[0] * 100 + (lo[1] or 0):
        return False
    if hi[0] is not None and cur > hi[0] * 100 + (hi[1] or 0):
        return False
    return True


def resolve(op_name, backend, *, caps=None, arch=None, dtype=None):
    """按（算子, 后端, 能力, 计算能力, dtype）选出实现；**单次查表 + 结果缓存**。

    返回可调用对象；没有可用实现则抛 `DispatchError`（列出候选与缺的能力）。
    """
    caps = caps or {}
    maj, mino = _arch_tuple(arch)
    # ★ 缓存键**必须包含 caps**：不同能力集会选出不同实现，漏掉会串味
    #   （实测踩过一次：先按 {"async_copy": True} 解析出快路径，再按 {} 查询时
    #   命中了同一条缓存，错误地返回了快路径）。
    #   caps 是后端属性、本进程内固定，且用户的要热路径会用 bind() 解析一次，
    #   所以这里做一次 sorted tuple 的代价可以接受。
    caps_key = tuple(sorted((k, v) for k, v in caps.items())) if caps else None
    key = (str(op_name), str(backend), None if dtype is None else str(dtype),
           maj, mino, caps_key)
    if key in _CACHE:
        fn = _CACHE[key]
        if fn is None:
            raise DispatchError(_explain(op_name, backend, caps, arch, dtype) +
                                "。修法：1) 放开调用条件（确认后端确实报告了该能力）；"
                                "2) 或给该后端补通用回退 plugin.op(name, fn, backend='*')")
        return fn

    # 合并「具体后端」与「通配后端（backend="*"，任何后端的通用回退）」两类候选。
    # 只在**解析时**合并一次，结果进缓存；缓存命中后仍是单次 dict 查表。
    # 注意：必须先把两类都取到再判空 —— 否则只登记了通配回退的后端会被误判为"无实现"。
    cands = list(_IMPLS.get((str(op_name), str(backend))) or [])
    cands += list(_IMPLS.get((str(op_name), "*")) or [])
    if not cands:
        _CACHE[key] = None
        raise DispatchError(_explain(op_name, backend, caps, arch, dtype) +
                            "。修法：1) 放开调用条件（确认后端确实报告了该能力）；"
                            "2) 或给该后端补通用回退 plugin.op(name, fn, backend='*')")

    best, best_prio = None, None
    for c in cands:                       # 单趟扫描，不做嵌套 fallback
        if c["dtype"] is not None and dtype is not None and c["dtype"] != str(dtype):
            continue
        if not _caps_ok(c["caps"], caps):
            continue
        if not _cc_ok((maj, mino), c["min_cc"], c["max_cc"]):
            continue
        if best is None or c["prio"] > best_prio:
            best, best_prio = c["fn"], c["prio"]
    _CACHE[key] = best
    if best is None:
        raise DispatchError(_explain(op_name, backend, caps, arch, dtype) +
                            "。修法：1) 放开调用条件（确认后端确实报告了该能力）；"
                            "2) 或给该后端补通用回退 plugin.op(name, fn, backend='*')")
    return best


def _explain(op_name, backend, caps, arch, dtype):
    cands = _IMPLS.get((str(op_name), str(backend)), [])
    if not cands:
        return ("算子 %r 在 %r 后端上没有登记任何实现" % (op_name, backend) +
                "。修法：plugin.op(...) 登记一个；或见 docs/PLUGIN_DISPATCH.md")
    miss = []
    for c in cands:
        gaps = [k for k in c["caps"] if k not in caps]
        if gaps:
            miss.append("缺能力 %s" % gaps)
        elif not _cc_ok(_arch_tuple(arch), c["min_cc"], c["max_cc"]):
            miss.append("计算能力不满足 %s..%s" % (c["min_cc"], c["max_cc"]))
        elif c["dtype"] is not None and str(c["dtype"]) != str(dtype):
            miss.append("只支持 dtype=%s" % c["dtype"])
        else:
            miss.append("其它条件不符")
    return ("算子 %r 在 %r 上没有可用实现（当前 caps=%s arch=%s dtype=%s）。%s"
            % (op_name, backend, sorted(caps), arch, dtype, "；".join(miss)))


def available(op_name, backend):
    """该算子在指定后端下登记了几个实现（供诊断）。"""
    return len(_IMPLS.get((str(op_name), str(backend)), []))


def clear():
    """清空登记与缓存（测试用）。"""
    _IMPLS.clear()
    _CACHE.clear()


def snapshot():
    """当前登记表的结构化快照（供 `T.plugins()` 与人看）。"""
    return {("%s@%s" % k): len(v) for k, v in sorted(_IMPLS.items())}


def benchmark_overhead(reps=200000):
    """实测分派开销：`resolve()` 命中缓存 vs 直接调用一个空函数。

    这是硬约束的量化手段 —— 若额外耗时占一次真实算子调用的比例 >1%，
    就应改成静态绑定（见 docs/PLUGIN_DISPATCH.md）。

    返回 dict：
      * `resolved_ns`  —— 一次缓存命中的 resolve + 调用
      * `direct_ns`    —— 一次直接调用
      * `overhead_ns`  —— 两者之差（分派本身的净开销）
    """
    register_impl("__bench__", "cpu", lambda x: x, prio=1)
    n = int(reps)

    t0 = time.perf_counter()
    for _ in range(n):
        resolve("__bench__", "cpu")(1)
    t1 = time.perf_counter()

    f = lambda x: x          # noqa: E731
    t2 = time.perf_counter()
    for _ in range(n):
        f(1)
    t3 = time.perf_counter()

    per_res = (t1 - t0) / n * 1e9
    per_dir = (t3 - t2) / n * 1e9
    return {"resolved_ns": per_res, "direct_ns": per_dir,
            "overhead_ns": per_res - per_dir}
