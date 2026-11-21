# -*- coding: utf-8 -*-
"""fp16 权重影子缓存的两条不变量 + 两条性能前提（2026-11-21）。

背景（5090 / seq512 / d=1024 / 12L / E=4 / top-k 2 / int8gpu 实测）：

* 影子缓存在 `free()` 里按裸指针失效，原实现是 **线性扫描整个缓存**
  （`[k for k in cache if int(k[0]) == p]`）。缓存里是本配置的 193 条权重影子，
  而 `free()` 每步要调 4093 次 ⇒ **29.6 ms/步**，占 `--amp gemm` 负优化
  （+42% → 修的正是这一项）的 71%。fp32 路径因为缓存为空只需 0.93 ms/步。
* 影子缓冲每步被 `f16_cache_clear()` 还给池子、下一步再 `empty()` 一遍
  （193 次 empty + 193 次 free/步）。我试过改成跨步原地复用（`_f16_spare`），
  交替 A/B 3 轮下来 **没有收益**（0.1436 vs 0.1429，落在噪声内），而且要多占
  1.4 GiB 峰值，所以**没有采用** —— 这条留在注释里，免得以后有人再走一遍。

所以本文件守三件事：

1. **正确性（根因不变量）**：`f16_cache_clear()` 之后，同一个 (ptr, count) 必须
   **重新 cast**，绝不能把上一步的 fp16 内容当本步的用。这是历史上 4130 倍梯度
   爆炸（gnorm 3.3e4 vs 8.0）那一类事故的直接防线。
2. **正确性（地址复用）**：`_f16_cache_drop_ptr(ptr)` 之后该地址必须彻底失效 ——
   池子把地址借给新张量时不能读到旧影子。
3. **性能前提（O(1) 失效）**：源码里不允许再出现线性扫描；必须有反向索引。
"""
import io
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA_PY = os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py")
sys.path.insert(0, os.path.join(ROOT, "python"))

CUDA_ON = bool(os.environ.get("TG_RUN_CUDA_TESTS"))


def _src():
    return io.open(CUDA_PY, encoding="utf-8").read()


# ---------------------------------------------------------------- 3. 源码前提
def _body(name, span=2400):
    """取某个方法**去掉 docstring 之后**的源码片段（注释里提到旧写法不算数）。"""
    src = _src()
    i = src.index("def %s" % name)
    seg = src[i:i + span]
    a = seg.find('"""')
    if a >= 0:
        b = seg.find('"""', a + 3)
        if b > a:
            seg = seg[:a] + seg[b + 3:]
    return seg


def test_drop_ptr_has_no_linear_scan():
    """失效路径不许再扫全表（每步 4093 次 × 193 条 ⇒ 29.6 ms）。"""
    body = _body("_f16_cache_drop_ptr")
    assert "[k for k in cache" not in body, (
        "`_f16_cache_drop_ptr` 又变回线性扫描了：本配置实测 29.6 ms/步。"
        "修法：用 `_f16_index`（ptr -> key 集合）做 O(1) 失效")
    assert "_f16_index" in body, (
        "`_f16_cache_drop_ptr` 必须走 `_f16_index` 反向索引")


def test_cache_clear_also_drops_the_index():
    """`f16_cache_clear` 必须把反向索引一起清掉（否则失效路径指向不存在的条目）。"""
    body = _body("f16_cache_clear", 1200)
    assert "self._f16_cache = {}" in body, (
        "`f16_cache_clear` 必须真的清空 `_f16_cache`（否则权重更新后仍命中旧影子）")
    assert "self._f16_index = {}" in body, (
        "`f16_cache_clear` 必须同时清 `_f16_index`")


# ------------------------------------------------------- 1./2. 行为（需要 GPU）
@pytest.mark.skipif(not CUDA_ON, reason="需要 GPU：设 TG_RUN_CUDA_TESTS=1")
def test_shadow_is_recast_after_clear():
    import ctypes

    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    n = 4096
    a = rt.h2d(np.full((1, n), 1.0, np.float32))
    rt.f16_cache_pin([a.ptr])
    rt.f16_cache_enable(True)
    try:
        b1 = rt._cast_f16_cached(a.ptr, n)
        assert b1 is not None
        # 一步之内重复使用 ⇒ 命中同一块影子（这就是缓存存在的意义）
        assert rt._cast_f16_cached(a.ptr, n) is b1
        got1 = np.asarray(rt.d2h(rt.cast_f16_to_f32(b1)), np.float64).reshape(-1)
        assert np.all(got1 == 1.0)

        # 模拟"优化器更新了权重"：原地改内容 + 让影子失效
        host = np.full(n, 2.0, np.float32)
        rt.h2d_into(a, ctypes.c_void_p(host.ctypes.data), n * 4)
        rt.f16_cache_clear()
        b2 = rt._cast_f16_cached(a.ptr, n)
        got2 = np.asarray(rt.d2h(rt.cast_f16_to_f32(b2)), np.float64).reshape(-1)
        assert np.all(got2 == 2.0), (
            "clear() 之后影子没重新 cast：读到上一步的值 %r（这正是 4130 倍梯度爆炸"
            "那类事故）。修法：`f16_cache_clear` 必须清空 `_f16_cache`"
            % (got2[:4],))
    finally:
        rt.f16_cache_pin([])
        rt.f16_cache_enable(False)


@pytest.mark.skipif(not CUDA_ON, reason="需要 GPU：设 TG_RUN_CUDA_TESTS=1")
def test_drop_ptr_invalidates_recycled_shadow():
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    n = 4096
    a = rt.h2d(np.full((1, n), 3.0, np.float32))
    rt.f16_cache_pin([a.ptr])
    rt.f16_cache_enable(True)
    try:
        b1 = rt._cast_f16_cached(a.ptr, n)
        rt.f16_cache_clear()             # 缓冲进了复用表
        rt._f16_cache_drop_ptr(a.ptr)    # 地址被归还池子
        b2 = rt._cast_f16_cached(a.ptr, n)
        assert b2 is not b1, (
            "地址被 drop 之后必须换一块影子（否则池子把地址借给新张量时会读到旧内容）")
    finally:
        rt.f16_cache_pin([])
        rt.f16_cache_enable(False)
