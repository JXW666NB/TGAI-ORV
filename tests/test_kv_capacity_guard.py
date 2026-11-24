# -*- coding: utf-8 -*-
"""KV 容量守卫回归测试 —— 缺陷 B（CPU 前向按 cache_pos+t 无边界写 KV）。

背景：
  `TGEngine.new_kv_slot()` 旧实现把 CPU 侧容量压到 max_seq=1（注释称
  "CUDA 路径不用这份数组"），但 **CPU 前向恰恰用这份数组**，并按
  `cache_pos + t` 无边界写 ⇒ 纯 CPU / Vulkan / CUDA 回落路径会跨层污染
  + 堆越界（静默数据损坏）。
修复分两层：
  1) Python：new_kv_slot 按"CUDA 是否接管"选容量；_session.generate 每次
     前向前检查 cache_pos+seq_len>capacity 并抛 RuntimeError；
  2) C 源码：tg_forward 入口同样的边界检查 + tg_kvcache_new 参数守卫
     （本机无编译器，_native/*.dll 还是旧二进制 ⇒ C 守卫需重建后生效，
       Python 守卫现在就已生效，所以下面的行为测试不依赖重建）。

覆盖：
  · 容量选择（显式 / 环境变量 / 模型上限 / CUDA 直通 / Vulkan 不算接管）
  · 纯 Python 守卫函数（不依赖原生库）
  · 有原生库时用 mini_stream.TG（max_seq=128）+ max_seq=4 的小槽真跑：
    第 5 个位置必须抛 RuntimeError，而不是崩溃/乱写
  · forward_logits 返回独立拷贝（缺陷 C 的前提）
  · CPU 前向执行期间 _cpu_fwd_lock 必须被持有（缺陷 C 的锁范围）
"""
import ctypes
import os
import types

import numpy as np
import pytest

from tgai_go._engine import TGEngine
from tgai_go._session import (_check_kv_capacity, _kv_capacity_of,
                              generate)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NATIVE_CANDIDATES = (
    os.path.join(REPO, 'python', 'tgai_go', '_native', 'libtgai_engine.dll'),
    os.path.join(REPO, 'python', 'tgai_go', '_native', 'libtgai_engine.so'),
    os.path.join(REPO, 'cpu', 'build', 'libtgai_engine.dll'),
    os.path.join(REPO, 'cpu', 'build', 'libtgai_engine.so'),
)
MODEL = os.path.join(REPO, 'tests', 'fixtures', 'mini_stream.TG')


# ══════════════════════════════════════════════════════════════════
#  1) 纯 Python：守卫函数
# ══════════════════════════════════════════════════════════════════
class _FakeEngine:
    def __init__(self, cuda=False, max_seq=8192):
        self.max_seq = max_seq
        self._cuda = cuda

    def cuda_forward_active(self):
        return self._cuda


def test_check_kv_capacity_boundaries():
    # None = 不做守卫（CUDA 前向接管）
    _check_kv_capacity(None, 10 ** 9, 10 ** 9)
    # 恰好填满允许
    _check_kv_capacity(4, 0, 4)
    _check_kv_capacity(4, 3, 1)
    # 超 1 个位置就抛（消息必须带 KV overflow 与三个数值）
    with pytest.raises(RuntimeError) as ei:
        _check_kv_capacity(4, 1, 4)
    msg = str(ei.value)
    assert '[KV overflow]' in msg
    assert 'cache_pos=1' in msg and 'seq_len=4' in msg and 'capacity=4' in msg
    with pytest.raises(RuntimeError):
        _check_kv_capacity(4, 4, 1)


def test_kv_capacity_of_prefers_slot_capacity():
    eng = _FakeEngine(cuda=False, max_seq=8192)
    kv = types.SimpleNamespace(capacity=512)
    assert _kv_capacity_of(eng, kv) == 512
    # kv 没有 capacity 字段 ⇒ 回落到 engine.max_seq
    assert _kv_capacity_of(eng, types.SimpleNamespace()) == 8192
    # kv=None（generate 自建 cache）⇒ engine.max_seq
    assert _kv_capacity_of(eng, None) == 8192


def test_kv_capacity_of_skips_cuda_takeover():
    # CUDA 前向按槽自己维护 GPU KV，不读 host 数组 ⇒ 不做 Python 守卫
    eng = _FakeEngine(cuda=True, max_seq=8192)
    assert _kv_capacity_of(eng, types.SimpleNamespace(capacity=1)) is None


# ══════════════════════════════════════════════════════════════════
#  2) 纯 Python：new_kv_slot 的容量选择（fake lib，不依赖原生库）
# ══════════════════════════════════════════════════════════════════
class _FakeLib:
    def __init__(self):
        self.created = []

    def tg_kvcache_new(self, model, seq):
        self.created.append(seq)
        return 0x1000 + seq


def _make_engine(cuda=False, backend='cpu', max_seq=8192, ready=False):
    eng = object.__new__(TGEngine)          # 跳过 __init__（不需要真 DLL）
    eng.lib = _FakeLib()
    eng.model = 0x1
    eng.max_seq = max_seq
    eng._slots = set()
    eng.gpu_info = {'ready': ready, 'backend': backend, 'reason': ''}
    eng.kv_cache_capacity = None
    return eng


def test_new_kv_slot_explicit_max_seq():
    eng = _make_engine()
    h = eng.new_kv_slot(max_seq=7)
    assert h.capacity == 7
    assert eng.lib.created == [7]
    with pytest.raises(ValueError):
        eng.new_kv_slot(max_seq=0)
    with pytest.raises(ValueError):
        eng.new_kv_slot(max_seq=-3)


def test_new_kv_slot_cpu_default_and_env(monkeypatch):
    monkeypatch.delenv('TGAI_KV_CPU_SLOT_SEQ', raising=False)
    eng = _make_engine(max_seq=8192)
    assert eng.new_kv_slot().capacity == 2048       # 默认 2048
    monkeypatch.setenv('TGAI_KV_CPU_SLOT_SEQ', '512')
    assert eng.new_kv_slot().capacity == 512
    # 非法值/0 回落到默认值（不能被 0 带偏）
    monkeypatch.setenv('TGAI_KV_CPU_SLOT_SEQ', 'not-a-number')
    assert eng.new_kv_slot().capacity == 2048
    monkeypatch.setenv('TGAI_KV_CPU_SLOT_SEQ', '0')
    assert eng.new_kv_slot().capacity == 1


def test_new_kv_slot_clamped_by_model_max_seq(monkeypatch):
    monkeypatch.setenv('TGAI_KV_CPU_SLOT_SEQ', '8192')
    eng = _make_engine(max_seq=256)
    assert eng.new_kv_slot().capacity == 256


def test_new_kv_slot_cuda_keeps_one(monkeypatch):
    # CUDA 接管 ⇒ host 容量 1（省内存），capacity 如实记 1
    monkeypatch.setenv('TGAI_KV_CPU_SLOT_SEQ', '2048')
    eng = _make_engine(cuda=True, backend='cuda', ready=True)
    h = eng.new_kv_slot()
    assert h.capacity == 1
    assert eng.lib.created == [1]


def test_new_kv_slot_vulkan_is_not_cuda():
    # Vulkan 是逐算子加速，tg_forward 仍走 CPU 路径（host KV 数组在用）
    eng = _make_engine(cuda=False, backend='vulkan', ready=True)
    assert eng.new_kv_slot().capacity == 2048


# ══════════════════════════════════════════════════════════════════
#  3) 有原生库 + mini 模型：真跑，验证"抛错而不是越界写"
# ══════════════════════════════════════════════════════════════════
def _native_dll():
    for p in _NATIVE_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


@pytest.fixture(scope='module')
def engine(tmp_path_factory):
    dll = _native_dll()
    if dll is None or not os.path.isfile(MODEL):
        pytest.skip('没有原生引擎库或 mini_stream.TG，跳过原生行为测试')
    # ★ 仓库路径可能含非 ASCII 字符（本机 D:\...\TGAI-对话档案\...），而引擎
    #   C 层用 ANSI fopen 打开模型 ⇒ 中文绝对路径会报"无法打开文件"。
    #   测试把 fixture 复制到 ASCII 临时目录再加载（不改库本身的路径行为）。
    import shutil
    tmp = tmp_path_factory.mktemp('tgai_kv')
    model_copy = os.path.join(str(tmp), 'mini_stream.TG')
    shutil.copyfile(MODEL, model_copy)
    e = TGEngine(dll_path=dll)
    e.load(model_copy)
    yield e
    try:
        e.free()
    except Exception:                                             # noqa: BLE001
        pass


def _gen_params(max_tokens=10):
    return {'temperature': 0.0, 'top_k': 0, 'top_p': 1.0,
            'max_tokens': max_tokens, 'min_tokens': 1}


def test_native_small_slot_raises_instead_of_overflowing(engine):
    """max_seq=4 的槽跑第 5 个位置 ⇒ RuntimeError（旧行为：越界/损坏）。"""
    kv = engine.new_kv_slot(max_seq=4)
    assert kv.capacity == 4
    try:
        ids = engine.encode('abc', add_bos=True)[:3]
        assert len(ids) == 3
        got = []
        with pytest.raises(RuntimeError) as ei:
            for tok in generate(engine, ids, _gen_params(10),
                                eos_id=99999999, eos_set={99999999}, kv=kv):
                got.append(tok)
        msg = str(ei.value)
        assert '[KV overflow]' in msg
        assert 'capacity=4' in msg
        # 越界发生在 prefill(3) 之后写第 5 个位置（cur=4）时
        assert 'cache_pos=4' in msg and 'seq_len=1' in msg
    finally:
        kv.release()


def test_native_prompt_longer_than_slot_raises_at_prefill(engine):
    kv = engine.new_kv_slot(max_seq=4)
    try:
        ids = engine.encode('abcdef', add_bos=True)
        if len(ids) < 5:
            ids = [2, 3, 4, 5, 6]
        with pytest.raises(RuntimeError) as ei:
            for _ in generate(engine, ids[:5], _gen_params(3),
                              eos_id=99999999, eos_set={99999999}, kv=kv):
                pass
        assert 'cache_pos=0' in str(ei.value)
    finally:
        kv.release()


def test_native_own_cache_capacity_is_model_max_seq(engine):
    """kv=None（generate 自建 cache）⇒ capacity = 创建时用的 max_seq。

    用超过 max_seq 的 prompt 触发 prefill 守卫（不会真的分配超大内存：
    守卫在 tg_forward 之前就抛了）。
    """
    base = engine.encode('hello', add_bos=True) or [2, 3, 4]
    long_ids = (base * (engine.max_seq // max(1, len(base)) + 3))
    assert len(long_ids) > engine.max_seq
    with pytest.raises(RuntimeError) as ei:
        for _ in generate(engine, long_ids, _gen_params(1),
                          eos_id=99999999, eos_set={99999999}, kv=None):
            pass
    assert 'capacity=%d' % engine.max_seq in str(ei.value)


def test_native_short_generation_ok_and_state_reports_length(engine):
    """守卫不能误伤正常路径；max_tokens 用完时 state 要报 length。"""
    kv = engine.new_kv_slot(max_seq=16)
    state = {}
    try:
        ids = engine.encode('abc', add_bos=True)
        got = list(generate(engine, ids, _gen_params(3),
                            eos_id=99999999, eos_set={99999999},
                            kv=kv, state=state))
        assert len(got) == 3
        assert state.get('truncated') is True
        assert state.get('finish_reason') == 'length'
    finally:
        kv.release()


def test_native_forward_logits_returns_independent_copy(engine):
    """缺陷 C 的前提：每次前向返回独立拷贝，不被下一次前向覆盖。"""
    ids = engine.encode('abc', add_bos=True)
    a1 = engine.forward_logits((ctypes.c_int32 * len(ids))(*ids),
                               len(ids), None, 0)
    assert a1 is not None and a1.shape == (engine.vocab_size,)
    snap = a1.copy()
    a2 = engine.forward_logits((ctypes.c_int32 * 2)(2, 3), 2, None, 0)
    assert a2 is not None and a2 is not a1
    assert np.array_equal(a1, snap), '第一次的 logits 被第二次前向覆盖了'


def test_native_cpu_forward_holds_python_lock(engine):
    """缺陷 C：CPU 前向执行期间必须持有 _cpu_fwd_lock（否则 copy 可能读到脏缓冲）。"""
    if engine.cuda_forward_active():
        pytest.skip('CUDA 前向按槽返回 logits，按设计不加这把锁')
    ids = engine.encode('abc', add_bos=True)
    holder = {
        'lock_free_at_entry': [],
        'orig': engine.lib.tg_forward,
    }

    def probe(*args):
        # 锁若被 forward_logits 持有 ⇒ 这里 acquire(blocking=False) 必然失败
        acquired = engine._cpu_fwd_lock.acquire(blocking=False)
        if acquired:
            engine._cpu_fwd_lock.release()
        holder['lock_free_at_entry'].append(acquired)
        return holder['orig'](*args)

    engine.lib.tg_forward = probe
    try:
        arr = engine.forward_logits((ctypes.c_int32 * len(ids))(*ids),
                                    len(ids), None, 0)
        assert arr is not None
    finally:
        del engine.lib.tg_forward        # 恢复 CDLL 的符号查找

    assert holder['lock_free_at_entry'] == [False], (
        'tg_forward 执行时 _cpu_fwd_lock 没有被持有（缺陷 C 会复现）')
