# -*- coding: utf-8 -*-
"""策略 `attn` 键必须真的生效（2026-11-21 用户反馈后修的对外承诺问题）。

之前：`_POLICY_PRESETS` 里写着 `attn="memory"/"materialized"`，但**没有任何代码读这个键**，
所以 `T.set_policy("memory")` 并不会切到 flash 注意力 —— 用户以为省了显存，其实只改了
`attn_tile`。同理 `checkpoint` / `inplace_acc` 也是死键（本文件也把这一点钉住，
免得以后有人再往这两个键上寄托期望）。

现在：调用点显式 `lowmem=` 优先；否则看策略；都没有则 auto。
"""
import pytest

import tgai_go.train as T

F = T.functional


@pytest.fixture(autouse=True)
def _restore_policy():
    # `get_policy()` 返回的是 (名字, 策略字典) 二元组，不是字典本身
    before = dict(T.get_policy()[1])
    yield
    for k, v in before.items():
        T.set_policy_value(k, v)


def test_standard_and_speed_stay_auto():
    """默认策略必须是 auto —— 否则所有老用户会被强制物化注意力（性能回归）。"""
    assert T.set_policy("standard")["attn"] == "auto"
    assert T.set_policy("speed")["attn"] == "auto"
    T.set_policy("standard")
    assert F._resolve_lowmem(None) is None, "auto ⇒ None（交给底层自动选择）"


def test_memory_policy_really_means_flash():
    pol = T.set_policy("memory")
    assert pol["attn"] == "flash"
    assert F._resolve_lowmem(None) is True, "memory 策略必须真的解析成 lowmem=True"


def test_extreme_policy_really_means_flash():
    T.set_policy("extreme")
    assert F._resolve_lowmem(None) is True


def test_explicit_argument_wins_over_policy():
    T.set_policy("memory")                       # 策略说 flash
    assert F._resolve_lowmem(False) is False, "调用点显式 False 必须胜过策略"
    T.set_policy("standard")                     # 策略说 auto
    assert F._resolve_lowmem(True) is True


def test_materialized_policy_forces_off():
    T.set_policy_value("attn", "materialized")
    assert F._resolve_lowmem(None) is False


def test_policy_keys_that_do_nothing_are_documented():
    """`checkpoint` / `inplace_acc` 目前没有任何消费者 —— 钉住事实，避免再被误解。"""
    pol = T.set_policy("standard")
    assert "checkpoint" in pol and "inplace_acc" in pol
    # 真正被读取的键只有这三个（见 v3_flash_op.py / v3_ops.py 的 policy() 调用）
    for used in ("cache_masks", "attn_tile", "fused_rope"):
        assert used in pol, "%s 是真实生效的键，不能被删" % used
