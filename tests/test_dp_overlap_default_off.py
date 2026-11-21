# -*- coding: utf-8 -*-
"""「反向即归约」默认关闭的行为守卫。

2026-11-21 实测（2×4090, `--dp 2`, fp32, host 后端, 4 步, gnorm 9 位小数）：

| 配置 | 普通 all_reduce 调用次数 | `[dp-ov] comm` | 两 rank gnorm |
| --- | --- | --- | --- |
| `TG_DP_OVERLAP=0` | 每步 1 次 | 无 | **逐位相同** |
| 默认（重叠开）| **0 次** | **0.00 ms** | **相差 1e-2** |

⇒ 重叠路径静默不归约 ⇒ 数据并行退化成「没开 DP」。所以它必须是**显式 opt-in**。
这条测试就是钉住"默认关"这个决定，防止以后有人"顺手打开"。
"""
import os

import pytest

from tgai_go.train import v3_pretrain as V


@pytest.mark.parametrize("val,expect", [
    (None, False),        # 未设置 ⇒ 关
    ("", False),
    ("0", False),
    ("false", False),
    ("no", False),
    ("off", False),
    ("1", True),          # 只有显式要求才开
    ("true", True),
    ("YES", True),
])
def test_overlap_is_opt_in(monkeypatch, val, expect):
    if val is None:
        monkeypatch.delenv("TG_DP_OVERLAP", raising=False)
    else:
        monkeypatch.setenv("TG_DP_OVERLAP", val)
    assert V._overlap_enabled() is expect


def test_overlap_default_is_off(monkeypatch):
    """没有环境变量时必须关 —— 这是本模块唯一的安全默认值。"""
    monkeypatch.delenv("TG_DP_OVERLAP", raising=False)
    assert V._overlap_enabled() is False
