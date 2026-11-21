# -*- coding: utf-8 -*-
"""`TG_DEBUG` 必须**真的**改变错误输出（2026-11-21）。

背景：全仓 192 处错误文案写着「加 TG_DEBUG=1 可出更多日志」，但在此之前
**没有任何代码读这个变量** —— 提示在教用户干一件没用的事。这个文件钉住它现在是真的。
"""
import os

import pytest

from tgai_go.train.cuda import CudaError


def test_debug_off_by_default(monkeypatch):
    monkeypatch.delenv("TG_DEBUG", raising=False)
    msg = str(CudaError("boom"))
    assert msg == "boom"
    assert "[TG_DEBUG]" not in msg


@pytest.mark.parametrize("val", ["1", "true", "YES", "on"])
def test_debug_on_appends_context(monkeypatch, val):
    monkeypatch.setenv("TG_DEBUG", val)
    msg = str(CudaError("boom"))
    assert msg.startswith("boom")            # 原始消息必须保留在最前面
    assert "[TG_DEBUG]" in msg
    assert "platform=" in msg and "pid=" in msg


@pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
def test_debug_off_values(monkeypatch, val):
    monkeypatch.setenv("TG_DEBUG", val)
    assert "[TG_DEBUG]" not in str(CudaError("boom"))


def test_debug_never_masks_original_error(monkeypatch):
    """诊断上下文本身出错时，也不能吞掉原始错误。"""
    monkeypatch.setenv("TG_DEBUG", "1")
    import tgai_go.train.cuda as C

    def boom():
        raise RuntimeError("context failed")

    monkeypatch.setattr(C, "_debug_context", boom)
    msg = str(CudaError("original"))
    assert msg == "original"


def test_message_still_carries_fix_hint(monkeypatch):
    """仓库规则：错误消息必须带「修法：」，加诊断不能把它挤掉。"""
    monkeypatch.setenv("TG_DEBUG", "1")
    msg = str(CudaError("scalar mismatch。修法：改形状"))
    assert "修法：" in msg
    assert "[TG_DEBUG]" in msg
