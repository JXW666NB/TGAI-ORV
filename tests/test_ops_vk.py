# -*- coding: utf-8 -*-
"""Vulkan 后端探测（纯 CPU，不依赖显卡）。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

from tgai_go.train import ops_vk  # noqa: E402


def test_available_is_bool():
    assert isinstance(ops_vk.available(), bool)


def test_devices_well_formed():
    for d in ops_vk.devices():
        assert set(d) == {"name", "type", "api"}
        assert d["type"] in ("other", "integrated-gpu", "discrete-gpu",
                             "virtual-gpu", "cpu")
        assert d["name"]


def test_missing_spirv_message():
    with pytest.raises(FileNotFoundError) as ei:
        ops_vk.load_spirv("no_such.spv")
    assert "修法" in str(ei.value)


def test_load_spirv_reports_clearly_when_not_implemented(tmp_path):
    p = tmp_path / "x.spv"
    p.write_bytes(b"\x03\x02#\x07")          # 假 SPIR-V magic
    try:
        ops_vk.load_spirv(str(p))
    except FileNotFoundError:
        pytest.fail("不该走到文件缺失分支")
    except NotImplementedError as exc:
        assert "dispatch" in str(exc) and "修法" in str(exc)
    except RuntimeError as exc:              # 本机没有 Vulkan 设备时
        assert "Vulkan" in str(exc)
