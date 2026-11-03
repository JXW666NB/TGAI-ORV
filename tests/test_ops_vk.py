# -*- coding: utf-8 -*-
"""Vulkan 探测（全部走子进程）。

**为什么都用子进程**：AMD 驱动在同一进程里加载/卸载 vulkan-1.dll 或反复建销 instance，
会让 pytest 进程在退出时段错误（实测）。子进程化后，驱动再怎么崩也只影响这一条。
"""
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import ops_vk as OPS_VK  # noqa: E402


def _child(code):
    full = "import sys, json; sys.path.insert(0, %r)\n%s" % (os.path.join(REPO, "python"), code)
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", full],
                       capture_output=True, text=True,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    return r


def test_available_is_bool_in_subprocess():
    r = _child("from tgai_go.train import ops_vk;"
               "print('V', isinstance(ops_vk.available(), bool))")
    assert r.returncode == 0, r.stderr[-600:]
    assert r.stdout.strip().splitlines()[-1] == "V True"


def test_devices_well_formed_in_subprocess():
    r = _child("from tgai_go.train import ops_vk; print(json.dumps(ops_vk.devices()))")
    assert r.returncode == 0, r.stderr[-600:]
    devs = json.loads(r.stdout.strip().splitlines()[-1])
    for d in devs:
        assert set(d) == {"name", "type", "api"}
        assert d["type"] in ("other", "integrated-gpu", "discrete-gpu",
                             "virtual-gpu", "cpu")
        assert d["name"]


def test_missing_spirv_message():
    with pytest.raises(FileNotFoundError) as ei:
        OPS_VK.load_spirv("no_such.spv")
    assert "修法" in str(ei.value)


def test_legacy_load_spirv_points_to_new_api(tmp_path):
    """早期骨架要明确把人指向 ops_vk_ops，而不是静默失败。"""
    p = tmp_path / "x.spv"
    p.write_bytes(b"\x03\x02\x23\x07")
    with pytest.raises(NotImplementedError) as ei:
        OPS_VK.load_spirv(str(p))
    msg = str(ei.value)
    assert "ops_vk_ops" in msg and "修法" in msg
