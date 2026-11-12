# -*- coding: utf-8 -*-
"""v3 ↔ v4 版本号改写 + `read_tg` 解析的回归。

## 背景（2026-11-12 实测）

V2.7 的 `TGAI_NB_V2.7_v4.tg` 与 ORV 训练检查点 `weights.tg` 都是
`magic=0x49414754`（TGAI）+ `version=4`，**布局与 v3 完全相同**（定长 128B 头 + FNV 权重表）。
所以"v4 支持"在引擎侧只需放行版本号，不需要新的读取路径。

这里用纯 Python 锁住这个事实：把 v3 夹具改成 version=4 后，
`read_tg()` 仍要能解析出同样的 layers/d_model —— 一旦有人真的把 v4 改成别种布局，
这条会红，提醒他引擎侧也要跟着改。

（引擎能否真的加载 v4，由 `tools/probe_infer.py` 在真卡上验证，
  不在 pytest 里跑：需要编译好的 DLL 与 CUDA。）
"""
import os
import struct
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
FIXTURE = r"D:\tgai_nlp\TGAI GO\tests\fixtures\mini_stream.TG"


def _have_fixture():
    return os.path.isfile(FIXTURE)


@pytest.mark.skipif(not _have_fixture(), reason="需要 TGAI GO 的 v3 测试夹具")
def test_version_bump_to_v4_keeps_layout_parseable(tmp_path):
    from tgai_go.read_tg import read_tg

    src = FIXTURE
    dst = str(tmp_path / "mini_v4.TG")
    r = subprocess.run([sys.executable, "-X", "utf8",
                        os.path.join(TOOLS, "set_tg_version.py"),
                        src, dst, "--version", "4"],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout + r.stderr

    base = read_tg(src)["header"]
    bumped = read_tg(dst)["header"]
    assert base["version"] == 3
    assert bumped["version"] == 4
    # 除了版本号，头里的架构字段必须一模一样
    for k in ("n_layers", "d_model", "n_heads", "d_ff", "vocab", "max_seq"):
        assert base[k] == bumped[k], "改版本号不应动到 %s" % k


@pytest.mark.skipif(not _have_fixture(), reason="需要 TGAI GO 的 v3 测试夹具")
def test_set_tg_version_rejects_tg4_container(tmp_path):
    """magic 是 TG4（0x00344754）的文件必须被拒绝 —— 那是另一套格式。"""
    fake = tmp_path / "fake_tg4.tg"
    buf = bytearray(128)
    struct.pack_into("<II", buf, 0, 0x00344754, 4)
    fake.write_bytes(bytes(buf))
    r = subprocess.run([sys.executable, "-X", "utf8",
                        os.path.join(TOOLS, "set_tg_version.py"),
                        str(fake), "--version", "3"],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    assert r.returncode != 0
    assert "TG4" in (r.stdout + r.stderr)
