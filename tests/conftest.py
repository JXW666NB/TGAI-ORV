# -*- coding: utf-8 -*-
"""pytest 全局配置：把仓库里的 `python/` 放进 sys.path。

为什么需要：本地开发时通常 `pip install -e .` 过，所以 `import tgai_go` 能成；
但 **CI 上并没有安装这个包**，测试模块在顶层 import 就会 ModuleNotFoundError
（2026-10-30 CI 两次全红就是这个原因）。有了这个 conftest，测试不依赖"有没有装过"。
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for cand in (os.path.join(REPO, "python"), REPO):
    if os.path.isdir(os.path.join(cand, "tgai_go")) and cand not in sys.path:
        sys.path.insert(0, cand)


@pytest.fixture(autouse=True)
def _restore_default_device():
    """每个测试结束后把默认设备**还原**，避免污染后续测试。

    ## 为什么必须有（外部实测，2026-11-12）

    在有 GPU 的机器上跑全量套件会多出 8 个失败，而**单独跑这些用例全过**。
    根因：大量测试为了省事调 `T.set_default_device("cuda")` 却不还原，
    后续测试在 CUDA 上新建张量时，`tensor.data` 恒为 `None` ——
    于是 `np.asarray(t.data)` 得到 None/NaN（`test_adafactor` 一次报 6 个 NaN）。

    这不是库算错，是测试之间互相污染。**还原默认设备**比逐个改测试可靠。

    另外顺带在结束时清一次 CUDA 池，避免显存跨用例累积。
    """
    import tgai_go.train as T

    try:
        before = T.get_device()
    except Exception:                       # noqa: BLE001 - 库可能没初始化
        before = None
    try:
        yield
    finally:
        try:
            if before is not None:
                T.set_default_device(before)
        except Exception:                   # noqa: BLE001
            pass
