# -*- coding: utf-8 -*-
"""pytest 全局配置：把仓库里的 `python/` 放进 sys.path。

为什么需要：本地开发时通常 `pip install -e .` 过，所以 `import tgai_go` 能成；
但 **CI 上并没有安装这个包**，测试模块在顶层 import 就会 ModuleNotFoundError
（2026-10-30 CI 两次全红就是这个原因）。有了这个 conftest，测试不依赖"有没有装过"。
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for cand in (os.path.join(REPO, "python"), REPO):
    if os.path.isdir(os.path.join(cand, "tgai_go")) and cand not in sys.path:
        sys.path.insert(0, cand)
