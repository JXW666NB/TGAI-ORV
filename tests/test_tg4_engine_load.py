# -*- coding: utf-8 -*-
"""C 引擎能否挂接 **TG4** 容器（magic `TG4\0` + section table + NAM）。

## 为什么要这条

TG4 与 v3 家族的头偏移不同、且没有 FNV 权重表。引擎侧新增的 `tg4_attach()` 负责把
NAM 翻译成引擎内部的 `TGWeightEntry`（名字 → FNV-1a）。这条用例锁住"挂得上"这件事：

* 造一个**最小** TG4（两个张量，够它挂接就行）；
* 用仓库自带的引擎库 `load()` 它；
* 期望**挂接成功**（能拿到 Model 句柄）。

注意：这个小文件缺少真正的 FFN/MoE 权重，所以 `block_cache` 会打"缺权重"警告 ——
那是**预期的**，本条只验证容器解析路径，不验证数值。

只在 Windows + 仓库内存在 `_native/libtgai_engine.dll` 时运行：
其它平台上这个 DLL 是 Linux/其它构建产物，不能拿来做这个断言。
"""
import os
import sys
import tempfile

import numpy as np
import pytest

WIN = sys.platform == "win32"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DLL = os.path.join(REPO, "python", "tgai_go", "_native", "libtgai_engine.dll")


@pytest.mark.skipif(not WIN, reason="引擎库是 Windows DLL；其它平台需对应构建产物")
@pytest.mark.skipif(not os.path.isfile(DLL), reason="仓库里没有 libtgai_engine.dll")
def test_engine_attaches_a_minimal_tg4():
    sys.path.insert(0, os.path.join(REPO, "python"))
    from tgai_go import tg4
    import tgai_go

    cfg = dict(n_layers=1, d_model=16, n_heads=2, head_dim=8, d_ff=32, vocab=32,
               max_seq=16, n_kv_heads=0, n_experts=0, n_activated=0, arch=0,
               weight_dtype=tg4.DT_BF16)
    items = [("token_embedding.weight", np.zeros((32, 16), np.float32)),
             ("final_norm.weight", np.ones((16,), np.float32))]
    p = os.path.join(tempfile.gettempdir(), "engine_attach_test.tg4")
    tg4.write_tg4_stream(p, iter(items), cfg)
    try:
        m = tgai_go.load(p)
        assert m.n_layers == 1 and m.d_model == 16 and m.vocab_size == 32
    finally:
        try:
            os.remove(p)
        except OSError:
            pass
