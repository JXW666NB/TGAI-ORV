# -*- coding: utf-8 -*-
"""`V27Trainer(param_dtype=...)` 的参数精度开关。

## 为什么要有

2.999B 全参训练在 32 GB 卡上的账（实测口径）：

    fp32 权重 + fp32 梯度 + fp32 Adam 状态 = 44.7 GiB  → 放不下
    fp32 权重 + fp32 梯度 + fp16 状态       = 33.5 GiB  → 仍放不下
    fp16 权重 + fp16 梯度 + fp16 状态       = 22.35 GiB → 进得去

所以训练器必须能把权重按 fp16 存放。这里锁住**开关本身**：
默认是 fp32（历史行为不变），显式传 fp16 才切。

真正的 fp16 数值正确性只能在 GPU 上验（CPU 路径不分配设备缓冲，`dtype` 不生效），
所以本文件只断言开关语义，不假装验证了数值。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train.v27_train import V27Trainer  # noqa: E402

CFG = dict(d_model=64, n_heads=4, d_ff=128, n_experts=2, n_activated=1,
           n_layers=1, vocab=64, max_seq=32, rope_theta=10000)


def test_default_is_fp32():
    tr = V27Trainer(CFG, n_layers=1)
    assert tr._pdt is np.float32, "默认必须是 fp32 —— 不能悄悄改掉老行为"


@pytest.mark.parametrize("name", ["fp16", "float16", "half", "FP16"])
def test_fp16_switch(name):
    """fp16 参数存储**当前不可用** —— 必须早失败并说清修法。

    2026-11-13 在 V100 上实测：参数存成 fp16 后，`T.cat(...)` 直接抛
    `CUDA cat 目前只支持 float32`。本引擎张量语义统一 float32，fp16 只在 GEMM 内部
    （`rt.fp16_gemm`）。所以这里断言"早失败 + 消息带修法"，而不是让它跑到深处才崩。
    """
    with pytest.raises(NotImplementedError) as ei:
        V27Trainer(CFG, n_layers=1, param_dtype=name)
    msg = str(ei.value)
    assert "float32" in msg and "修法" in msg


@pytest.mark.parametrize("name", ["fp32", "float32", "", None])
def test_other_values_stay_fp32(name):
    tr = V27Trainer(CFG, n_layers=1, param_dtype=name)
    assert tr._pdt is np.float32, "非 fp16 一律回退 fp32（含空值/None）"
