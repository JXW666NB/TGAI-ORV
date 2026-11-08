# -*- coding: utf-8 -*-
"""MoE 在前向 token 数 > 小批量时越界（**复现 ORV 的真 bug**）。

实测（V100，真实 V2.7 第 0 层 MoE 权重，8 专家 top-2）：
  * n=8    → 正常，输出有限
  * n=128  → `cuLaunchKernel(ewise_mul_v4)` **rc=700 illegal memory access**
同一份权重、同一个 gate，只改 token 数就崩 ⇒ 不是权重问题，是分组/路由缓冲按小 n 假设分配。

影响：V2.7 全参训练每步要处理 T=512~2048 个 token ⇒ **此 bug 直接阻塞 P3**。
修好之前，不要在生产训练里跑 n>8 的 MoE 前向。

CI 无 GPU 时跳过（本条只在有卡的机器上有意义）。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))


def _has_cuda():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:                      # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="需要 CUDA：MoE 的 CUDA 路径才有这个问题")


@pytest.mark.xfail(strict=False, reason="ORV MoE 在 n>=64 时 ewise_mul_v4 越界（待修）；修好后去掉此标记")
def test_moe_large_batch_no_ime():
    import tgai_go.train as T
    from tgai_go.train import MoE

    T.set_default_device("cuda")
    D, FF, E, K = 1024, 4096, 8, 2
    rng = np.random.default_rng(0)
    m = MoE(D, FF, E, top_k=K, gate="softmax_topk", qb_bias=rng.normal(0, .1, E).astype(np.float32))
    m.load_state_dict({
        "router.weight": rng.normal(0, .02, (E, D)).astype(np.float32),
        **{"experts.%d.%s.weight" % (e, s_): rng.normal(0, .02, sh).astype(np.float32)
           for e in range(E)
           for s_, sh in (("gate", (FF, D)), ("up", (FF, D)), ("down", (D, FF)))},
    })
    for n in (8, 128):
        x = T.tensor(rng.normal(0, .1, (n, D)).astype(np.float32))
        y = np.asarray(m(x).to_numpy(), np.float32)
        assert y.shape == (n, D)
        assert np.isfinite(y).all()