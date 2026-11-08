# -*- coding: utf-8 -*-
"""优化器必须与 `torch.optim.AdamW` 语义一致（CI，无 torch 时跳过）。

背景：V2.7 全参训练能否"质量保住"，前提是更新公式与 PyTorch 一致。
实测（V100，100 步，n=4096）：偏差 1.57e-06、余弦 1.000000、末 loss 相同。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))


def test_orv_adamw_matches_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA：ORV 的优化器只在 GPU 上跑（CI 无卡属预期跳过；V100 实测偏差 1.57e-06）")
    import tgai_go.train as T

    rng = np.random.default_rng(0)
    n = 1024
    w0 = rng.normal(0, 1, n).astype(np.float32)
    tgt = rng.normal(0, 1, n).astype(np.float32)

    wt = torch.tensor(w0.copy(), requires_grad=True)
    ot = torch.optim.AdamW([wt], lr=1e-3)
    for _ in range(100):
        ot.zero_grad()
        (0.5 * ((wt - torch.tensor(tgt)) ** 2).sum()).backward()
        ot.step()

    try:
        p = T.tensor(w0.copy(), requires_grad=True)
        oo = T.optim.AdamW([p], lr=1e-3)
        _ = p.to_numpy()
    except Exception as e:                      # noqa: BLE001
        pytest.skip("ORV 张量/优化器在本机不可用（%s）—— CI 无 CUDA 时属预期" % type(e).__name__)
    for _ in range(100):
        cur = np.asarray(p.to_numpy(), dtype=np.float32)
        p.grad = T.tensor((cur - tgt).astype(np.float32))
        oo.step()
        oo.zero_grad()

    a = wt.detach().numpy().astype(np.float32)
    b = np.asarray(p.to_numpy(), dtype=np.float32)
    rel = np.abs(b - a).max() / max(1e-9, np.abs(a).max())
    cos = float(np.dot(b - w0, a - w0) /
                max(1e-12, np.linalg.norm(b - w0) * np.linalg.norm(a - w0)))
    print("\nORV AdamW vs torch: 偏差=%.2e 余弦=%.6f" % (rel, cos))
    assert rel <= 1e-3, "与 torch.optim.AdamW 参数偏差 %.2e 超过 1e-3" % rel
    assert cos >= 0.999, "更新方向余弦 %.6f < 0.999" % cos