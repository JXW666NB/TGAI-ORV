# -*- coding: utf-8 -*-
"""ORV 与 PyTorch 的**共享权重**前向数值对拍。

判据（ORV skill §6）：前向相对误差 ≤ 1e-5。

## 为什么必须共享权重

两侧各用自己的 `default_rng(0)` 初始化，**随机数消费顺序不同** ⇒ 权重不同 ⇒
loss 差异（实测 8.5012 vs 8.4227，相对 0.93%）**不含任何精度信息**。
2026-11-16 就在这一点上差点得出错误结论。

## 用法

需要一个同时能 import `torch` 和 `tgai_go.train` 的 Python。
若 ORV 装在别的 venv/site-packages，把它的路径插到最前面：

    # 例：V100 上 torch 在 C:\\Python311，ORV 0.9.83 在 D:\\orv_pip_0962
    #   sys.path 前插 D:\\orv_pip_0962\\Lib\\site-packages 后即可同进程比对
    python tools/parity_orv_vs_torch.py

## 当前覆盖

- [x] RMSNorm(γ, eps) → Linear(D→F) → SiLU → Linear(F→D)

## 待补（按风险排序）

- [ ] RoPE 约定（half-split vs interleaved；用错会 rel≈0.44 且**不报错**）
- [ ] GQA 的头扩展顺序
- [ ] 因果掩码的边界（causal 与 is_causal 语义一致）
- [ ] MoE 路由 top-k 的并列处理
- [ ] CSA（ORV 侧尚未实现，暂无法对拍）
- [ ] 整模型端到端（需逐层搬权重）
"""
import sys

import numpy as np


def check_ffn_block(B=2, S=64, D=128, F=256, seed=0, tol=1e-5):
    """RMSNorm → Linear → SiLU → Linear 的前向对拍。返回 (max_abs, max_rel)。"""
    import torch

    import tgai_go.train as T
    from tgai_go.train import nn_ops

    T.set_default_device("cuda")
    rng = np.random.default_rng(seed)
    x_n = rng.normal(0, 1, (B, S, D)).astype(np.float32)
    w1 = rng.normal(0, 0.02, (F, D)).astype(np.float32)
    w2 = rng.normal(0, 0.02, (D, F)).astype(np.float32)
    g = rng.normal(1, 0.01, (D,)).astype(np.float32)

    tx, tw1 = torch.tensor(x_n, device="cuda"), torch.tensor(w1, device="cuda")
    tw2, tg = torch.tensor(w2, device="cuda"), torch.tensor(g, device="cuda")
    ty = torch.nn.functional.silu(
        torch.nn.functional.rms_norm(tx, (D,), tg, 1e-6) @ tw1.T) @ tw2.T

    ox, ow1 = T.tensor(x_n), T.tensor(w1)
    ow2, og = T.tensor(w2), T.tensor(g)
    oy = T.functional.silu(nn_ops.rms_norm(ox, og, 1e-6) @ ow1.T) @ ow2.T

    a, b = oy.to_numpy(), ty.cpu().numpy()
    max_abs = float(np.abs(a - b).max())
    max_rel = max_abs / max(float(np.abs(b).max()), 1e-9)
    return max_abs, max_rel, float(np.abs(b).max())


def main():
    max_abs, max_rel, scale = check_ffn_block()
    print("FFN  块：max_abs=%.3e  max_rel=%.3e  输出量级=%.4f"
          % (max_abs, max_rel, scale))
    ok = max_rel <= 1e-5
    print("判定：%s（判据 max_rel <= 1e-5）" % ("通过" if ok else "不通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
