# -*- coding: utf-8 -*-
"""**10 行以内**训练一个你自己的 Transformer（ORV 开箱即用层，0.9.21）。

去掉注释和 import，用户要写的实质代码就这么多：

    import numpy as np
    import tgai_go.train as T
    base  = np.array([1, 5, 9, 13, 17, 21, 25])      # 一个周期模式（换成你的 token）
    ids   = np.resize(base, 4000)
    data  = T.tokens(ids, seq=32)
    model = T.preset.transformer(vocab=32, d_model=32, n_layers=2, n_heads=4)
    hist  = T.fit(model, data, steps=80, lr=3e-3)
    print(hist["loss"][0], "->", hist["loss"][-1])

> 为什么示例用**周期性** token 而不是纯随机：随机 token 的熵就是 `ln(vocab)`≈4.16，
> 任何模型都降不下去，看曲线会误以为"没学会"。周期性数据可学，loss 会明显下降，
> 才能验证"这套训练循环真的在优化"。

## 换成你自己的模型

把 `T.preset.transformer(...)` 换成你自己写的 `nn.Module` 即可 —— `T.fit` 不关心模型长什么样，
只需要 `parameters()` 和 `forward(x, y) -> loss`。

## 想加 AMP

`T.fit(..., amp=True)`（V100 这类卡上大 GEMM 有 6 倍级收益）。

运行：
    python examples/ten_lines.py
"""
import numpy as np

import tgai_go.train as T

base = np.array([1, 5, 9, 13, 17, 21, 25], np.int64)
ids = np.resize(base, 4000)
data = T.tokens(ids, seq=32)
model = T.preset.transformer(vocab=32, d_model=32, n_layers=2, n_heads=4)
hist = T.fit(model, data, steps=80, lr=3e-3, log_every=20)
print("first/last loss: %.4f -> %.4f（参数量 %d）"
      % (hist["loss"][0], hist["loss"][-1], model.n_params()))
