# -*- coding: utf-8 -*-
"""示例插件 B：自带反向的算子（精确 GELU，需要 erf）。

核心之所以不内置它：精确 GELU 需要 `erf`，而 LLM 训练里 tanh 近似已经够用
（见 `tgai_go.train.nn_ops.gelu`）。想要精确版的人可以装这个插件。
"""
import math

import numpy as np

from tgai_go.train import ops_plugin


def _erf(x):
    # 用 math.erf 的向量化封装（不需要 scipy）
    return np.vectorize(math.erf, otypes=[np.float32])(x)


def gelu_exact_fwd(x):
    return (x * 0.5 * (1.0 + _erf(x / np.float32(np.sqrt(2.0))))).astype(np.float32)


def gelu_exact_bwd(g, out, xs):
    (x,) = xs
    cdf = 0.5 * (1.0 + _erf(x / np.float32(np.sqrt(2.0))))
    pdf = np.exp(-0.5 * x * x) / np.float32(np.sqrt(2.0 * np.pi))
    return [g * (cdf + x * pdf)]


ops_plugin.register_numpy(
    "gelu_exact", gelu_exact_fwd, gelu_exact_bwd,
    kind="tensor",
    doc="精确 GELU（erf 版），带解析反向",
)
