# -*- coding: utf-8 -*-
"""示例插件 A：组合式算子（不用写反向，autograd 自动生效）。

软加一（Softplus）：softplus(x) = log(1 + exp(x))
只由已有算子拼出来 —— 这就是「用户想加算子」最省力的方式。
"""
from tgai_go.train import ops_plugin


def softplus(x):
    # log(1+exp(x)) 数值上直接写会溢出；用 log1p(exp(-|x|)) + max(x,0) 更稳
    # 这里为了演示「只用已有算子」，采用 exp→加→log 的简单写法。
    return (x.exp() + 1.0).log()


ops_plugin.register(
    "softplus", softplus,
    kind="tensor",
    doc="log(1+exp(x))，组合式实现（示例）",
)
