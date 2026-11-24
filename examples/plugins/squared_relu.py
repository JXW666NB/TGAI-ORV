# -*- coding: utf-8 -*-
"""示例插件 ①：自带反向的激活函数（`squared_relu`）。

这是**最简单的一种"带数学"的插件**：前向 8 行、反向 3 行，再写个 `PLUGIN` 契约，
就能被 `tools/check_plugin.py` 自动做数值梯度校验。

用法：
    python tools/check_plugin.py examples/plugins/squared_relu.py
    # 或在自己的代码里：
    from tgai_go.train import ops_plugin
    ops_plugin.load_plugin_file("examples/plugins/squared_relu.py")
    import tgai_go.train as T
    y = T.functional.squared_relu(x)

为什么激活函数要做成"平方 ReLU"：它是个常用的小变体（比 ReLU 平滑一点、比 GELU 便宜），
正好用来示范"前向/反向/契约/自检"四件事，而没有依赖任何私有实现。
"""
import numpy as np

from tgai_go.train import ops_plugin


def squared_relu_fwd(x):
    """max(x,0)^2"""
    return np.square(np.maximum(x, 0.0))


def squared_relu_bwd(g, out, xs):
    """d/dx max(x,0)^2 = 2*max(x,0)，再乘上游梯度 g。

    注意参数顺序是 (g, out, xs)：
      g   —— 上游梯度（与输出同形）
      out —— 前向输出（这里可以直接用 out 省一次乘法：d/dx = 2*sqrt(out)）
      xs  —— 前向输入列表（与注册时的参数顺序一致）
    """
    x = np.asarray(xs[0], dtype=np.float32)
    return [np.asarray(g, np.float32) * 2.0 * np.maximum(x, 0.0)]


ops_plugin.register_numpy(
    "squared_relu", squared_relu_fwd, squared_relu_bwd,
    doc="max(x,0)^2（示例插件：自带反向）")


#: 插件契约 —— 有了它，`tools/check_plugin.py` 才能自动做数值校验。
PLUGIN = {
    "name": "squared_relu",
    "version": "0.1.0",
    "api_version": ops_plugin.API_VERSION,
    "author": "TGAI ORV 示例",
    "description": "示例：自带反向的激活函数插件",
    "ops": {
        "squared_relu": {
            # 每个元素描述一个**位置参数**：形状/取值范围/是否需要梯度
            "test_inputs": [
                {"shape": [2, 4], "low": -2.0, "high": 2.0, "requires_grad": True,
                 "seed": 1},
            ],
        },
    },
}
