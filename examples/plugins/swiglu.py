# -*- coding: utf-8 -*-
"""示例插件 ②：**组合式**算子（`swiglu`）—— 一行数学，反向自动生效。

和 ①（自带反向）的区别：
  * 组合式只用**已有 Tensor 算子**拼出来，autograd 自动接管，**不用写反向**；
  * 因此它天然适配所有后端（CUDA/CPU/未来后端），也不绕开融合核；
  * 代价：性能取决于拼出来的那些算子，不会有专门优化。

`swiglu(a, b) = silu(a) * b` —— LLaMA/Qwen 的 MLP 门控就是这么写的，
这里把它做成插件是为了示范"不改核心也能加算子"。

用法：
    python tools/check_plugin.py examples/plugins/swiglu.py
"""
import tgai_go.train as T
from tgai_go.train import ops_plugin


def swiglu(gate, up):
    """silu(gate) * up —— 全用已有算子，autograd 自动生效。"""
    return T.functional.silu(gate) * up


ops_plugin.register(
    "swiglu", swiglu, kind="functional",
    doc="silu(gate) * up（示例插件：组合式，反向自动）")


PLUGIN = {
    "name": "swiglu",
    "version": "0.1.0",
    "api_version": ops_plugin.API_VERSION,
    "author": "TGAI ORV 示例",
    "description": "示例：组合式插件（两个输入，无手写反向）",
    "ops": {
        "swiglu": {
            "test_inputs": [
                {"shape": [2, 4], "low": -3.0, "high": 3.0, "requires_grad": True,
                 "seed": 1},
                {"shape": [2, 4], "low": -3.0, "high": 3.0, "requires_grad": True,
                 "seed": 2},
            ],
        },
    },
}
