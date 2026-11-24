# -*- coding: utf-8 -*-
"""随包分发的示例插件。

放这里（而不是 `examples/`）是为了让 `pip install tgai-orv` 的用户也能直接加载，
不必去 GitHub 拷文件。`examples/plugins/` 下保留纯算子的示例。

`torch_compat.py`：把 `torch` 这个名字指向 ORV，演示「PyTorch API 面本身可以是插件」。

```python
import tgai_go.train.plugins.torch_compat   # 导入即登记（只登记，不安装）
import tgai_go.train as T
T.plugin.activate("torch-compat")           # 显式激活（会改 sys.modules）
import torch                                # 拿到的其实是 ORV
```
"""
