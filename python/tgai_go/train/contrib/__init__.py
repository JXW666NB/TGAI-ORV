# -*- coding: utf-8 -*-
"""官方 contrib 插件：PyTorch 有、但**刻意不进核心**的算子。

## 为什么有这一层

核心只保留 LLM 训练必需的算子（见 `docs/OP_COVERAGE.md`）。
但用户从 PyTorch 迁移过来时，脚本里常常顺手用了 `erf` / `elu` / `log1p` / `flip` 这类算子 ——
它们不在核心里，但**必须有地方能拿到**，否则"改一行 import"就是空话。

所以：

* **核心** = 每台机器都要绿、每个后端都要实现的最小集；
* **contrib（本包）** = 官方维护的插件集，按 **PyTorch 同名同义** 实现，
  用户 `from tgai_go.train.contrib import autoload; autoload()` 一行就全部可用；
* **第三方插件** = 图形学/科研/私有算子，各自独立发版（见 `docs/PLUGINS.md`）。

## 实现约定（所有 contrib 算子都必须遵守）

1. **语义与 PyTorch 一致**：同名、同参数顺序、同默认值（如 `elu(alpha=1.0)`）；
2. **必须带解析反向**（除非该算子本身不可导，如 `floor`/`argmax`，则反向为 0）；
3. **必须有 PyTorch 对照测试**（`tests/test_contrib_ops.py`），判据 `allclose(rtol=1e-5)`；
4. **错误信息三要素**：哪个算子 / 什么形状 / 怎么修；
5. 不改变后端契约：全部走张量层，CUDA / Vulkan / ROCm 通用。

## 怎么用

```python
from tgai_go.train.contrib import autoload
autoload()                       # 注册全部 contrib 算子

import tgai_go.train as T
y = T.functional.erf(x)          # 或 x.erf()
```
"""
from . import activations, shape  # noqa: F401


def autoload():
    """注册全部官方 contrib 算子（重复调用安全：已注册的会跳过）。"""
    from . import _base
    return _base.register_all()


def list_ops():
    """列出 contrib 提供的算子名。"""
    from . import _base
    return _base.contrib_names()
