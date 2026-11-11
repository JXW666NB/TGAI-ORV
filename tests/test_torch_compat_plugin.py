# -*- coding: utf-8 -*-
"""`torch-compat` 示例插件：「PyTorch 的 API 面本身可以是插件」。

## 这个测试在证明什么

`examples/plugins/torch_compat.py` 是一个**单文件插件**。装上并激活后，
用 PyTorch 写法写的代码可以**一行不改**跑在 ORV 上：

```python
T.plugin.load("examples/plugins/torch_compat.py")
T.plugin.activate("torch-compat")

import torch, torch.nn as nn                      # 拿到的其实是 ORV
m = nn.Linear(4, 8)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
loss = (m(torch.tensor(x)) ** 2).mean(); loss.backward(); opt.step()
```

这条能成立，说明**核心不必背 PyTorch 的 API 面** —— 社区一个文件就能补。

## 守住什么

1. 装完 `import torch` 拿到的是 ORV（带标记，可识别）；
2. PyTorch 风格的完整闭环（建层 → 前向 → 反向 → 优化器 step）能跑通；
3. **绝不覆盖真的 torch** —— 已存在真 torch 时必须拒绝，而不是悄悄替换；
4. 卸载后 `sys.modules` 复原（幂等）；
5. 该插件只登记、不自动安装（装插件不该在不知情时改进程行为）。
"""
import os
import sys

import numpy as np
import pytest

import tgai_go.train as T
from tgai_go.train import plugin

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: 示例插件**随包分发**（放在包内，wheel 与 sdist 都带得上），
#: 这样 `pip install tgai-orv` 的用户也能直接 load，不必去 GitHub 拷文件。
PLUGIN_FILE = os.path.join(HERE, "python", "tgai_go", "train", "plugins",
                           "torch_compat.py")
MARK = "__tgai_torch_compat__"

#: 全量套件里其它测试会 `import torch`（真的 PyTorch）——那时本插件会**正确拒绝**覆盖，
#: 所以这两个用例只能在没有真 torch 的进程里跑。这不是跳过问题，是安全边界在工作。
_real_torch_loaded = pytest.mark.skipif(
    sys.modules.get("torch") is not None
    and not getattr(sys.modules.get("torch"), MARK, False),
    reason="本进程已加载真 torch，torch-compat 会（正确地）拒绝覆盖")


@pytest.fixture()
def compat():
    """装上并激活；用完卸载并复原 sys.modules。"""
    saved = {k: sys.modules.get(k) for k in
             ("torch", "torch.nn", "torch.nn.functional", "torch.optim", "torch.cuda")}
    plugin.load(PLUGIN_FILE)
    plugin.activate("torch-compat")
    yield
    for k in ("torch", "torch.nn", "torch.nn.functional", "torch.optim", "torch.cuda"):
        m = sys.modules.get(k)
        if m is not None and getattr(m, MARK, False):
            del sys.modules[k]
    for k, v in saved.items():
        if v is not None:
            sys.modules[k] = v
    plugin._APIS.clear()
    plugin._ACTIVATED.clear()


def test_plugin_file_exists():
    assert os.path.isfile(PLUGIN_FILE), "示例插件必须存在：%s" % PLUGIN_FILE


def test_registers_but_does_not_activate(compat=None):
    """加载只登记；激活是单独的显式动作。"""
    plugin.load(PLUGIN_FILE)
    assert "torch-compat" in T.plugins()["apis"]
    assert "torch-compat" not in T.plugins()["activated"]
    plugin._APIS.clear()


@_real_torch_loaded
def test_import_torch_yields_orv(compat):
    import torch
    assert getattr(torch, MARK, False), "装完后 import torch 应当是 ORV 的 shim"
    assert "tgai-compat" in torch.__version__
    assert torch.Tensor is T.Tensor


@_real_torch_loaded
def test_pytorch_style_loop_runs(compat):
    """★ 核心证明：PyTorch 写法的闭环一行不改能跑。"""
    import torch
    import torch.nn as nn

    m = nn.Linear(4, 8)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    x = torch.tensor(np.ones((2, 4), np.float32))

    before = np.asarray(m.weight.data, np.float32).copy()
    y = m(x)
    loss = (y * y).mean()
    loss.backward()
    opt.step()
    after = np.asarray(m.weight.data, np.float32)

    assert tuple(int(s) for s in y.shape) == (2, 8)
    assert np.isfinite(after).all()
    assert np.abs(after - before).max() > 0, "优化器 step 应真的更新权重"


def test_refuses_to_shadow_real_torch(compat=None):
    """★ 安全边界：真 torch 已加载时必须拒绝，不能悄悄替换。"""
    import types
    plugin.load(PLUGIN_FILE)                 # 先登记，否则 activate 什么都不做
    fake = types.ModuleType("torch")
    fake.__file__ = "<real torch>"
    sys.modules["torch"] = fake
    try:
        with pytest.raises(RuntimeError) as e:
            plugin.activate("torch-compat")
        assert "拒绝覆盖" in str(e.value)
        assert sys.modules["torch"] is fake, "拒绝时不该改动 sys.modules"
    finally:
        del sys.modules["torch"]
        plugin._APIS.clear()
        plugin._ACTIVATED.clear()


def test_uninstall_restores_sys_modules(compat):
    import torch  # noqa: F401
    spec = __import__("importlib.util", fromlist=["util"]).spec_from_file_location(
        "_tc", PLUGIN_FILE)
    mod = __import__("importlib.util", fromlist=["util"]).module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.uninstall_torch_compat()
    assert "torch" not in sys.modules, "卸载后应移除 shim"
    assert "torch.nn" not in sys.modules


def test_reload_is_idempotent(compat):
    """重复加载同一插件文件不该报重名（用户会 reload）。"""
    plugin.load(PLUGIN_FILE)
    plugin.load(PLUGIN_FILE)
    assert "torch-compat" in T.plugins()["apis"]
