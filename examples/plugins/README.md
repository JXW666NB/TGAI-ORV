# 插件示例

两个最小示例，分别演示两种插件写法：

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `softplus_plugin.py` | 组合式 | 只用已有算子，**不用写反向**（autograd 自动生效） |
| `gelu_exact_plugin.py` | 自带反向 | numpy 前向 + 解析反向，适合需要精确数值/性能的场景 |

## 试一下

```bash
python -c "
import tgai_go.train as T
from tgai_go.train import ops_plugin
ops_plugin.load_from_env()          # 需先设 TG_OPS_PATH=examples/plugins
import numpy as np
x = T.tensor(np.array([[-1.0, 0.0, 2.0]], np.float32), requires_grad=True)
y = ops_plugin._REGISTRY['gelu_exact']['fn'](x)
print(y.to_numpy())
y.sum().backward()
print(x.grad)
"
```

或：

```bash
set TG_OPS_PATH=examples/plugins      # Windows
export TG_OPS_PATH=examples/plugins   # Linux
set TG_OPS_AUTOLOAD=1
python -m tgai_go.train.doctor       # 自检仍应 12/12（插件不影响核心）
```

详见 `docs/PLUGINS.md`。
