# 插件式算子（一切皆插件）

## 为什么要插件，而不是把算子全塞进核心

我们是 **LLM 训练引擎**，核心只保留训练必需的那几十个算子（清单见 `docs/OP_COVERAGE.md`）。
但需求会外溢：

* **图形学**：`conv2d`、`grid_sample`、法线/光照；
* **科研**：`fft`、`svd`、特殊函数（`erf`、贝塞尔）；
* **私有算子**：自定义门控、稀疏路由、自定义损失。

把这些塞进核心有三个坏处：
1. 每个后端（CUDA / Vulkan / ROCm）都要跟着实现一遍；
2. 内核对不上就没法自检，`doctor` 会越来越难绿；
3. 99% 的用户根本用不到，却要为它们承担体积与维护成本。

**所以：核心精简，其余一切以插件形式加。**

## 两种插件

| 类型 | 适用 | 要不要写反向 | 性能 |
| --- | --- | --- | --- |
| **组合式** | 能由已有算子拼出来（多数情况） | **不用**，autograd 自动生效 | 一般（多次 kernel 调用） |
| **自带反向** | 需要精确数值或想减少中间张量 | 要（numpy 前向 + 解析反向） | 取决于实现 |

### A. 组合式（推荐先试这个）

```python
from tgai_go.train import ops_plugin

def softplus(x):
    return (x.exp() + 1.0).log()

ops_plugin.register("softplus", softplus, kind="tensor")
# 之后：T.functional.softplus(x)  或  x.softplus()
```

**实测**（`examples/plugins/softplus_plugin.py`）：与 `torch.nn.functional.softplus`
相比 fwd `2.4e-07`、**bwd `6.0e-08`** —— 一行反向都没写，梯度就是对的。

### B. 自带反向

```python
ops_plugin.register_numpy("gelu_exact", fwd, bwd, kind="tensor")
# fwd(x) -> np.ndarray         ；bwd(g, out, xs) -> [dL/dx]
```

**实测**（`examples/plugins/gelu_exact_plugin.py`）：精确 GELU（erf 版）与 torch 相比
fwd `2.4e-07`、**bwd `1.2e-07`**。

## 怎么让插件被加载（三选一）

| 方式 | 做法 | 适合 |
| --- | --- | --- |
| 手工 | `import 你的模块` | 临时试验 |
| 目录/环境变量 | `TG_OPS_PATH=examples/plugins`（可多路径，`:` 分隔）+ `ops_plugin.autoload()` | 团队内共享 |
| 包入口点 | `[project.entry-points."tgai_go.ops"] my_ops = "my_pkg.ops:register_all"` | 发布给别人 `pip install` |

> 引擎**不会**在 import 时自动执行用户代码（安全考虑）。要自动加载请显式调用
> `ops_plugin.autoload()`，或设 `TG_OPS_AUTOLOAD=1`。

## 规则

* **不许静默覆盖核心算子**：重名会直接报错；确实想替换核心实现要显式 `replace=True`，
  且会被记入 `list_ops()`（透明化）。
* **插件是用户代码**，引擎不审计其正确性 —— 数值正确性由插件自己的测试负责。
  建议照 `tests/test_op_plugins.py` 的写法，用 PyTorch 做对照。
* **插件不改变后端契约**：组合式插件自动适配任何后端；自带反向的插件走 numpy
  （任何后端都能跑，但绕过了融合核）。
* **卸载**：`unregister(name)`。覆盖过核心的插件无法热恢复，会明确报错让你重启进程。

---

## 官方 contrib：我们相对 PyTorch 缺的算子，按插件提供

核心不装冷门算子，但**必须有官方范例**，否则别人不会用、也不敢自己写。
所以有一个官方维护的 contrib 包：`tgai_go.train.contrib`。

```python
from tgai_go.train.contrib import autoload
autoload()                       # 一行注册全部
import tgai_go.train as T
y = T.functional.erf(x)          # 或 x.erf()
```

**已有 20 个**（每个都与 PyTorch 同名同义，且带解析反向、有对照测试）：

| 类别 | 算子 |
| --- | --- |
| 激活/逐元素 | `erf` `gelu_exact` `elu` `selu` `leaky_relu` `softplus` `mish` `hardswish` `hardsigmoid` `log1p` `lerp` |
| 不可导（反向为 0） | `sign` `floor` `ceil` `trunc` |
| 形状 | `flip` `roll` `flatten` `unsqueeze` `squeeze` |

测试：`tests/test_contrib_ops.py`（前向 + 反向都与 torch 对比，判据 `allclose(rtol=1e-5)`）。

### 为什么这些进 contrib 而不进核心

判断标准只有一条：**LLM 训练必需吗？**

* `silu` / `gelu(tanh)` / `softmax` / `rmsnorm` / `embedding` / `cross_entropy` → **核心**；
* `erf`（精确 GELU 才用）、`elu` / `selu` / `mish` / `hardswish`（都是"别的模型的激活"）、
  `flip` / `roll`（数据增强或特殊布局）→ **contrib**；
* `conv2d` / `fft` / `svd` / `sparse` → **不在仓库里**，由第三方插件包（如 `tgai-go-ops-vision`）提供。

### 加新 contrib 算子的流程（照抄即可）

```python
# python/tgai_go/train/contrib/activations.py
from ._base import numpy_op

@numpy_op("my_act", doc="一句话说明 + 与 torch 哪个算子对应")
def _():
    def fwd(x, alpha=1.0):
        return (alpha * x).astype(F32)          # numpy 前向

    def bwd(g, out, xs, **kw):                   # 反向：额外位置参数从 xs 取
        alpha = xs[1] if len(xs) > 1 else kw.get("alpha", 1.0)
        return [(g * alpha).astype(F32)]

    return fwd, bwd
```

然后在 `tests/test_contrib_ops.py` 的 `DIFF_CASES` 里加一行对照即可。

## 与「后端移植」的关系

插件算子运行在**张量层**，所以它天然跟后端无关：
* CUDA 上：组合式算子的每次调用都走已有内核；
* Vulkan/ROCm 上：只要核心那 54 个方法齐了，插件同样能用。

换句话说：**插件系统让"新硬件"和"新算子"解耦** —— 移植后端的人不用管图形学算子，
写图形学算子的人不用管后端。

## 图形学/科研用户怎么用

1. 按 `tier` 判断要不要进核心：**只有 LLM 训练必需才进核心**，其余一律插件；
2. 图形学算子建议打包成 `tgai-go-ops-vision`，科研打包成 `tgai-go-ops-sci`，各自独立发版；
3. 它们的正确性测试用 numpy/torch 做基准（不需要 GPU），性能测试再上目标后端。

这样核心永远是「一个精简、可自检、每台机器都能绿」的 LLM 训练引擎，
而生态由插件自由长出来。

---

# 写一个"别人能用"的插件（生态规范）

上面的最小示例够你自己用。但如果要**发给别人**，还差三样：契约、自检、来源可追溯。
这三样都已实现，下面给出照抄步骤。

## 1. 加 `PLUGIN` 契约块

在插件文件**末尾**加一个模块级 `PLUGIN` dict：

```python
PLUGIN = {
    "name": "my_ops",              # 插件名（小写，建议与文件名一致）
    "version": "0.1.0",            # 插件自己的版本（semver）
    "api_version": ops_plugin.API_VERSION,   # 你按哪一版接口写的
    "author": "你的名字",
    "description": "一句话说明",
    "ops": {
        "my_act": {
            # 每个位置参数一条：形状 / 取值范围 / 是否需要梯度
            "test_inputs": [
                {"shape": [2, 4], "low": -2.0, "high": 2.0,
                 "requires_grad": True, "seed": 1},
            ],
        },
    },
}
```

`test_inputs` 是给**自动数值自检**用的：工具会照这个规格造输入，跑前向，
再用**中心差分**核对你的反向。没有它，工具只能确认"算子注册上了"，无法验证数学。

## 2. 跑自检

```bash
python tools/check_plugin.py examples/plugins/squared_relu.py
```

输出形如：

```
插件：squared_relu.py
  加载：OK
  API：1（引擎 1）  版本：0.1.0  指纹：bcae5f3f4ac7efab…
  [OK ] squared_relu       前向+梯度 OK（最大相对误差 3.06e-05）
结论：PASS
```

反向写错时（比如漏乘系数 2）：

```
  [FAIL] bad_act            输入 0 的梯度和中心差分差 5.00e-01（阈值 2e-02）。
                            常见原因：反向的转置方向、sum 的维度、少了/多了系数
结论：FAIL
```

> 阈值取 2e-2 是因为引擎跑 float32、中心差分本身有 ~1e-3 误差；
> 结构性错误（转置反了、符号错了）是 O(1) 量级，一定抓得住。

## 3. 团队/CI 里开严格模式

```bash
TGAI_OPS_STRICT=1 python your_train_script.py
```

严格模式（`TGAI_OPS_STRICT=1`）下：
* 插件没有 `PLUGIN` 契约 → **拒绝加载**；
* 契约不合规（缺字段、api_version 不匹配、声明了却没注册）→ **拒绝加载**；
* 被拒绝时会把该插件已注册的算子**回滚**，不留半截状态。

## 4. 看清"现在到底加载了什么"

```python
from tgai_go.train import ops_plugin
for r in ops_plugin.manifest():
    print(r["name"], r["version"], r["sha256"][:16], r["ops"])
```

每条记录含 **sha256**（插件文件的指纹）与它新注册的算子。
出问题时把这段贴出来，别人就能确认是不是同一份代码。

## 5. 打包成可安装的插件（可选）

让别人 `pip install` 就能用，靠 entry point：

```toml
# 你的插件包 pyproject.toml
[project.entry-points."tgai_go.ops"]
my_ops = "my_ops_pkg.plugin"
```

```python
# 用户侧
from tgai_go.train import ops_plugin
ops_plugin.load_entry_points()        # 扫描并加载所有 tgai_go.ops 入口
```

## 6. 插件作者检查清单

发出去之前逐条对一遍：

- [ ] `PLUGIN` 四项齐全：`name` / `version` / `api_version` / `ops`
- [ ] 每个算子都有 `test_inputs`，且 `python tools/check_plugin.py` **PASS**
- [ ] 没有和核心算子重名（要覆盖核心必须显式 `replace=True`，且说明理由）
- [ ] `register_numpy` 的 `backward` 返回**与输入同顺序**的列表
- [ ] 边界输入不炸：全零、负数、极大值（`test_inputs` 里可加一组极值）
- [ ] 文档字符串写清：输入形状、输出形状、是否可导
- [ ] 说明性能预期（组合式 ≈ 拼出来的算子之和；自带反向绕过融合核，可能更慢）

## 7. 信任模型（先说清楚，别误会）

* 插件是**用户代码**，加载时会被**直接执行**；引擎**不做沙箱**。
  所以只加载你信任的插件 —— 这点和 pytest 插件、Jupyter 扩展一样。
* `sha256` 只用于**溯源与复现**，不是签名验证，不能证明"这份代码是安全的"。
* `TGAI_OPS_STRICT=1` 管的是**契约合规**，不是安全。
* 想让插件跑在 GPU 上：组合式插件自动跟随后端；`register_numpy` 的插件走 numpy，
  对 CUDA 也能用，但**绕过了融合核**，性能要自己评估。
