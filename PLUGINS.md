# 插件规范 / Plugin Spec

ORV 的插件系统让第三方**不改核心**就能加算子、层、优化器。

## 插件形式

1. **单文件**（如 `polaris.py`）
2. **wheel + entry point**（推荐分发）
3. 包内 `import xxx` 触发注册

## entry point 约定

```toml
[project.entry-points."tgai_go.ops"]
myplugin = "myplugin_module:register"
```

`tgai_go.train.plugin.autoload()` 自动发现并调用 `register()`。

## 必须实现的注册函数

```python
def register(T=None, replace=False):
    if T is None:
        import tgai_go.train as T
    existing = T.plugin.resolve_optimizer("MyOpt")
    if existing is not None and not replace:
        return existing
    T.plugin.optimizer("MyOpt", MyOptClass, works_inplace=False, replace=replace)
    return MyOptClass
```

模块级再调一次 `_auto_register()`，这样 `import myplugin` 也生效。

## 优化器插件接口

| 方法 | 说明 |
|---|---|
| `__init__(self, params, lr=..., **kw)` | `params` 可迭代张量 |
| `step(self, closure=None)` | 一次更新 |
| `zero_grad(self)` | 清梯度（ORV 约定 `p.grad = None`）|
| `status(self)` | 可选但强烈建议，返回 dict 供日志/调试 |

`status()` 建议含 `step`、当前 lr 与该优化器的关键内部量。

## 数值验证是硬要求

进官方索引前必须提供：

1. 与参考实现（AdamW 或该算法 CPU 版）在**同 lr** 与**各自最优 lr** 两种口径下的收敛对比
2. 至少一个**真实模型**上的短跑结果
3. 明确说明 CUDA 路径与 CPU 路径是否**算法等价**；若用了近似，必须显式写出来

> 血泪案例：社区插件 Polaris 的 CUDA 路径把「逐行 RMS」近似成「整体 RMS」；
> 玩具任务上同 lr 比 AdamW 差 19×，各自最优 lr 下差 28×。
> 这类近似必须写在 README 最显眼处。

## 官方插件索引

| 插件 | 类型 | 状态 | 作者 |
|---|---|---|---|
| `tgai-polaris` | 优化器（Muon 系）| experimental；CUDA 路径有近似待修 | BLLM |