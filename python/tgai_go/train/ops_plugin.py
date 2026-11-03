# -*- coding: utf-8 -*-
"""算子插件系统 —— 「核心精简，一切皆插件」。

## 为什么要有它

我们是 **LLM 训练引擎**，核心只覆盖训练必需的那几十个算子（见 `docs/OP_COVERAGE.md`）。
但别人可能要：
* 图形学：`conv2d`、`grid_sample`、法线/光照类算子；
* 科研：`fft`、`svd`、特殊函数（`erf`、`bessel_j0`）；
* 自家模型的私有算子：门控、稀疏路由、自定义损失。

这些**不该塞进核心**（会让内核越滚越大、每个后端都要跟着实现），
而应该由需要的人**以插件形式加进来** —— 用完即走，核心不受影响。

## 两种插件（按难度从低到高）

### A. 组合式（推荐，几乎不用写数学）
只由已有 Tensor 算子拼出来 ⇒ **autograd 自动生效**，不用写反向。

```python
from tgai_go.train import ops_plugin, Tensor

def softplus(x):
    return (x.exp() + 1.0).log()          # 全用已有算子

ops_plugin.register("softplus", softplus)
# 之后：T.functional.softplus(x) 或  x.softplus()
```

### B. 自带反向（要自己写数学，性能优先）
给出 numpy 前向 + 反向（返回每个输入父节点的梯度）：

```python
import numpy as np

def gelu_exact(x):                       # numpy 前向
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def gelu_exact_bwd(g, out, xs):          # 反向：返回 [dL/dx]
    x, = xs
    pdf = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    cdf = 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    return [g * (cdf + x * pdf)]

ops_plugin.register_numpy("gelu_exact", gelu_exact, gelu_exact_bwd)
```

## 怎么被发现（三种，任选）

1. **手工注册**：`import` 你的模块即可；
2. **环境变量**：把 `.py` 放到某目录，设 `TG_OPS_PATH=/path/to/dir[:/path2]`，导入引擎时自动加载；
3. **包入口点**（推荐发布给别人用时）：
   ```toml
   [project.entry-points."tgai_go.ops"]
   my_ops = "my_pkg.ops:register_all"
   ```
   `pip install` 后自动被发现。

## 规则（写在这里，免得踩坑）

* **不许覆盖核心算子**：除非显式 `replace=True`，重名直接报错；
  「覆盖核心」是被允许的（有人想拿更快的实现替换），但必须自己声明，且会被记录。
* 插件是**用户代码**，引擎不会审计它的正确性 —— 数值对不对由插件的测试负责。
  自检命令可以列出当前加载了哪些插件（`ops_plugin.list_ops()`）。
* 插件**不改变**后端契约：组合式插件自动适配所有后端；自带反向的插件用的是 numpy，
  对 CUDA/CPU 都能跑，但绕过了融合核（性能另说）。
* 卸载：`unregister(name)`，用于实验对比或回滚。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np

from . import Tensor, _acc
from . import functional as F

#: 插件接口版本。**破坏性改动**（比如 backward 签名变了）必须 +1，
#: 这样老插件能明确报"接口不兼容"，而不是在某次训练里算出个错数。
API_VERSION = 1

#: name -> dict(kind, fn, backward, doc, source, replaced_core)
_REGISTRY = {}
#: 已加载的插件记录：file / sha256 / ops / api_version / loaded_at
_LOADED = []
#: 记录被插件替换掉的核心名字（透明化）
_REPLACED_CORE = set()

_CORE_NAMES = None


def _core_names(kind="functional"):
    """核心在**对应命名空间**里已提供的名字。

    为什么按命名空间：`functional` 和 `Tensor` 是两个地方。
    比如核心有 `Tensor.squeeze`（方法），插件想加 `functional.squeeze`（函数）——
    两者不冲突，不该判成重名。早期把两边并成一个集合，于是 contrib 的
    `squeeze` 直接被拒（实测踩过）。
    """
    global _CORE_NAMES
    if _CORE_NAMES is None:
        _CORE_NAMES = {
            "functional": {n for n in dir(F) if not n.startswith("_")},
            "tensor": {n for n in dir(Tensor) if not n.startswith("_")},
        }
    if kind == "tensor":
        # 会同时挂到 functional 与 Tensor，所以两边都要比
        return _CORE_NAMES["functional"] | _CORE_NAMES["tensor"]
    return _CORE_NAMES["functional"]


def register(name, fn, *, kind="functional", doc="", replace=False):
    """注册**组合式**算子（fn 只用已有 Tensor 算子实现，autograd 自动生效）。

    kind="functional" ⇒ 挂到 `tgai_go.train.functional.<name>`
    kind="tensor"     ⇒ 同时挂成 `Tensor.<name>` 方法（第一个参数是 self）
    """
    name = str(name).strip()
    if not name.isidentifier():
        raise ValueError("算子名必须是合法标识符：%r。修法：用字母/下划线开头，如 'my_op'" % (name,))
    if not callable(fn):
        raise TypeError("算子 %r 的实现必须可调用（Tensor -> Tensor）" % (name,)
                        + "。修法：传一个函数，参数是 Tensor、返回 Tensor")
    if name in _REGISTRY and not replace:
        raise ValueError(
            "算子 %r 已注册（来源：%s）。修法：换一个名字，或显式 replace=True 覆盖"
            % (name, _REGISTRY[name].get("source", "?")))
    if name in _core_names(kind) and not replace:
        raise ValueError(
            "算子 %r 与核心算子重名。修法：换名字；确实要替换核心实现请显式 replace=True"
            % (name,))
    if name in _core_names(kind):
        _REPLACED_CORE.add(name)

    _REGISTRY[name] = dict(kind=kind, fn=fn, backward=None, doc=doc, source="manual",
                           replaced_core=name in _core_names(kind))
    setattr(F, name, fn)
    if kind == "tensor":
        setattr(Tensor, name, fn)
    return fn


def register_numpy(name, forward, backward, *, kind="functional", doc="", replace=False):
    """注册**自带反向**的算子。

    forward(*np_arrays, **kwargs) -> np.ndarray
    backward(g, out, xs) -> list[np.ndarray]（与 xs 一一对应；不需要梯度的位置给 None）
    """
    if not callable(forward) or not callable(backward):
        raise TypeError("register_numpy: forward 与 backward 都必须可调用"
                        + "。修法：两个参数都传函数；不需要反向就改用 register()")

    def op(*args, **kwargs):
        tensors = [a for a in args if isinstance(a, Tensor)]
        if not tensors:
            raise TypeError(
                "算子 %r 的输入里没有 Tensor。修法：至少传一个 Tensor（纯 numpy 输入直接算即可）"
                % (name,))
        arrays = [a.data if isinstance(a, Tensor) else a for a in args]
        out_data = np.asarray(forward(*arrays, **kwargs), dtype=np.float32)
        need_grad = any(getattr(a, "requires_grad", False) for a in args)
        out = Tensor(out_data, requires_grad=need_grad, _parents=tuple(tensors))

        def bw(g, grads):
            grads_in = backward(np.asarray(g, np.float32), out_data, arrays, **kwargs)
            if not isinstance(grads_in, (list, tuple)):
                grads_in = [grads_in]
            for a, gi in zip(args, grads_in):
                if isinstance(a, Tensor) and a.requires_grad and gi is not None:
                    _acc(grads, a, np.asarray(gi, dtype=np.float32))

        out._backward = bw
        return out

    return register(name, op, kind=kind, doc=doc, replace=replace)


def unregister(name):
    """卸载插件算子（恢复核心原样，若它覆盖过核心则无法自动恢复 —— 会明确告知）。"""
    rec = _REGISTRY.pop(name, None)
    if rec is None:
        raise KeyError("算子 %r 没有注册过。已注册：%s" % (name, ", ".join(sorted(_REGISTRY)) or "(空)")
                       + "。修法：先 ops_plugin.register() 再调用")
    if rec.get("replaced_core"):
        raise RuntimeError(
            "算子 %r 覆盖过核心实现，卸载无法自动恢复原实现。"
            "修法：重启进程（核心定义在 import 时建立）" % (name,))
    delattr(F, name)
    if rec["kind"] == "tensor" and hasattr(Tensor, name):
        delattr(Tensor, name)
    return rec


def list_ops():
    """列出当前已注册的插件算子（人读格式）。"""
    rows = []
    for name in sorted(_REGISTRY):
        r = _REGISTRY[name]
        rows.append(dict(name=name, kind=r["kind"],
                         has_backward=r["backward"] is not None or r["kind"] == "numpy",
                         replaced_core=r.get("replaced_core", False),
                         doc=(r.get("doc") or "").splitlines()[0] if r.get("doc") else ""))
    return rows


def registered_names():
    return sorted(_REGISTRY)


# ---------------- 发现机制 -------------------------------------------------

def _strict():
    v = os.environ.get("TGAI_OPS_STRICT", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _rollback(added):
    """把被拒绝的插件注册过的算子撤掉。

    不撤销的话，严格模式下"拒绝了插件"但注册表已经被污染 ——
    后续代码仍然能调到它，等于拒绝了个寂寞。
    """
    for name in added:
        try:
            unregister(name)
        except Exception:                        # noqa: BLE001  覆盖过核心的没法自动撤
            pass


def load_plugin_file(path):
    """加载单个 .py 插件文件（模块内调用 register 即可生效）。

    加载后会记录：文件路径、**sha256**、新增了哪些算子、插件声明的 api_version。
    `TGAI_OPS_STRICT=1` 时，没有 `PLUGIN` 契约的插件会被拒绝 —— 给 CI/团队用。
    """
    import hashlib

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError("插件文件不存在：%s" % (path,)
                                + "。修法：检查路径；.tgop 请用 ops_bundle.load()")
    raw = open(path, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()

    before = set(_REGISTRY)
    mod_name = "tgai_plugin_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    added = sorted(set(_REGISTRY) - before)

    info = getattr(mod, "PLUGIN", None)
    rec = dict(file=path, sha256=digest, ops=added,
               api_version=(info or {}).get("api_version") if isinstance(info, dict) else None,
               root=os.path.dirname(path), loaded_at=time.time())
    if isinstance(info, dict):
        rec.update({k: info.get(k) for k in ("name", "version", "author", "description")
                    if info.get(k) is not None})
        problems = check_contract(info, added)
        rec["contract_problems"] = problems
        if problems and _strict():
            _rollback(added)
            raise ValueError(
                "插件 %s 的 PLUGIN 契约不合规（TGAI_OPS_STRICT=1）：\n  %s\n"
                "修法：按 docs/PLUGINS.md 的契约补齐；或用 python tools/check_plugin.py 逐条修"
                % (os.path.basename(path), "\n  ".join(problems)))
    elif _strict():
        _rollback(added)
        raise ValueError(
            "插件 %s 没有 PLUGIN 元数据块（TGAI_OPS_STRICT=1）。\n"
            "修法：在插件末尾加 PLUGIN = {\"name\": ..., \"version\": ..., "
            "\"api_version\": %d, \"ops\": {...}}；详见 docs/PLUGINS.md"
            % (os.path.basename(path), API_VERSION))
    else:
        rec["contract_problems"] = ["没有 PLUGIN 元数据块（无法校验接口版本与自检输入）"]

    _LOADED.append(rec)
    return mod


def check_contract(info, added_ops=()):
    """校验插件声明的 PLUGIN 契约，返回问题列表（空 = 合规）。

    契约（都可被工具检查）：
      * `name` / `version` / `api_version` 必填；
      * `api_version` 必须与引擎的 `API_VERSION` 主版本一致；
      * `ops` 里声明的算子在本次加载中必须**真的注册了**；
      * 每个算子若声明 `test_inputs`，必须是 list[dict]（供 tools/check_plugin.py 做数值自检）。
    """
    problems = []
    if not isinstance(info, dict):
        return ["PLUGIN 必须是 dict"]
    for key in ("name", "version", "api_version"):
        if info.get(key) in (None, ""):
            problems.append("缺少 %s" % key)
    api = info.get("api_version")
    if isinstance(api, int) and api != API_VERSION:
        problems.append("api_version=%s 与引擎 API_VERSION=%s 不一致" % (api, API_VERSION))
    ops = info.get("ops")
    if ops is not None:
        if not isinstance(ops, dict):
            problems.append("ops 必须是 dict：{算子名: {...}}")
        else:
            added = set(added_ops or ())
            for name, spec in ops.items():
                if added and name not in added:
                    problems.append("ops 里声明了 %r，但本次加载没有注册它" % (name,))
                if spec is not None and not isinstance(spec, dict):
                    problems.append("%s 的规格必须是 dict（可含 test_inputs）" % (name,))
                elif isinstance(spec, dict):
                    ti = spec.get("test_inputs")
                    if ti is not None and not (isinstance(ti, list)
                                               and all(isinstance(x, dict) for x in ti)):
                        problems.append("%s.test_inputs 必须是 list[dict]" % (name,))
    return problems


def manifest():
    """当前进程加载过的插件清单（含 sha256，可复现）。"""
    out = []
    for r in _LOADED:
        out.append({k: r.get(k) for k in
                    ("name", "version", "api_version", "file", "sha256", "ops",
                     "contract_problems")})
    return out


def load_from_env():
    """按 TG_OPS_PATH（os.pathsep 分隔的目录或文件）加载插件。"""
    raw = os.environ.get("TG_OPS_PATH", "").strip()
    if not raw:
        return []
    loaded = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        if os.path.isdir(part):
            for fn in sorted(os.listdir(part)):
                if fn.endswith(".py") and not fn.startswith("_"):
                    loaded.append(load_plugin_file(os.path.join(part, fn)))
        elif part.endswith(".py"):
            loaded.append(load_plugin_file(part))
    return loaded


def load_entry_points(group="tgai_go.ops"):
    """加载已安装包声明的插件入口点（[project.entry-points."tgai_go.ops"]）。"""
    try:
        from importlib.metadata import entry_points
    except Exception:  # noqa: BLE001
        return []
    loaded = []
    try:
        eps = entry_points()
        selected = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
    except Exception:  # noqa: BLE001
        return []
    for ep in selected:
        try:
            obj = ep.load()
            if callable(obj):
                obj()                 # 约定：入口点是一个「注册函数」
            loaded.append(ep)
        except Exception as exc:  # noqa: BLE001
            # 插件坏了不能拖垮引擎：记录后继续
            sys.stderr.write("[tgai_go.ops] 插件 %s 加载失败：%s\n" % (getattr(ep, "name", "?"), exc))
    return loaded


def autoload():
    """引擎 import 时调用：先按入口点、再按 TG_OPS_PATH 自动加载。"""
    load_entry_points()
    load_from_env()


# 说明：这里**不在 import 时自动 autoload** —— 自动执行用户代码应当是显式的。
# 需要自动加载请在启动脚本里调用 ops_plugin.autoload()，或设 TG_OPS_AUTOLOAD=1。
if os.environ.get("TG_OPS_AUTOLOAD", "").strip() in ("1", "true", "yes"):
    autoload()
