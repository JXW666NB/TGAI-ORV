# -*- coding: utf-8 -*-
"""contrib 的内部登记表：让「声明式定义算子」和「批量注册」只写一次。"""
from .. import ops_plugin

#: name -> (fwd, bwd, doc)
_SPECS = {}


def numpy_op(name, doc=""):
    """装饰器：`fwd(x, **kw)` 与 `bwd(g, out, xs, **kw)` 成对声明。

    这样每个算子的数学只写一遍，注册/文档/测试清单都从这张表里来。
    """

    def deco(pair):
        # 支持两种写法：@numpy_op(...) 套在「返回 (fwd,bwd) 的函数」上，
        # 或直接套在 (fwd, bwd) 元组上。
        if callable(pair) and not isinstance(pair, tuple):
            fwd, bwd = pair()
        else:
            fwd, bwd = pair
        if name in _SPECS:
            raise RuntimeError("contrib 算子 %r 重复声明" % (name,))
        _SPECS[name] = (fwd, bwd, doc)
        return pair

    return deco


def register_all():
    """把所有已声明算子注册进插件系统（幂等）。"""
    already = set(ops_plugin.registered_names())
    added = []
    for name, (fwd, bwd, doc) in sorted(_SPECS.items()):
        if name in already:
            continue
        ops_plugin.register_numpy(name, fwd, bwd, kind="tensor", doc=doc)
        added.append(name)
    return added


def contrib_names():
    return sorted(_SPECS)
