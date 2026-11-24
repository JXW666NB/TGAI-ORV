# -*- coding: utf-8 -*-
"""纯声明式原生插件清单（JSON）—— **不需要写一行 Python**。

## 为什么要有它

插件应当**优先是 C 端**：用户用自己的工具链编译出原生库，再用一份 JSON 说明
「哪个符号、参数怎么映射」，ORV 负责加载与调用。Python 只在 ORV 内部跑一次
（读 JSON、做绑定），热路径完全是原生代码。

## 清单格式

```json
{
  "name": "myops",
  "version": "0.1.0",
  "api_version": "1.0",
  "library": "myops.dll",
  "ops": [
    {
      "name": "scale",
      "symbol": "my_scale",
      "args": ["ptr:0", "ptr:1", "f32:2", "int:3"]
    }
  ]
}
```

* `library`：相对清单文件所在目录解析，也可给绝对路径。
* `ops[].args`：按顺序描述每个 C 参数怎么来。支持：
  - `"ptr:N"` —— 第 N 个调用实参的原始指针（`_gpu.ptr`，CUDA 缓冲）
  - `"f32:N"` / `"f64:N"` —— 转成 float
  - `"int:N"` —— 转成 int
  - `"raw:N"` —— 原样传（只适用于标量）
* `ops[].restype`：可选，`"void"`（默认）/ `"int"` / `"double"`。

## 安全声明

加载原生库等于执行任意机器码。ORV **不做沙箱、不校验来源**。
只加载你信任的库 —— 这一点在文档里也必须写明，不能含糊。
"""

from __future__ import annotations

import json
import os

__all__ = ["load_manifest", "parse_args_spec"]


def parse_args_spec(spec, argv):
    """把 `["ptr:0", "f32:2"]` 这样的声明转成实际的 C 参数列表。"""
    if not isinstance(spec, list):
        raise ValueError("args 必须是列表，收到 %r。修法：写成 [\"ptr:0\", \"f32:1\"]" % (spec,))
    out = []
    for item in spec:
        s = str(item)
        if ":" not in s:
            raise ValueError(
                "参数声明 %r 缺少类型前缀" % s +
                "。修法：写成 ptr:0 / f32:1 / int:2 / raw:3")
        kind, idx_s = s.split(":", 1)
        try:
            idx = int(idx_s)
        except ValueError:
            raise ValueError(
                "参数声明 %r 的索引不是整数" % s +
                "。修法：索引是调用实参的位置，从 0 开始") from None
        if idx < 0 or idx >= len(argv):
            raise ValueError(
                "参数声明 %r 超出实参个数 %d" % (s, len(argv)) +
                "。修法：检查该算子的调用点是否少传了参数")
        v = argv[idx]
        if kind == "ptr":
            buf = getattr(v, "_gpu", None)
            if buf is None:
                raise ValueError(
                    "ptr:%d 对应的实参没有设备缓冲" % idx +
                    "。修法：ptr 只能用于 CUDA 张量；CPU 请用 op_c 直接传 numpy 指针")
            out.append(buf.ptr)
        elif kind == "f32":
            out.append(float(v))
        elif kind == "f64":
            out.append(float(v))
        elif kind == "int":
            out.append(int(v))
        elif kind == "raw":
            out.append(v)
        else:
            raise ValueError(
                "未知参数类型 %r" % kind +
                "。修法：只支持 ptr / f32 / f64 / int / raw")
    return out


def load_manifest(path, *, replace=False):
    """加载一份 JSON 清单，把里面的原生算子全部绑定好。返回清单内容。

    解析与绑定只发生一次；之后调用是纯原生路径。
    """
    from . import plugin as P

    p = os.path.abspath(str(path))
    if not os.path.isfile(p):
        raise P.PluginError(
            "插件清单不存在: %s" % p +
            "。修法：检查路径；清单是 .json 文件")
    try:
        with open(p, encoding="utf-8") as f:
            info = json.load(f)
    except json.JSONDecodeError as exc:
        raise P.PluginError(
            "插件清单不是合法 JSON: %s (%s)" % (p, exc) +
            "。修法：用 python -m json.tool 校验格式") from exc

    for key in ("name", "api_version", "library"):
        if not info.get(key):
            raise P.PluginError(
                "插件清单缺少字段 %r: %s" % (key, p) +
                "。修法：至少要有 name / api_version / library")

    base = os.path.dirname(p)
    lib_path = info["library"]
    if not os.path.isabs(lib_path):
        lib_path = os.path.join(base, lib_path)
    lib = P.load_lib(lib_path, replace=replace)

    for spec in info.get("ops") or []:
        op_name = spec.get("name")
        symbol = spec.get("symbol")
        if not op_name or not symbol:
            raise P.PluginError(
                "清单里的算子缺少 name 或 symbol: %r" % (spec,) +
                "。修法：每个算子要写 {\"name\": ..., \"symbol\": ...}")
        restype = {"void": None, "int": __import__("ctypes").c_int,
                   "double": __import__("ctypes").c_double}.get(
            str(spec.get("restype", "void")).lower())
        args_spec = spec.get("args") or []
        P.op_c(op_name, lib, symbol,
               args=(lambda *a, _s=args_spec: parse_args_spec(_s, a)),
               restype=restype,
               doc=spec.get("doc", ""))
    return info
