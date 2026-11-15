# -*- coding: utf-8 -*-
"""插件与插件包的**统一元信息规范**（v1）。

## 为什么要有这份规范

插件现在长在三条不同的轨道上：

| 轨道 | 入口 | 已有元信息 |
| --- | --- | --- |
| 组合式 / 带反向的 Python 算子 | `ops_plugin.register(...)` | 只有 `kind` 和 `doc` |
| 原生库 + JSON 声明 | `plugin_manifest.load_manifest` | `name / version / api_version / library / ops` |
| 算子插件包（zip） | `ops_bundle.load` | `schema / name / arity / backends` |

三套字段各写各的，导致：**装了什么、什么版本、提供哪些后端、能不能在别的后端跑，用户看不出来。**
本模块给出**一份 schema**，三处共用；缺字段的按默认值补，非法值**在注册阶段就报错**。

## 与 DSL 的关系（这是为后面铺的路）

`PLUGIN_DSL.md` 定了三档可移植性：

| 档 | 形态 | 元信息里怎么写 |
| --- | --- | --- |
| **A 组合式** | 用已有算子拼（Python）| `form: "composed"`，`impl.python` |
| **B 声明式内核（DSL）** | 受限表达式，ORV 生成各后端代码 | `form: "dsl"`，`impl.expr` + `impl.inputs` |
| **C 后端原生** | 手写 .cu/.c/SPIR-V | `form: "native"`，`impl.backends.{cuda,c,...}` |

**纪律**（沿用 `PLUGIN_DSL.md`）：C 档必须同时给 A 或 B 档的回退，
否则该插件在别的后端上是"不存在"，不是"跑出错结果"。本模块把这条做成**校验项**。

## 规范（v1）

```json
{
  "schema": 1,
  "name": "svs-ops",                  // ^[a-z][a-z0-9-]{1,63}$
  "version": "0.2.0",                 // 语义化版本
  "api_version": "1.0",               // 兼容 ORV 的插件 API
  "summary": "虚拟歌手音频算子",        // 一句话，<=120 字
  "description": "...",               // 可选，长描述
  "author": "JXW",                    // 可选
  "license": "MIT",                   // 可选
  "homepage": "https://...",          // 可选
  "tags": ["audio", "stft"],          // 可选
  "requires": {"orv": ">=0.9.65"},    // 可选，ORV 版本下限
  "ops": [
    {
      "name": "stft_mag",
      "arity": 1,
      "doc": "…",
      "form": "composed",             // composed | dsl | native
      "impl": {"python": "stft_mag"}  // 依 form 而定
    }
  ],
  "backends": ["cpu", "cuda"]         // 这个插件实际能跑的后端
}
```

插件包（`.tgop`，zip）就是在根目录放这份 `manifest.json`，再放各实现文件。
"""

import json
import os
import re

#: 当前规范版本
SCHEMA = 1

#: 允许的后端名
BACKENDS = ("cpu", "cuda", "hip", "musa", "vulkan", "metal")

#: 允许的实现形态（与 PLUGIN_DSL.md 的三档一一对应）
FORMS = ("composed", "dsl", "native")

NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,63}$")        # 包名：可做文件名/CLI
OP_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{1,63}$")   # 算子名：要成为 Python 属性，允许下划线
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+][0-9A-Za-z.-]+)?$")


class PluginMetaError(ValueError):
    """元信息不合法。消息里一定带「修法」。"""


def parse_semver(v):
    m = SEMVER_RE.match(str(v))
    if not m:
        return None
    return tuple(int(x) for x in m.groups())


def default_meta(name, **kw):
    """给一个名字，补出最小合法元信息（其余字段走默认）。"""
    meta = {
        "schema": SCHEMA,
        "name": name,
        "version": "0.0.0",
        "api_version": "1.0",
        "summary": "",
        "description": "",
        "author": "",
        "license": "",
        "homepage": "",
        "tags": [],
        "requires": {},
        "ops": [],
        "backends": ["cpu"],
    }
    meta.update(kw)
    return meta


def validate(meta):
    """校验元信息，返回**规范化后的副本**；不合法就抛 `PluginMetaError`。

    严格程度：只对"会让人踩坑"的字段强制（name/version/schema/form/fallback），
    其余可缺省 —— 规范要能落地，不能变成填表负担。
    """
    if not isinstance(meta, dict):
        raise PluginMetaError("元信息必须是 dict，收到 %s。修法：传 manifest 解析出的字典"
                              % type(meta).__name__)
    out = default_meta(str(meta.get("name") or ""))
    out.update({k: v for k, v in meta.items() if v is not None})

    if int(out.get("schema") or 0) != SCHEMA:
        raise PluginMetaError(
            "schema 只支持 %d，收到 %r。修法：把 manifest 的 schema 改成 %d（或升级 ORV）"
            % (SCHEMA, out.get("schema"), SCHEMA))

    name = out["name"]
    if not NAME_RE.match(name or ""):
        raise PluginMetaError(
            "插件名 %r 不合法。修法：用小写字母开头、只含小写字母/数字/连字符，长度 2–64，例如 my-audio-ops"
            % (name,))

    if parse_semver(out.get("version")) is None:
        raise PluginMetaError(
            "版本号 %r 不是语义化版本。修法：写成 MAJOR.MINOR.PATCH，例如 0.2.0"
            % (out.get("version"),))

    bs = out.get("backends") or []
    if not isinstance(bs, (list, tuple)):
        raise PluginMetaError("backends 必须是列表。修法：写成 [\"cpu\", \"cuda\"]")
    bad = [b for b in bs if b not in BACKENDS]
    if bad:
        raise PluginMetaError(
            "未知后端 %s。修法：只能用 %s" % (bad, list(BACKENDS)))
    if not bs:
        raise PluginMetaError("backends 不能为空。修法：至少写 [\"cpu\"]")

    ops = out.get("ops") or []
    if not isinstance(ops, list):
        raise PluginMetaError("ops 必须是列表。修法：每个算子一项")
    seen = set()
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            raise PluginMetaError("ops[%d] 必须是 dict。修法：{\"name\": ..., \"form\": ...}" % i)
        nm = op.get("name")
        if not nm or not OP_NAME_RE.match(str(nm)):
            raise PluginMetaError(
                "ops[%d].name=%r 不合法。修法：小写字母或下划线开头、只含小写字母/数字/下划线，长度 2–64" % (i, nm))
        if nm in seen:
            raise PluginMetaError("算子 %r 重复声明。修法：同一 manifest 里不要重复" % (nm,))
        seen.add(nm)
        form = op.get("form")
        if form not in FORMS:
            raise PluginMetaError(
                "算子 %r 的 form=%r 不合法。修法：用 %s 之一（composed=组合式、dsl=声明式、native=后端原生）"
                % (nm, form, list(FORMS)))
        impl = op.get("impl") or {}
        if not isinstance(impl, dict) or not impl:
            raise PluginMetaError(
                "算子 %r 缺少 impl。修法：composed 给 {\"python\": \"函数名\"}；"
                "dsl 给 {\"expr\": \"...\", \"inputs\": [...]}；native 给 {\"backends\": {...}}" % (nm,))

    # 纪律：C 档（native）必须同时有 A 或 B 档回退，否则别的后端上等于不存在
    has_fallback = any(o.get("form") in ("composed", "dsl") for o in ops) or \
        bool(out.get("fallback"))
    for op in ops:
        if op.get("form") == "native" and not has_fallback:
            raise PluginMetaError(
                "算子 %r 是 native（只在一个后端可用），但整个插件没有 composed/dsl 回退。"
                "修法：按 PLUGIN_DSL.md 的纪律，再提供一个组合式或 DSL 版实现"
                "（在 ops 里加一项，或给顶层 fallback）" % (op.get("name"),))
    return out


def load_manifest_file(path):
    """读并校验一个 manifest.json，返回规范化元信息。"""
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return validate(meta)


def summary_line(meta):
    """一行摘要，给 `T.plugin.summary()` 之类的界面用。"""
    return "%-24s %-8s %-6s %s" % (
        meta["name"], meta["version"], ",".join(meta["backends"]),
        (meta.get("summary") or "")[:60])


def install(path, *, replace=False, verbose=False):
    """**一行装完整个插件包**。路径可以是 zip（.tgop/.zip）或含 manifest.json 的目录。

    返回规范化后的元信息 dict。安装方式按每个算子的 orm 分派：

    * composed / dsl：从包里取 Python 实现文件，按 impl.python 找到函数并注册
    * 
ative：交给 plugin_manifest.load_manifest（它读的是原生库 + JSON 声明）

    包里所有实现文件都先解到临时目录再加载 —— 这样 zip 与目录两种形态共用同一条路径。
    """
    import json
    import tempfile
    import zipfile

    from tgai_go.train import ops_plugin
    from tgai_go.train import plugin as _plugin

    tmp = None
    try:
        if os.path.isdir(path):
            root = path
        elif zipfile.is_zipfile(path):
            tmp = tempfile.mkdtemp(prefix="tgop_")
            with zipfile.ZipFile(path) as z:
                z.extractall(tmp)
            root = tmp
        else:
            raise PluginMetaError("既不是目录也不是 zip：%s。修法：给 .tgop（zip）或含 manifest.json 的目录" % (path,))

        mpath = os.path.join(root, "manifest.json")
        if not os.path.exists(mpath):
            raise PluginMetaError(
                "包里没有 manifest.json（%s）。修法：插件包根目录必须放一份 manifest.json" % (root,))
        meta = load_manifest_file(mpath)

        installed = []
        for op in meta["ops"]:
            form = op["form"]
            impl = op["impl"]
            if form in ("composed", "dsl"):
                mod_name = impl.get("module")
                fn_name = impl.get("python")
                if not mod_name or not fn_name:
                    raise PluginMetaError(
                        "算子 %r 的 impl 缺少 module/python。修法：写成 "
                        '{"module": "ops.py", "python": "函数名"}' % (op["name"],))
                fn = _load_callable(os.path.join(root, mod_name), fn_name)
                ops_plugin.register(op["name"], fn, kind=op.get("kind", "functional"),
                                    doc=op.get("doc", ""), replace=replace)
                installed.append(op["name"])
            elif form == "native":
                lib = impl.get("library") or impl.get("backends", {}).get("cuda", {}).get("source")
                if not lib:
                    raise PluginMetaError(
                        "算子 %r 是 native 但没给 library/source。修法：补 impl.library" % (op["name"],))
                _plugin.load_manifest(os.path.join(root, lib), replace=replace) \
                    if os.path.exists(os.path.join(root, lib)) else None
                installed.append(op["name"])

        # 若 manifest 声明了 namespace，就把它装成 `T.<namespace>`，一行导入即可用
        ns_name = meta.get("namespace")
        if ns_name:
            import types

            import tgai_go.train as _T
            members = {}
            for op in meta["ops"]:
                ent = ops_plugin._REGISTRY.get(op["name"]) or {}
                fn = ent.get("fn") if isinstance(ent, dict) else getattr(ent, "fn", None)
                if fn is not None:
                    members[op["name"]] = fn
            setattr(_T, ns_name, types.SimpleNamespace(**members))
        # 记录已装清单，供 T.plugin.list() / info() / uninstall() 使用
        _INSTALLED[meta["name"]] = {
            "meta": meta,
            "ops": installed,
            "namespace": meta.get("namespace"),
        }
        if verbose:
            print("已安装 %s %s：%s" % (meta["name"], meta["version"], ", ".join(installed)))
        return meta
    finally:
        if tmp:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


def _load_callable(pyfile, fn_name):
    """从包里某个 .py 加载一个可调用对象（不污染 sys.path）。"""
    import importlib.util
    if not os.path.exists(pyfile):
        raise PluginMetaError("包里找不到实现文件 %s。修法：manifest 里的路径要相对包根" % (pyfile,))
    spec = importlib.util.spec_from_file_location("tgop_impl_" + str(abs(hash(pyfile))), pyfile)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise PluginMetaError(
            "实现文件 %s 里没有 %r。修法：manifest 的 impl.python 要写文件里真实存在的函数名"
            % (os.path.basename(pyfile), fn_name))
    return fn


#: 已安装的插件包：name -> {meta, ops, namespace}
_INSTALLED = {}


def list_plugins():
    """列出已装的插件包，返回 list[dict]（name/version/backends/ops_count/namespace）。"""
    out = []
    for name, rec in sorted(_INSTALLED.items()):
        m = rec["meta"]
        out.append({
            "name": name,
            "version": m.get("version"),
            "summary": m.get("summary", ""),
            "backends": list(m.get("backends") or []),
            "ops_count": len(rec["ops"]),
            "namespace": rec.get("namespace"),
        })
    return out


def info(name):
    """看某个插件包的详细信息（元信息 + 它装了哪些算子）。"""
    rec = _INSTALLED.get(name)
    if rec is None:
        raise PluginMetaError(
            "没有装过插件包 %r。修法：先 T.plugin.install(路径)；已装的有 %s"
            % (name, [r["name"] for r in _INSTALLED.values()] or "（空）"))
    m = dict(rec["meta"])
    m["ops"] = list(rec["ops"])
    m["namespace"] = rec.get("namespace")
    return m


def uninstall(name):
    """卸载插件包：注销它装过的算子，并移除 T.<namespace>。

    返回被注销的算子名列表。没装过就报错（带已装列表）。
    """
    from tgai_go.train import ops_plugin
    import tgai_go.train as _T

    rec = _INSTALLED.get(name)
    if rec is None:
        raise PluginMetaError(
            "没有装过插件包 %r。修法：T.plugin.list() 看已装的有哪些" % (name,))
    removed = []
    for op in rec["ops"]:
        try:
            ops_plugin.unregister(op)
            removed.append(op)
        except Exception:      # noqa: BLE001  已被别的插件覆盖/手动注销
            pass
    ns = rec.get("namespace")
    if ns and hasattr(_T, ns):
        delattr(_T, ns)
    del _INSTALLED[name]
    return removed
