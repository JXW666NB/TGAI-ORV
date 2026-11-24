# -*- coding: utf-8 -*-
"""L0 清单 + L1 调用示例 —— `T.help` 热更新 Wiki 系统的前两层。

四层模型（详见 `docs/HELP_SYSTEM.md`）：

* **L0**：引擎运行时自省出来的"有什么"（`op / api / kernel / cli / env / plugin`）。
  只读 `dir()` / `inspect` / 源码（**不初始化 CUDA**），天然最新。
* **L1**：由 L0 的签名**机械生成**的调用示例（参数怎么填）。引擎现算，
  词条作者写错签名也误导不了用户。
* **L2**：外部词条文案（`wiki/*.txt`），由 `help_wiki.py` 加载。
* **L3**：渲染（终端 / JSON / 离线网页），渲染器都是惰性导入。

本模块**不在引擎启动路径上**：只有调用 `T.help(...)` 的 JSON / Web / check /
missing / contrib 时才会被导入。因此这里可以放心用 `ast` / `difflib` 等标准库。

对外接口 / API
--------------
    catalog()        -> list[dict]   全部 L0 条目（含 L1 示例 + 可达性自证字段）
    entry(item_id)   -> dict | None
    nearest(name, n) -> list[str]    最接近的候选 ID（给报错用）
    default_category(kind)           contrib 骨架用的分类默认值
    device_source_kernel_names() -> set[str]  NVRTC 真正编译的内核名
    invalidate()                     清空 L0 缓存（T.help.refresh 会调）

**每条的"可达性自证"字段**（2026-11-21 加，起因：`T.help('op')` 曾把
`t.mul(x)` 写成"调用方式"，而 Tensor 上根本没有 `.mul()` —— 文档在骗人）：

    reachable      : bool   引擎里真的能调到吗
    form           : str    真实可用的写法（`x.sum()` / `a * b` / `T.matmul(a, b)` /
                            `rt.launch("k", ...)` / `python -m ... --flag` / `TG_X=1`）
    form_kind      : str    method / operator / class / function / kernel / cli / env
    reach_evidence : str    证据（`Tensor.sum 存在` / `cuda.py:93 os.environ.get` …）
    unreachable    : str    不可调用时的原因（可达时为空串）

判定全部**实算**，不写死：op 查 `Tensor` 属性、api 查 `getattr(T, 名字)`、
kernel 查 `cuda.py::_device_source()`（NVRTC 编译集合，静态重建 AST 不 import 模块）、
cli 查 `add_argument` + `__main__` 入口、env 查真实的 `os.environ` 读取点。
"""

import ast
import difflib
import inspect
import os
import re
import types

__all__ = ["catalog", "entry", "nearest", "default_category", "invalidate",
           "device_source_kernel_names", "optional_source_kernel_names",
    "KIND_ORDER", "KINDS"]

#: kind 轴的规范顺序（也是 JSON / 网页里的展示顺序）。
KIND_ORDER = ("op", "api", "kernel", "cli", "env", "plugin", "fmt")
KINDS = frozenset(KIND_ORDER)

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = None

#: op 的"运算符条目"：Tensor 没有 `.mul()` 方法，乘法就是 `x * y`。
#: 这里给出**真实存在**的 dunder 与机械示例 —— 绝不编造不存在的调用方式。
#: 每个元素是 (名字, dunder, 真实调用形式, 备注)；备注是给用户看的边界说明。
_OPERATORS = (
    ("add", "__add__", "a + b", "逐元素加；Tensor 没有 .add() 方法"),
    ("sub", "__sub__", "a - b", "逐元素减；Tensor 没有 .sub() 方法"),
    ("mul", "__mul__", "a * b", "逐元素乘；Tensor 没有 .mul() 方法，顶层也没有 T.mul"),
    ("div", "__truediv__", "a / b", "逐元素除；Tensor 没有 .div() 方法"),
    ("matmul", "__matmul__", "a @ b", "矩阵乘；也可用 T.matmul(a, b)，但没有 .matmul() 方法"),
    # pow 只支持**标量**指数：a ** 2 可以；a ** b（张量指数）会 TypeError（实测）。
    ("pow", "__pow__", "a ** 2", "标量指数；张量指数 a ** b 不支持（会 TypeError）"),
)

#: 顶层惰性导出（`__init__.py` 里用 `__getattr__` 实现，避免拖慢引擎启动）。
_LAZY_API = ("DUO", "DuoSession", "DuoResult", "DuoRefused")

#: kind -> contrib 骨架里的默认分类（作者可改）。
_DEFAULT_CATEGORY = {
    "op": "tensor",
    "api": "other",
    "kernel": "compute",
    "cli": "getting-started",
    "env": "debug",
    "plugin": "plugin",
}


# --------------------------------------------------------------------- 工具
def default_category(kind):
    """contrib 生成骨架时填的分类默认值（机械猜测，作者可改）。"""
    return _DEFAULT_CATEGORY.get(kind, "other")


def _signature_of(obj):
    try:
        return str(inspect.signature(obj))
    except Exception:
        return "(...)"


def _call_suffix(sig, drop_self=False):
    """按签名机械生成 `(必填参数...)` —— 可选参数省略，示例保持最短可运行。

    这是 L1 的核心：**参数怎么填**。它永远和真实签名一起，不会过期。
    """
    try:
        params = list(sig.parameters.values())
    except Exception:
        return "(...)"
    if drop_self and params and params[0].name in ("self", "cls"):
        params = params[1:]
    parts = []
    for p in params:
        if p.kind == p.VAR_POSITIONAL:
            parts.append("*" + p.name)
        elif p.kind == p.VAR_KEYWORD:
            continue
        elif p.default is p.empty:
            parts.append(p.name)
        # 有默认值的参数：省略（示例保持最短可运行；完整签名在 usage 里）
    return "(" + ", ".join(parts) + ")"


# ------------------------------------------------------------------ op 轴
def _tensor_class():
    try:
        import tgai_go.train as T
        return T.Tensor
    except Exception:
        return None


def _op_entries():
    cls = _tensor_class()
    out = []
    if cls is not None:
        for name in sorted(dir(cls)):
            if name.startswith("_"):
                continue
            fn = getattr(cls, name, None)
            if not callable(fn):
                continue
            try:
                sig = inspect.signature(fn)
            except Exception:
                sig = None
            sig_text = str(sig) if sig is not None else "(...)"
            suffix = _call_suffix(sig, drop_self=True) if sig is not None else "(...)"
            example = "x." + name + suffix
            # 自证可达：只有 `Tensor.<名字>` 真的存在且可调用，才把它写成方法形式。
            reachable = callable(getattr(cls, name, None))
            out.append({
                "id": "op:" + name,
                "kind": "op",
                "name": name,
                "signature": sig_text,
                "example": example if reachable else "",
                "module": "tgai_go.train.Tensor",
                "subkind": "",
                "form": example if reachable else "",
                "form_kind": "method",
                "reachable": reachable,
                "reach_evidence": "Tensor.%s 存在且可调用" % name,
                "unreachable": "" if reachable else "Tensor 上没有可调用的 .%s（不应出现在清单里）" % name,
            })
    # 运算符：Tensor 上没有 .mul()/.matmul() 方法，但 `x * y` / `x @ y` 是真的可用。
    # 可达性由 dunder 是否存在证明；form 是**运算符写法**而不是方法名。
    for name, dunder, form, note in _OPERATORS:
        if cls is None or not hasattr(cls, dunder):
            continue
        try:
            sig_text = str(inspect.signature(getattr(cls, dunder)))
        except Exception:
            sig_text = "(self, other)"
        out.append({
            "id": "op:" + name,
            "kind": "op",
            "name": name,
            "signature": sig_text,
            "example": form,
            "module": "tgai_go.train.Tensor (" + dunder + ")",
            "subkind": "operator",
            "form": form,
            "form_kind": "operator",
            "reachable": True,
            "reach_evidence": "Tensor.%s 存在（运算符写法 %s）" % (dunder, form),
            "unreachable": "",
            "note": note,
        })
    return out



# ----------------------------------------------------------------- api 轴
def _top_level_api_entries():
    try:
        import tgai_go.train as T
    except Exception:
        return []
    names = set(n for n in dir(T) if not n.startswith("_"))
    for extra in _LAZY_API:                 # 惰性导出：只有帮助系统会去触碰
        try:
            if hasattr(T, extra):
                names.add(extra)
        except Exception:
            pass
    out = []
    for name in sorted(names):
        try:
            obj = getattr(T, name)
        except Exception:
            continue
        if not isinstance(obj, (types.FunctionType, types.BuiltinFunctionType, type)):
            continue
        mod = getattr(obj, "__module__", "") or ""
        if not mod.startswith("tgai_go.train"):
            continue
        try:
            sig = inspect.signature(obj)      # 类拿到的是构造函数签名（不含 self）
        except Exception:
            sig = None
        subkind = "optimizer" if "opt" in mod.split(".")[-1] or name in (
            "AdamW", "SGD", "Adafactor", "LAMB", "Lion", "Muon", "Accumulator") else ""
        example = "T." + name + (_call_suffix(sig) if sig is not None else "(...)")
        is_class = isinstance(obj, type)
        # 自证可达：`getattr(T, 名字)` 真的拿得到、且可调用（类=构造调用）。
        reachable = callable(obj)
        out.append({
            "id": "api:" + name,
            "kind": "api",
            "name": name,
            "signature": _signature_of(obj),
            "example": example if reachable else "",
            "module": mod,
            "subkind": subkind,
            "form": example if reachable else "",
            "form_kind": "class" if is_class else "function",
            "reachable": reachable,
            "reach_evidence": "T.%s 存在且可调用（%s）"
                              % (name, "类/构造" if is_class else "函数"),
            "unreachable": "" if reachable else "T.%s 存在但不可调用" % name,
        })
    return out


# ------------------------------------------------- 推理侧（GO：ORV 是推训一体）
#: 推理侧条目的 ID 前缀。`tgai_go` 与 `tgai_go.train` 有同名导出（`load` / `Model` /
#: `Params`），所以推理条目用三段式 `api:go:<名字>` 消歧；kind 仍然是 `api`。
_INFER_PREFIX = "go:"


def _inference_api_entries():
    """GO 推理侧的顶层 API —— **ORV 是推训一体，`tgai_go` 包本身就是推理库**。

    为什么必须单列（2026-11-21 用户指出）：L0 之前只扫 `tgai_go.train`，于是整个推理面
    在帮助系统里**完全隐形** —— 用户看不到"训练完之后怎么跑"，而 Wiki 里写
    `api:TGEngine` 会被加载器判成"未知 ID"直接拒收。补上这一组，Wiki 才能真正
    覆盖「训练完 → 用起来」的闭环。
    """
    try:
        import tgai_go as G
    except Exception:                            # noqa: BLE001  推理侧不可用时不要拖垮帮助
        return []
    out = []
    for name in sorted(n for n in dir(G) if not n.startswith("_")):
        try:
            obj = getattr(G, name)
        except Exception:                        # noqa: BLE001
            continue
        if not isinstance(obj, (types.FunctionType, types.BuiltinFunctionType, type)):
            continue
        mod = getattr(obj, "__module__", "") or ""
        if not mod.startswith("tgai_go") or mod.startswith("tgai_go.train"):
            continue                             # 训练侧已由上面那组覆盖
        try:
            sig = inspect.signature(obj)
        except Exception:                        # noqa: BLE001
            sig = None
        is_class = isinstance(obj, type)
        example = "tgai_go." + name + (_call_suffix(sig) if sig is not None else "(...)")
        reachable = callable(obj)
        out.append({
            "id": "api:" + _INFER_PREFIX + name,
            "kind": "api",
            "name": name,
            "signature": _signature_of(obj),
            "example": example if reachable else "",
            "module": mod,
            "subkind": "inference",
            "form": example if reachable else "",
            "form_kind": "class" if is_class else "function",
            "reachable": reachable,
            "reach_evidence": "tgai_go.%s 存在且可调用（%s，模块 %s）"
                              % (name, "类/构造" if is_class else "函数", mod),
            "unreachable": "" if reachable else "tgai_go.%s 存在但不可调用" % name,
        })
    return out


def _inference_cli_entries():
    """`python -m tgai_go <cmd>` 的条目。

    子命令**从源码机械提取**：`__main__.py` 里 `add_argument('cmd', choices=[...])`
    的 choices 列表。解析不到就返回空 —— 绝不硬编码一份可能过期的列表。
    """
    path = os.path.join(os.path.dirname(_HERE), "__main__.py")
    try:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
    except Exception:                            # noqa: BLE001
        return []
    choices, line_of = [], 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "add_argument"):
            continue
        for kw in node.keywords:
            if kw.arg != "choices" or not isinstance(kw.value, ast.List):
                continue
            names = [e.value for e in kw.value.elts
                     if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if names:
                choices = names
                line_of = getattr(kw.value, "lineno", 0)
    out = []
    for cmd in choices:
        if cmd == "version":                 # `version` 不需要模型，别生成假参数
            example = "python -m tgai_go version"
        else:
            extras = ' "问题"' if cmd == "gen" else ""
            example = "python -m tgai_go %s -m <model.TG>%s" % (cmd, extras)
        out.append({
            "id": "cli:" + _INFER_PREFIX + cmd,
            "kind": "cli",
            "name": cmd,
            "signature": "tgai_go CLI 子命令",
            "example": example,
            "module": "tgai_go.__main__",
            "subkind": "inference",
            "form": example,
            "form_kind": "cli",
            "reachable": True,
            "reach_evidence": "__main__.py 第 %d 行的 argparse choices 里有 %r"
                              % (line_of, cmd),
            "unreachable": "",
        })
    return out


# --------------------------------------------------------- 文件格式轴（fmt）
#: `.TG` / `.TG4` 容器。**证据只能来自源码**（magic、头结构、dtype 表都是常量），
#: 所以这里解析 `read_tg.py`，解析不到就不产出条目 —— 绝不硬编码一份可能过期的说明。
_FMT_SPECS = (
    ("TG", "TG_MAGIC", ".TG（v3 家族容器）", "read_tg.read_tg()"),
    ("TG4", "TG4_MAGIC", ".TG4（自描述分区容器）", "read_tg._read_tg4()"),
)


def _fmt_entries():
    """文件格式轴：模型容器 `.TG` / `.TG4` 是什么、有哪些头字段、dtype 码怎么对应。

    ★ 2026-11-21：用户问「有没有 `.TG` 的介绍」—— 查下来 `docs/` 里有
    （`TG_V4_FORMAT.md`、`TG对话配置元数据`），但 **Wiki 和 `T.help` 里一条都没有**。
    `.TG` 是训练产物**同时也是推理输入**，它比任何单个 API 都更值得有一条词条。
    """
    path = os.path.join(os.path.dirname(_HERE), "read_tg.py")
    try:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
    except Exception:                            # noqa: BLE001
        return []

    consts = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        val = node.value
        if isinstance(val, ast.Constant) and isinstance(val.value, (int, str)):
            consts[tgt.id] = val.value

    # dtype 码表（0 fp32 / 1 fp16 / 2 int8 / 3 int32 / 4 q8_0 / 5 q4_0 / 6 bf16）
    dtypes = []
    m = re.search(r"^DT\s*=\s*\{([^}]*)\}", src, re.S | re.M)
    if m:
        for code, name in re.findall(r'(\d+)\s*:\s*\(\s*"([a-z0-9_]+)"', m.group(1)):
            dtypes.append("%s=%s" % (code, name))
    header_struct = ""
    ms = re.search(r'struct\.unpack\("<([^"]+)"', src)
    if ms:
        header_struct = ms.group(1)

    out = []
    for short, const_name, title, reader in _FMT_SPECS:
        magic = consts.get(const_name)
        if not isinstance(magic, int):
            continue                             # 源码里没这个常量 ⇒ 不产出（不自证不写）
        detail = "%s；magic=0x%08X" % (title, magic)
        if short == "TG" and header_struct:
            detail += "；头 = `struct <%s`；dtype 码：%s" % (
                header_struct, "、".join(dtypes) if dtypes else "见源码")
        ev = "read_tg.py 里 `%s = 0x%08X`，且 %s 负责解析" % (const_name, magic, reader)
        out.append({
            "id": "fmt:" + short,
            "kind": "fmt",
            "name": short,
            "signature": detail,
            "example": ".TG 文件（加载方式见 api:go:load）",
            "module": "tgai_go.read_tg",
            "subkind": "container",
            "form": ".TG 文件（用 tgai_go.load(path) 加载）",
            "form_kind": "format",
            "reachable": True,
            "reach_evidence": ev,
            "unreachable": "",
        })
    return out


# -------------------------------------------------------------- kernel 轴
def _module_string_constants(tree):
    """模块顶层的 `名字 = "字符串"` 常量表（cuda.py 的内核源码都是这种写法）。"""
    consts = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            continue
        if isinstance(value, str):
            consts[node.targets[0].id] = value
    return consts


def _eval_concat(node, consts):
    """把 `A + "\\n" + B` 这样的字符串拼接表达式静态求值（求不出返回 None）。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _eval_concat(node.left, consts)
        right = _eval_concat(node.right, consts)
        if left is not None and right is not None:
            return left + right
    return None


def _cuda_device_source():
    """**静态重建** `cuda.py::_device_source()`（NVRTC 真正编译的那份源码）。

    不 import cuda 模块（import 它要 0.3 s 量级，帮助系统不该为一次自证付这个成本），
    而是用 AST 求出同一份常量拼接；`catalog()` 的内核可达性就以此为准。
    返回 None 表示结构变了、求不出来 —— 调用方会降级并把证据写清楚。
    """
    path = os.path.join(_HERE, "cuda.py")
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
    except Exception:
        return None
    consts = _module_string_constants(tree)
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name == "_device_source"):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                return _eval_concat(stmt.value, consts)
    return None


_DEVICE_SOURCE_NAMES = None


def device_source_kernel_names():
    """`_device_source()` 里 `extern "C" __global__ void` 的内核名集合（缓存）。

    集合里的名字一定能被 `rt.launch(名字, ...)` 解析到（NVRTC 编译进模块）。
    """
    global _DEVICE_SOURCE_NAMES
    if _DEVICE_SOURCE_NAMES is None:
        text = _cuda_device_source()
        if text is None:
            _DEVICE_SOURCE_NAMES = set()
        else:
            _DEVICE_SOURCE_NAMES = set(
                re.findall(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(', text))
    return _DEVICE_SOURCE_NAMES


_OPTIONAL_NAMES = None


def _cuda_optional_sources():
    """静态求出 `cuda.py::_OPTIONAL_KERNEL_SOURCES` 登记的源码常量拼接结果。"""
    path = os.path.join(_HERE, "cuda.py")
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
    except Exception:
        return ""
    consts = _module_string_constants(tree)
    names = ()
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_OPTIONAL_KERNEL_SOURCES"):
            try:
                names = ast.literal_eval(node.value) or ()
            except Exception:
                names = ()
    return "".join(consts.get(n, "") for n in names)


def optional_source_kernel_names():
    """可选 NVRTC 模块（如 `_FUSE_BCAST_KERNELS`）里的内核名集合（缓存）。

    这些内核**默认不编译**，只在对应开关开启时才加载第二个模块；
    自省/可达性必须把它们与主模块一起看，否则会误判成"声明了但没编译"。
    """
    global _OPTIONAL_NAMES
    if _OPTIONAL_NAMES is None:
        _OPTIONAL_NAMES = set(re.findall(
            r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(', _cuda_optional_sources()))
    return _OPTIONAL_NAMES


def _kernel_entries():
    try:
        from .apilist import _kernels
    except Exception:
        return []
    try:
        rows = _kernels()
    except Exception:
        return []
    compiled = device_source_kernel_names() | optional_source_kernel_names()
    out = []
    for it in rows:
        name = it.get("name")
        if not name:
            continue
        params = it.get("params") or ""
        sig = "(" + params + ")" if params else "(...)"
        form = 'rt.launch("%s", grid, block, [...])' % name
        # 自证可达：内核名必须出现在 `_device_source()`（NVRTC 编译集合）里，
        # 否则源码里有 __global__ 定义、运行时也找不到它。
        if compiled:
            reachable = name in compiled
            evidence = ("_device_source() 含该内核（NVRTC 编译进模块，rt.launch 可解析）"
                        if reachable else
                        "_device_source() 里找不到该内核（源码声明了但没进编译集合）")
        else:
            # 静态重建失败（cuda.py 结构变了）：退回"源码里声明过"，并把降级说清楚。
            reachable = True
            evidence = "cuda.py 声明了 __global__（未能静态重建 _device_source，按源码声明计）"
        out.append({
            "id": "kernel:" + name,
            "kind": "kernel",
            "name": name,
            "signature": sig,
            "example": form if reachable else "",
            "module": "tgai_go.train.cuda",
            "subkind": "",
            "form": form if reachable else "",
            "form_kind": "kernel",
            "reachable": reachable,
            "reach_evidence": evidence,
            "unreachable": "" if reachable else
                           "cuda.py 有 __global__ 定义，但不在 _device_source() 里 —— 别调它",
        })
    return out



# ----------------------------------------------------------------- cli 轴
_CLI_FILES = (
    ("tgai_go.train.v3_pretrain", "v3_pretrain.py"),
    ("tgai_go.train.v3_train_cli", "v3_train_cli.py"),
)


def _node_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _has_main_guard(tree):
    """模块里有没有 `if __name__ == "__main__":`（没有就不能 `python -m` 跑）。"""
    def is_main_name(node):
        return isinstance(node, ast.Name) and node.id == "__name__"

    def is_main_str(node):
        return isinstance(node, ast.Constant) and node.value == "__main__"

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)):
            continue
        left, right = test.left, test.comparators[0]
        if (is_main_name(left) and is_main_str(right)) or \
                (is_main_name(right) and is_main_str(left)):
            return True
    return False


def _cli_entries():
    out = []
    seen = set()
    for modname, filename in _CLI_FILES:
        path = os.path.join(_HERE, filename)
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                tree = ast.parse(fh.read())
        except Exception:
            continue
        # 自证可达：flag 来自真实 `add_argument("--x")`，且模块有 `__main__` 才能真正跑起来。
        has_main = _has_main_guard(tree)
        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "add_argument"):
                continue
            if not node.args:
                continue
            arg0 = node.args[0]
            if not (isinstance(arg0, ast.Constant) and isinstance(arg0.value, str)):
                continue
            flag = arg0.value
            if not flag.startswith("--") or flag in seen:
                continue
            seen.add(flag)
            info = {"help": "", "type": "", "action": "", "default": None,
                    "choices": None, "metavar": ""}
            for kw in node.keywords:
                if kw.arg in ("help", "default", "action", "choices", "metavar"):
                    try:
                        info[kw.arg] = ast.literal_eval(kw.value)
                    except Exception:
                        if kw.arg == "default":
                            info[kw.arg] = None
                elif kw.arg == "type":
                    info["type"] = _node_name(kw.value) or ""
            calls.append((flag, info))
        for flag, info in calls:
            action = info.get("action") or ""
            typ = info.get("type") or ""
            sig_parts = []
            if action in ("store_true", "store_false"):
                sig_parts.append("开关 / flag")
            elif typ:
                sig_parts.append(typ)
            else:
                sig_parts.append("值 / value")
            if info.get("metavar"):
                sig_parts.append("metavar=%s" % info["metavar"])
            if info.get("default") not in (None, ""):
                sig_parts.append("默认 %r" % (info["default"],))
            if info.get("choices"):
                sig_parts.append("可选 %s" % "/".join(str(c) for c in info["choices"]))
            if typ in ("int", "float") and action not in ("store_true", "store_false"):
                example = "python -m %s %s %s" % (modname, flag,
                                                   "2" if typ == "int" else "0.1")
            elif action in ("store_true", "store_false"):
                example = "python -m %s %s" % (modname, flag)
            else:
                example = "python -m %s %s VALUE" % (modname, flag)
            reachable = bool(has_main)
            out.append({
                "id": "cli:" + flag,
                "kind": "cli",
                "name": flag,
                "signature": "，".join(sig_parts) if sig_parts else "值 / value",
                "example": example if reachable else "",
                "module": modname,
                "subkind": "",
                "help": str(info.get("help") or ""),
                "form": example if reachable else "",
                "form_kind": "cli",
                "reachable": reachable,
                "reach_evidence": "%s 的 add_argument(%r)（模块有 __main__ 入口）"
                                  % (filename, flag)
                                  if reachable else
                                  "%s 的 add_argument(%r)（模块没有 __main__，跑不起来）"
                                  % (filename, flag),
                "unreachable": "" if reachable else
                               "%s 没有 __main__ 入口，python -m 跑不起来" % filename,
            })
    return out


# ----------------------------------------------------------------- env 轴
_ENV_RE = re.compile(r"\b((?:TGAI|TG)_[A-Z][A-Z0-9_]*[A-Z0-9])\b")
#: 直接读取写法的三种形态（自证用）。
_ENV_DIRECT_RES = (
    (re.compile(r'os\.environ\.(?:get|setdefault)\s*\(\s*["\']'
                r'((?:TGAI|TG)_[A-Z][A-Z0-9_]*)["\']'), "os.environ.get"),
    (re.compile(r'os\.getenv\s*\(\s*["\']'
                r'((?:TGAI|TG)_[A-Z][A-Z0-9_]*)["\']'), "os.getenv"),
    (re.compile(r'os\.environ\s*\[\s*["\']'
                r'((?:TGAI|TG)_[A-Z][A-Z0-9_]*)["\']'), "os.environ[...]"),
)
#: `def _env_int(name, default):` 这类"读环境变量的助手函数"定义。
_ENV_HELPER_DEF_RE = re.compile(r"^\s*def\s+(\w*env\w*)\s*\(", re.M)
#: `SOME_ENV = "TG_X"` 这类别名常量（真正的读取处写的是别名）。
_ENV_ALIAS_DEF_RE = re.compile(
    r"^\s*(\w+)\s*=\s*[\"']((?:TGAI|TG)_[A-Z][A-Z0-9_]*)", re.M)


def _line_of(text, pos):
    """字符偏移 -> 1 起始行号（报错/证据用）。"""
    return text.count("\n", 0, pos) + 1


def _env_read_evidence(texts, names):
    """静态证明每个环境变量**真的被代码读过**；返回 {名字: 证据字符串}。

    只认三种读取形态（注释/报错文案里的 `TG_X=1` 不算）：
      1. `os.environ.get("TG_X")` / `os.getenv("TG_X")` / `os.environ["TG_X"]`
      2. 项目内的 env 读取助手：`_env_int("TG_X", ...)`（助手名由 `def ...env...()` 发现）
      3. 别名常量 `X_ENV = "TG_X"`，随后用 `os.environ.get(X_ENV)` / 助手(X_ENV) 读
    """
    evidence = {}
    files = list(texts.items())

    def record(name, filename, pos, how):
        if name not in evidence:
            evidence[name] = "%s:%d %s" % (filename, _line_of(texts[filename], pos), how)

    # 2a. 先发现"项目内 env 助手"的名字
    helpers = set()
    for filename, text in files:
        for m in _ENV_HELPER_DEF_RE.finditer(text):
            helpers.add(m.group(1))
    helper_re = None
    if helpers:
        helper_re = re.compile(
            r"\b(?:%s)\s*\(\s*[\"']((?:TGAI|TG)_[A-Z][A-Z0-9_]*)"
            % "|".join(re.escape(h) for h in sorted(helpers)))

    # 3a. 别名常量：SOME_ENV = "TG_X"（这只是**定义**，不算读取证据）
    alias_defs = []   # [(alias, env_name, filename, pos)]
    for filename, text in files:
        for m in _ENV_ALIAS_DEF_RE.finditer(text):
            alias, env_name = m.group(1), m.group(2)
            if env_name in names:
                alias_defs.append((alias, env_name, filename, m.start()))

    for filename, text in files:
        # 1. 直接读取
        for regex, how in _ENV_DIRECT_RES:
            for m in regex.finditer(text):
                record(m.group(1), filename, m.start(), how)
        # 2b. 助手读取
        if helper_re is not None:
            for m in helper_re.finditer(text):
                record(m.group(1), filename, m.start(), "env 读取助手")

    # 3b. 别名使用：os.environ.get(ALIAS) / env 助手(ALIAS)  —— 只有真的读了才算
    helper_alt = "|".join(sorted(helpers)) if helpers else r"\w*env\w*"
    for alias, env_name, filename, pos in alias_defs:
        if env_name in evidence:
            continue
        use = re.search(
            r"(?:os\.environ[^\n]{0,80}|\b(?:%s)\s*\([^\n]{0,80})\b%s\b"
            % (helper_alt, re.escape(alias)), texts[filename])
        if use is not None:
            record(env_name, filename, use.start(),
                   "别名 %s（定义在 %s:%d）被环境变量读取处使用"
                   % (alias, filename, _line_of(texts[filename], pos)))
    return evidence


def _env_entries():
    texts = {}
    found = {}
    try:
        filenames = sorted(os.listdir(_HERE))
    except Exception:
        filenames = []
    for filename in filenames:
        if not filename.endswith(".py"):
            continue
        path = os.path.join(_HERE, filename)
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except Exception:
            continue
        texts[filename] = text
        for m in _ENV_RE.finditer(text):
            name = m.group(1)
            if len(name) <= 4:
                continue
            found.setdefault(name, filename)
    # ★ 2026-11-21：环境变量候选是**文本扫描**出来的，会被"长得像变量名"的东西污染。
    #   实测被误收的两个：`TG_MAGIC`（read_tg.py 里的模块常量）与
    #   `TG_V4_FORMAT`（我文档里写的 `TG_V4_FORMAT.md` 文件名）。判据补两条：
    #     ① 在包内任何 .py 里有模块级 `NAME = ...` 定义 ⇒ 它是**常量**，不是环境变量；
    #     ② 出现形式是 `NAME.md/.py/.txt` ⇒ 它是**文件名**，不是环境变量。
    try:
        consts = {}
        pkg_root = os.path.dirname(_HERE)
        for dirpath, _dirs, names in os.walk(pkg_root):
            for fn in names:
                if not fn.endswith(".py"):
                    continue
                try:
                    with open(os.path.join(dirpath, fn), encoding="utf-8",
                              errors="replace") as fh:
                        consts[fn] = fh.read()
                except Exception:                # noqa: BLE001
                    continue
    except Exception:                            # noqa: BLE001
        consts = {}
    # ★ 集合式过滤（性能）：早先写成"每个候选 × 每份全文"的 re.search，67 个候选 ×
    #   整个包源码 ⇒ 单次 catalog() 3.4 秒，直接超了网页生成 <1s 的预算。
    #   改成**每份文件只扫一遍**，产出两个集合再判断，复杂度从 O(N×M) 降到 O(M)。
    all_texts = list(texts.values()) + list(consts.values())
    const_names = set()
    filename_tokens = set()
    for t in all_texts:
        const_names.update(re.findall(r"^\s*([A-Z][A-Z0-9_]{3,})\s*=", t, re.M))
        filename_tokens.update(re.findall(r"\b([A-Z][A-Z0-9_]{3,})\.(?:md|py|txt)\b", t))
    dropped = []
    for name in list(found):
        # 模块级常量 / 出现在文件名里 ⇒ 不是环境变量
        if name in const_names or name in filename_tokens:
            dropped.append(name)
    for name in dropped:
        found.pop(name, None)

    evidence = _env_read_evidence(texts, found)
    out = []
    for name in sorted(found):
        proof = evidence.get(name, "")
        reachable = bool(proof)
        out.append({
            "id": "env:" + name,
            "kind": "env",
            "name": name,
            "signature": "环境变量 / environment variable",
            "example": (name + "=1") if reachable else "",
            "module": "tgai_go.train." + found[name][:-3],
            "subkind": "",
            "form": (name + "=1") if reachable else "",
            "form_kind": "env",
            "reachable": reachable,
            "reach_evidence": proof,
            "unreachable": "" if reachable else
                           "只在注释/报错文案里出现过，代码里没有读取点 —— 设置它不会生效",
        })
    return out


# -------------------------------------------------------------------- 主入口
def catalog():
    """返回全部 L0 条目（含机械生成的 L1 示例）。惰性缓存，`invalidate()` 可清。"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    entries = []
    entries.extend(_op_entries())
    entries.extend(_top_level_api_entries())
    entries.extend(_inference_api_entries())        # 推训一体：GO 推理侧
    entries.extend(_inference_cli_entries())          # python -m tgai_go <cmd>
    entries.extend(_fmt_entries())                   # .TG / .TG4 容器
    entries.extend(_kernel_entries())
    entries.extend(_cli_entries())
    entries.extend(_env_entries())
    # 稳定顺序：kind 轴顺序 -> 名字
    order = {k: i for i, k in enumerate(KIND_ORDER)}
    entries.sort(key=lambda e: (order.get(e["kind"], 99), e["name"]))
    _CACHE = entries
    return _CACHE


def entry(item_id):
    """按稳定 ID 取一条 L0 条目；不存在返回 None。"""
    for it in catalog():
        if it["id"] == item_id:
            return it
    return None


def nearest(name, n=3):
    """最接近的候选 ID（报错时给出，避免"没有这个条目"就结束）。"""
    ids = [it["id"] for it in catalog()]
    if not name:
        return ids[:n]
    return difflib.get_close_matches(str(name), ids, n=n, cutoff=0.45)


def invalidate():
    """清空 L0 缓存（`T.help.refresh()` 会调；插件注册新条目后也有用）。"""
    global _CACHE
    _CACHE = None
