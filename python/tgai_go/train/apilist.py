# -*- coding: utf-8 -*-
"""库内自省：`T.help()` —— 这个库到底提供了什么、有什么用、怎么用。

为什么需要它：开发中反复出现"以为没有、其实早就写好了"的情况
（`flash_attn` / `_checkpoint_layer` / `pool_dump_live` 都踩过），
代价是重复劳动甚至重复造轮子。这里给一个**带中英双语说明的权威清单**。

用法 / Usage
------------
    T.help()                        # 前 10 个内核 + 10 个 API + 10 个算子
    T.help(tag="API")               # 只看运行时 API（每页 30）
    T.help(tag="内核", page=2)      # 内核第 2 页
    T.help(tag="算子", s="mul")     # 在算子里搜 mul
    T.help(s="flash")               # 全类别搜索
    T.help("flash")                 # 等价：位置参数当关键字
    T.help(n=50)                    # 自定每页条数
    T.help("mul", format="json")    # 给 AI/脚本：stdout 只有 JSON
    T.help(open="web")              # 生成离线 Wiki（单个自包含 HTML）并打印路径

给 AI / 自动化脚本 / for AI & scripts
--------------------------------------
* `format="json"`：**纯 JSON**（stdout 里除 JSON 外没有任何东西、无 ANSI 颜色、
  不分页），schema 稳定为 `tgai-help/2`（字段名是契约）：

      顶层：schema / query / kind / count / items / hint / wiki_version /
            coverage / errors / warnings
      item：id / kind / name / title_zh / title_en / zh / en / usage /
            namespace / category / documented / module / wiki_example /
            see_also / author / reachable / form / form_kind /
            reach_evidence / unreachable

  `namespace` 是引擎按真实签名机械生成的调用示例（`x.sum()` / `a * b` /
  `T.matmul(a, b)` / `rt.launch(...)` / `python -m ... --flag` / `TG_X=1`）；
  **不可调用的条目 `reachable=false`、`namespace` 为空**，`unreachable` 给出原因，
  绝不给你一个照抄就报错的假示例。同名条目可能出现在不同 kind（例如 `abs` 既是
  运行时 API 又是张量算子），用 item 里的 `kind` 区分。
  未命中时 `count=0`、`hint` 给出可执行的下一步，**不会静默空输出**。
* `open="web"`：把当前（可带关键词过滤的）条目渲染成**一个自包含 HTML**，
  写进 `tempfile` 临时目录，尝试用默认浏览器打开，并把文件路径打印到 stdout。
  无头/CI 下浏览器打不开不算失败；路径照常打印并返回。渲染模块 `help_web`
  **只在此时才 import**，默认路径的导入开销为零。页面黑白极简、零外链，
  顶栏 `EN | 中文` 全局语言开关（默认英文；`?lang=zh` 可指定）。

tag 认中英文 / tag accepts CN & EN
-----------------------------------
    内核  kernel / kernels / cuda内核
    API   api / apis / 接口 / 运行时 / backend
    算子  op / ops / operator / 张量算子
    优化器 opt / optimizer
    模块  mod / module / 子模块

设计约束 / Design notes
-----------------------
* **不需要 GPU**：只读源码文本与 `__doc__`，不初始化 CUDA。
* 纯标准库；默认打印，`print=False` 时返回字符串（便于测试/管道）。
"""

import inspect
import os
import re

__all__ = ["help", "describe"]      # noqa: A001  （刻意重名，用户写 T.help()）

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAGE = 30

#: 常用条目的英文说明（其余用通用模板）。键是名字，值是英文一句话。
_EN = {
    # ---- 内核 ----
    "sgemm": "Row-major fp32 GEMM via cuBLAS.",
    "sgemm_nn": "Batched fp32 GEMM, A*B (no transpose).",
    "sgemm_nt": "Batched fp32 GEMM, A*B^T (transposed second operand).",
    "sgemm_tn": "Batched fp32 GEMM, A^T*B.",
    "gemm_f16_nt_raw": "fp16-input / fp32-accumulate GEMM, A*B^T, raw pointers.",
    "mul_bcast8": "Elementwise multiply with broadcasting, computed by stride (no broadcast copy).",
    "add_bcast8": "Elementwise add with broadcasting, computed by stride.",
    "ewise_mul_v4": "Plain elementwise multiply (float4 vectorized).",
    "ewise_add_v4": "Plain elementwise add (float4 vectorized).",
    "fused_add_ip_v4": "In-place add (x += y), used for gradient accumulation.",
    "swap_last2": "Transpose the last two dims into a new contiguous buffer.",
    "transpose2d": "2-D transpose kernel.",
    "sum_axis8": "Reduction along one axis (up to 8-D).",
    "rmsnorm_fwd": "Fused RMSNorm forward.",
    "rmsnorm_bwd_det": "Fused RMSNorm backward (deterministic).",
    "scatter_rows_owned": "Row scatter-add, deterministic (row-owned, ascending source order, no atomics).",
    "scatter_flat_owned": "Flat scatter-add, deterministic (index-owned, ascending source order, no atomics).",
    "situ_glu_fwd": "Fused SiTU-GLU activation forward (no intermediates kept).",
    "situ_glu_bwd": "Fused SiTU-GLU backward (recomputes from gate/up).",
    "adamw_update_multi_bf16": "Multi-tensor AdamW with bf16 moments, one launch for all params.",
    "adamw_multi_int8": "Multi-tensor AdamW with int8 blockwise moments (+bf16 v variant).",
    "adamw_multi_m8v16": "Multi-tensor AdamW: int8 first moment + bf16 second moment.",
    "cast_f32_to_bf16": "fp32 -> bf16 narrowing (round-to-nearest-even).",
    "cast_bf16_to_f32": "bf16 -> fp32 widening (exact bit shift).",
    "index_gather8": "Gather rows by index.",
    "index_scatter_add8": "Scatter-add rows by index (atomic accumulate).",
    # ---- API ----
    "add": "Elementwise add of two buffers (same shape).",
    "mul": "Elementwise multiply of two buffers (same shape).",
    "sgemm": "Run a GEMM on GpuBuf arguments.",
    "empty": "Allocate a device buffer from the pool.",
    "h2d": "Host-to-device copy.",
    "d2h": "Device-to-host copy.",
    "sync": "Block until all queued work finishes.",
    "pool_stats": "Allocator statistics: hits/misses, live/cached/peak bytes.",
    "pool_dump_live": "Current checked-out buffers grouped by size (authoritative memory ledger).",
    "adamw_multi": "Fused multi-tensor AdamW with fp32 moments.",
    "adamw_multi_bf16": "Fused multi-tensor AdamW with bf16 moments.",
    "device_caps": "GPU capability probe: compute capability, fp16/bf16 arithmetic and storage.",
    "resolve_precision": "Pick storage/compute precision for this GPU.",
    "graph_available": "Whether CUDA graph capture is usable.",
    # ---- 算子 ----
    "backward": "Run autograd from this tensor; pass a gradient for non-scalars.",
    "detach": "Copy without autograd history (note: ORV copies, not a view).",
    "transpose": "2-D transpose; recorded as a lazy transpose so matmul can use gemm_nt.",
    "reshape": "Reinterpret shape (metadata only when contiguous).",
    "compact_bf16": "Store this tensor's data as bf16 (halves memory); reads upcast temporarily.",
    "is_bf16_stored": "Whether this tensor is currently bf16-stored.",
    "to_numpy": "Copy back to a NumPy array.",
    "sum": "Sum all elements (or along an axis).",
    "mean": "Mean of all elements.",
    "softmax": "Softmax along an axis.",
    "relu": "ReLU activation.",
}

#: 常用条目的中文说明（docstring 缺失或太技术时用）
_CN = {
    "sgemm": "行优先单精度 GEMM（内部转成列主序调 cuBLAS）。",
    "mul_bcast8": "带广播的逐元素乘，按步长直接算，不物化广播副本。",
    "add_bcast8": "带广播的逐元素加，按步长直接算。",
    "fused_add_ip_v4": "就地相加（x += y），用于梯度累加。",
    "swap_last2": "把最后两维转置成新的连续缓冲。",
    "rmsnorm_fwd": "融合 RMSNorm 前向。",
    "rmsnorm_bwd_det": "融合 RMSNorm 反向（确定性归约）。",
    "scatter_rows_owned": "行散射累加（确定性：目标行独占、源行号升序、无原子）。",
    "scatter_flat_owned": "逐元素散射累加（确定性：下标独占、升序、无原子）。",
    "situ_glu_fwd": "融合 SiTU-GLU 激活前向（不留中间量）。",
    "situ_glu_bwd": "融合 SiTU-GLU 反向（由 gate/up 重算）。",
    "adamw_update_multi_bf16": "多张量融合 AdamW（bf16 动量），全部参数一次发射。",
    "adamw_multi_m8v16": "多张量 AdamW：一阶动量 int8 分块 + 二阶动量 bf16。",
    "cast_f32_to_bf16": "fp32 → bf16 收窄（RTNE 就近舍入）。",
    "cast_bf16_to_f32": "bf16 → fp32 升宽（纯左移，精确）。",
    "index_gather8": "按索引取行。",
    "index_scatter_add8": "按索引散加（原子累加）。",
    "pool_dump_live": "当前借出的显存按大小分布（权威账本，用于定位显存归属）。",
    "device_caps": "探测 GPU 能力：算力等级、fp16/bf16 的算术与存储支持。",
    "compact_bf16": "把本张量改成 bf16 存储（省一半显存），读时临时升回 fp32。",
    "transpose": "二维转置；记录为惰性转置，供 matmul 走 gemm_nt 省一次拷贝。",
    "detach": "脱离计算图（注意：ORV 上是**拷贝**而非视图）。",
}

_INTRO = {
    "kernels": (
        "CUDA 内核 / CUDA kernels — GPU 上真正执行的函数，NVRTC 运行时编译。\n"
        "    调用方式是 rt.launch(\"<名字>\", grid, block, [参数...])。\n"
        "    These are the GPU functions themselves; launch with rt.launch(name, grid, block, args)."
    ),
    "api": (
        "运行时 API / Runtime APIs — Python 侧的后端原语（分配、拷贝、GEMM、归约、优化器…）。\n"
        "    调用方式是 rt.<名字>(...)。\n"
        "    Backend primitives; call them as rt.<name>(...)."
    ),
    "op": (
        "张量算子 / Tensor operators — Tensor 上**真实存在**的方法（自动建图并参与求导）。\n"
        "    方法形式：<张量>.<方法>(...)，例如 a.sum() / a.reshape((2,3)) / a.numel()。\n"
        "    **没有方法形式**的运算走运算符：逐元素加 a + b、减 a - b、乘 a * b、除 a / b，\n"
        "    矩阵乘 a @ b（或顶层 T.matmul(a, b)），标量幂 a ** 2。\n"
        "    ⚠️ Tensor 没有 .mul()/.add()/.matmul() 方法，顶层也没有 T.mul ——\n"
        "    照 PyTorch 写 a.mul(b) 会 AttributeError；矩阵乘更不能用 *（那是逐元素）。\n"
        "    Methods on Tensor: a.<method>(...). Element-wise ops are operators (a + b, a * b);\n"
        "    matmul is a @ b or T.matmul(a, b); scalar power is a ** 2. There is no .mul()/T.mul."
    ),
    "opt": (
        "优化器 / Optimizers — 直接实例化即可，接口同 torch.optim（step()/zero_grad()）。\n"
        "    Instantiate directly; torch.optim-like step()/zero_grad()."
    ),
    "mods": (
        "子模块 / Submodules — 按主题分组的实现文件，import 后用其中的函数/类。\n"
        "    Topic-grouped modules; import and use their functions/classes."
    ),
}


def _first_line(text, limit=120):
    if not text:
        return ""
    for ln in str(text).strip().splitlines():
        ln = " ".join(ln.split())          # 收紧空白（注释里常有对齐用的多空格）
        if not ln or ln.startswith((":param", ":return", "---", "===", "```")):
            continue
        if len(ln) > limit:
            ln = ln[:limit - 1] + "…"
        return ln
    return ""


def _read(name):
    try:
        with open(os.path.join(_HERE, name), encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


# ---------------------------------------------------------------- CUDA 内核
_KERN_RE = re.compile(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(', re.S)


def _kernels():
    txt = _read("cuda.py")
    out = []
    for m in _KERN_RE.finditer(txt):
        name = m.group(1)
        i = m.end() - 1
        depth = 0
        j = i
        while j < len(txt):
            c = txt[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        params = " ".join(txt[i + 1:j].split())
        head = txt[max(0, m.start() - 3000):m.start()]
        desc = ""
        # ★ 2026-11-19 修：原来取"内核前 800 字符内最后一条 /* */"，
        #   于是**没有自己注释**的内核会借用上一个内核的说明
        #   （实测 `ewise_div` 显示了 `ewise_clamp` 的 "lo/hi ..."）。
        #   现在只认**紧邻上方**的注释：注释结束到内核开头之间除空行外不允许有别的东西。
        last_cm = None
        for cm in re.finditer(r"/\*(.*?)\*/", head, re.S):
            last_cm = cm
        if last_cm is not None and head[last_cm.end():].strip() == "":
            desc = _first_line(last_cm.group(1).replace("*", " "))
        if not desc:
            # 同理处理 `//` 注释：只取紧邻上方的**连续**几行，遇到代码就停
            got = []
            for ln in reversed(head.splitlines()):
                s = ln.strip()
                if not s:
                    if got:
                        break
                    continue
                if s.startswith("//"):
                    got.append(s.lstrip("/ ").strip())
                    continue
                break
            if got:
                desc = _first_line(" ".join(reversed(got)))
        # ★ 只接受"像一句话"的注释：过短、以符号开头、或明显是句中片段（★/→/…开头）
        #   的一律丢掉，否则列出来会像 "ce_fwd_rows: ★ 改成 跨步 访问：原来…" 这种碎片。
        if desc:
            bad = (len(desc) < 6 or desc[0] in "★→·-—*/" or "原来" in desc[:6]
                   or not (desc.endswith(("。", ".", "）", ")")) or len(desc) > 20))
            if bad:
                desc = ""
        out.append({"name": name, "params": params, "doc": desc})
    return out


# ------------------------------------------------------------ 运行时 API
def _rt_class():
    """拿到运行时类，**不需要 GPU**。"""
    try:
        import tgai_go.train as T
        return type(T._cuda())
    except Exception:
        pass
    try:
        from . import cuda as _c
    except Exception:
        _c = None
    if _c is not None:
        cls = getattr(_c, "CudaRuntime", None)
        if isinstance(cls, type):
            return cls
    # ★ `tgai_go.train.cuda` 可能是**转发 shim**，真正的模块在别处
    #   （实测：shim 上 `CudaRuntime` 是 None）。在 sys.modules 里按类名找。
    import sys as _sys
    for mod in list(_sys.modules.values()):
        try:
            cls = getattr(mod, "CudaRuntime", None)
        except Exception:
            continue
        if isinstance(cls, type):
            return cls
    return None


def _rt_methods():
    cls = _rt_class()
    if cls is None:
        return []
    out = []
    for nm in sorted(dir(cls)):
        if nm.startswith("_"):
            continue
        fn = getattr(cls, nm, None)
        if not callable(fn):
            continue
        try:
            sig = str(inspect.signature(fn))
        except Exception:
            sig = "(…)"
        out.append({"name": nm, "sig": sig,
                    "doc": _first_line(getattr(fn, "__doc__", ""))})
    return out


# ---------------------------------------------------------------- 张量算子
def _tensor_ops():
    try:
        import tgai_go.train as T
        cls = T.Tensor
    except Exception:
        return []
    out = []
    for nm in sorted(dir(cls)):
        if nm.startswith("_"):
            continue
        fn = getattr(cls, nm, None)
        if not callable(fn):
            continue
        try:
            sig = str(inspect.signature(fn))
        except Exception:
            sig = "(…)"
        out.append({"name": nm, "sig": sig,
                    "doc": _first_line(getattr(fn, "__doc__", ""))})
    return out


# ------------------------------------------------------------------ 子模块
def _modules():
    out = []
    for fn in sorted(os.listdir(_HERE)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        mod = fn[:-3]
        if mod in ("help", "apilist", "backend", "dispatch", "dtypes",
                   "help_l0", "help_wiki", "help_web"):
            continue
        txt = _read(fn)
        names = sorted(set(re.findall(r"^def\s+([a-zA-Z]\w*)", txt, re.M))
                       | set(re.findall(r"^class\s+([a-zA-Z]\w*)", txt, re.M)))
        if not names:
            continue
        m = re.match(r'\s*(?:#[^\n]*\n)*\s*(?:r?)"""(.*?)"""', txt, re.S)
        doc = _first_line(m.group(1)) if m else ""
        out.append({"name": mod, "doc": doc, "names": names})
    return out


def _optimizers():
    try:
        from . import optim as _o
    except Exception:
        return []
    out = []
    for nm in sorted(dir(_o)):
        if nm.startswith("_"):
            continue
        obj = getattr(_o, nm, None)
        if inspect.isclass(obj):
            out.append({"name": nm, "doc": _first_line(getattr(obj, "__doc__", ""))})
    return out


# ------------------------------------------------------------------ 渲染
def _render(kind, items, page, n):
    """渲染一页条目（中英双语）。"""
    lines = []
    start = (page - 1) * n
    chunk = items[start:start + n]
    total_page = max(1, (len(items) + n - 1) // n)
    lines.append("[%s] %d 条 / %d items  —  第 %d/%d 页 (page %d/%d, 每页 %d)"
                 % (kind, len(items), len(items), page, total_page, page, total_page, n))
    lines.append("  " + _INTRO.get(kind, ""))
    if not chunk:
        lines.append("  （本页无内容；用 page=1 从头看 / empty page, try page=1）")
        return "\n".join(lines)
    for idx, it in enumerate(chunk, start=start + 1):
        nm = it["name"]
        cn = _CN.get(nm) or it.get("doc") or ""
        en = _EN.get(nm) or ""
        lines.append("\n  %d. %s" % (idx, nm))
        # ★ 说明是**渐进补充**的：没填写就只列名字 + 用法，不刷占位符
        #   （给上百个内核写双语简介是个长期工程，先保证"能查得到名字"。
        #    插件作者可以调 `T.describe(name, cn=..., en=...)` 自己挂。）
        if cn:
            lines.append("     中文: %s" % cn)
        if en:
            lines.append("     EN  : %s" % en)
        if it.get("sig"):
            lines.append("     用法/Usage: %s" % it["sig"])
        elif it.get("params"):
            lines.append("     参数/Params: (%s)" % it["params"])
        if kind == "mods" and it.get("names"):
            more = ", ".join(it["names"][:12])
            lines.append("     内含/Contains: %s%s"
                         % (more, " …" if len(it["names"]) > 12 else ""))
    if total_page > 1:
        lines.append("\n  翻页 / next: T.help(tag=%r, page=%d)"
                     % (kind, min(page + 1, total_page)))
    return "\n".join(lines)


# -------------------------------------------------------------------- 主入口
def describe(name, cn=None, en=None, doc=None):
    """给某条目登记/覆盖中英文说明 —— **插件作者用这个接口**。

    内置说明是渐进补充的（给上百个内核写双语简介是长期工程），
    所以这里允许运行时登记，`T.help()` 会立刻看到：

        T.describe("my_kernel", cn="我的融合内核，做 X。", en="My fused kernel doing X.")
        T.describe("MyPluginOp", doc="一句话说明")   # 只给中文也认

    参数：
      name : 条目名（内核名 / rt 方法名 / 算子名 / 类名）
      cn   : 中文说明（等价写法 `doc=`）
      en   : 英文说明
    返回登记后的 {"cn", "en"}。
    """
    key = str(name).strip()
    if not key:
        raise ValueError("describe 需要非空名字。修法：传内核/方法/算子的名字")
    if cn is None and doc is not None:
        cn = doc
    if cn is not None:
        _CN[key] = _first_line(cn) or str(cn)
    if en is not None:
        _EN[key] = _first_line(en) or str(en)
    return {"cn": _CN.get(key, ""), "en": _EN.get(key, "")}


def help(topic=None, tag=None, s=None, page=1, n=_PAGE, full=False, *,
         format=None, open=None, launch=None, **kw):   # noqa: A001
    """`T.help()` —— 这个库提供了什么、有什么用、怎么用（中英双语）。

    参数 / Args:
      tag   : 类别 `kernel` / `api` / `op` / `opt` / `mod`（也认中文：内核/API/算子/优化器/模块）
      s     : 关键词（在名字、中英文说明与**用法签名**里做不区分大小写的子串匹配）
      page  : 页码，从 1 开始（每页默认 30 条）
      n     : 每页条数
      topic : 位置参数；若等于某个类别名则当 tag，否则当关键词 s
      full  : 保留参数（当前用于打印完整签名）
      print : 传 `print=False` 时返回字符串而不打印
      format: `None`/`"text"`（默认）或 `"json"` —— json 只输出纯 JSON（给 AI/脚本）
      open  : `None` 或 `"web"` —— 生成自包含离线 Wiki（单文件）并把路径打印到 stdout

    附加属性 / attributes（全部惰性加载，默认路径不付任何代价）：
      T.help.check()          词条健康度：解析失败/未知 ID/中英缺边/死链/覆盖率
      T.help.missing()        待编写清单（按 kind/category 分组，给朋友们的看板）
      T.help.contrib("op:mul") 有词条打印路径；没有就生成 TXT 骨架
      T.help.refresh()        强制清空 wiki 缓存（热更新一般不需要，改文件会自动生效）
    """
    do_print = bool(kw.pop("print", True))
    if kw:
        raise TypeError("help() 不认识参数 %s。修法：可用 tag / s / page / n / print / "
                        "format / open / launch（launch=True 才会用浏览器打开网页）"
                        % ", ".join(sorted(kw)))

    def out(text):
        if do_print:
            print(text)
        return text

    alias = {
        "kernels": ("内核", "内核表", "kernel", "kernels", "cuda内核", "gpu内核"),
        "api": ("api", "apis", "接口", "运行时", "rt", "后端", "backend", "运行时接口"),
        "op": ("算子", "张量算子", "op", "ops", "operator", "operators"),
        "opt": ("优化器", "opt", "optimizer", "optimizers", "optim"),
        "mods": ("模块", "子模块", "mod", "mods", "module", "modules"),
    }

    def norm(x):
        if x is None:
            return None
        k = str(x).strip().lower()
        for cat, names in alias.items():
            if k in names:
                return cat
        return k

    kind = norm(tag)
    if topic is not None:
        t = norm(topic)
        if kind is None and t in alias:
            kind = t
        elif s is None:
            s = str(topic)
    if kind is not None and kind not in alias:
        # 不是类别名 ⇒ 当作关键词
        if s is None:
            s = str(tag)
        kind = None

    try:
        page = max(1, int(page))
    except Exception:
        page = 1
    try:
        n = max(1, int(n))
    except Exception:
        n = _PAGE

    # ---- 给 AI / 脚本的接口：纯 JSON、离线网页（★ 惰性导入，默认路径零开销）----
    fmt = None if format is None else str(format).strip().lower()
    if fmt in ("", "text", "txt", "plain"):
        fmt = None
    if fmt not in (None, "json"):
        raise ValueError("format 只支持 'text'（默认）或 'json'，收到 %r。"
                         "修法：批量场景用 T.help('关键词', format='json')" % (format,))
    opn = None if open is None else str(open).strip().lower()
    if opn == "":
        opn = None
    if opn not in (None, "web", "browser", "html"):
        raise ValueError("open 只支持 'web'（生成离线 Wiki），收到 %r。"
                         "修法：用 T.help(open='web')" % (open,))
    if fmt == "json" and opn is not None:
        raise ValueError("format='json' 与 open='web' 不能同时用。"
                         "修法：分两次调用 —— JSON 走 stdout，Wiki 走浏览器")
    if fmt == "json":
        from . import help_wiki
        import json as _json
        payload = help_wiki.json_payload(kind, s)
        return out(_json.dumps(payload, ensure_ascii=False, indent=2))
    if opn is not None:
        from . import help_wiki
        from . import help_web
        # ★ 2026-11-21：`launch` 必须透传。我第一次只改了 open_web 的签名，
        #   没把它接到 T.help 上，用户按文档写 T.help(open='web', launch=True)
        #   直接 TypeError —— 文档承诺了但代码没实现，比没有更糟。
        path = help_web.open_web(help_wiki.web_payload(kind, s), launch=launch)
        return out(path)

    src = {"kernels": _kernels, "api": _rt_methods, "op": _tensor_ops,
           "opt": _optimizers, "mods": _modules}
    label = {"kernels": "内核 / kernels", "api": "运行时 API / runtime APIs",
             "op": "张量算子 / tensor ops", "opt": "优化器 / optimizers",
             "mods": "子模块 / submodules"}

    def gather(k):
        items = src[k]()
        if s:
            key = str(s).lower()

            def hit(it):
                nm = it["name"]
                # ★ 关键词要能命中"用法签名"：只知道参数名也能搜到（例如 entries / keepdims）
                blob = " ".join((nm, str(it.get("doc", "")),
                                 _CN.get(nm, ""), _EN.get(nm, ""),
                                 str(it.get("sig", "")), str(it.get("params", "")),
                                 " ".join(it.get("names") or ()))).lower()
                return key in blob
            items = [it for it in items if hit(it)]
        # ★ 有中英文说明的排前面：默认列举时才看得出"这东西有什么用"，
        #   否则前 10 条会全是没注释的冷门内核。
        items = sorted(items, key=lambda it: (it["name"] not in _EN
                                              and it["name"] not in _CN, it["name"]))
        return items

    # ---- 无参数：各列 10 个 ----
    if kind is None and not s:
        lines = ["=" * 70,
                 "tgai_go.train 能力总览 / capability overview",
                 "=" * 70,
                 "  这个库用纯 Python + CUDA 驱动 + cuBLAS + NVRTC 做训练：",
                 "  内核在 GPU 上跑，Python 只负责编排与自动求导。",
                 "  Pure-Python autograd over CUDA driver/cuBLAS/NVRTC.",
                 "",
                 "  用 T.help(tag=..., s=..., page=...) 深入：",
                 "    T.help(tag=\"内核\")              全部 CUDA 内核（分页 30）",
                 "    T.help(tag=\"API\", s=\"gemm\")    在运行时 API 里搜 gemm",
                 "    T.help(s=\"flash\")               全类别搜索 flash",
                 "=" * 70]
        for k in ("kernels", "api", "op"):
            items = gather(k)
            lines.append("")
            lines.append(_render(k, items[:10], 1, 10))
            if len(items) > 10:
                lines.append("  ……共 %d 条，看全部：T.help(tag=%r)" % (len(items), k))
        return out("\n".join(lines))

    # ---- 只给关键词：跨类别搜索 ----
    if kind is None:
        lines = ["搜索 %r / search results（不区分大小写）:" % s]
        total = 0
        for k in ("kernels", "api", "op", "opt", "mods"):
            items = gather(k)
            if not items:
                continue
            total += len(items)
            lines.append("\n" + _render(k, items, 1, n))
        wiki = _wiki_text(s) if s else ""
        if total == 0:
            if wiki:
                lines.append("  内置类别里没有命中；下面是外部 Wiki 词条 / matched wiki entries:")
            else:
                lines.append("  没有命中 / no match（不区分大小写）。")
                lines.extend(_no_hit_hints())
        if wiki:
            lines.append("\n" + wiki)
        return out("\n".join(lines))

    # ---- 指定类别 ----
    items = gather(kind)
    if not items:
        text = "%s 里没有匹配 %r 的条目 / no match in %s. 换个关键词，或不带 s 列出全部。" \
               % (label[kind], s, kind)
        wiki = _wiki_text(s, kind) if s else ""
        if wiki:
            text += "\n\n" + wiki
        else:
            text += "\n" + "\n".join(_no_hit_hints())
        return out(text)
    text = _render(kind, items, page, n)
    if s:
        wiki = _wiki_text(s, kind)
        if wiki:
            text += "\n\n" + wiki
    return out(text)


# ═══════════════════════════════════════════════════════════════════════════
# 外部词条（wiki/*.txt）的惰性入口 —— 默认 `T.help()` 不会 import 这些模块
# ═══════════════════════════════════════════════════════════════════════════
def _lazy_wiki():
    """惰性拿到 L2 加载器（只有 JSON / web / check / missing / contrib / 搜索时才导入）。"""
    from . import help_wiki
    return help_wiki


def _wiki_text(s, kind=None):
    """终端搜索追加的 Wiki 片段；任何异常都降级为空（帮助本身不能因此报错）。"""
    try:
        return _lazy_wiki().text_search(s, kind)
    except Exception:
        return ""


def _no_hit_hints():
    """没有命中时给出**可执行**的下一步（不要静默空输出）。"""
    return [
        "  可执行的下一步 / next:",
        "    T.help()                     总览（每类前 10 条）",
        "    T.help(tag='kernels')        全部 CUDA 内核",
        "    T.help(tag='api')            运行时 API",
        "    T.help(tag='op')             张量算子（t.<名字>）",
        "    T.help(tag='opt' / 'mods')   优化器 / 子模块",
        "    T.help(s='gemm')             换关键词，跨类别搜索",
        "    T.help(open='web')           生成可搜索的离线 Wiki（单个自包含 HTML）",
        "    T.help(format='json')        给 AI/脚本的纯 JSON（stdout 只有 JSON）",
    ]


def _help_refresh():
    """`T.help.refresh()` —— 强制清空 wiki 缓存（改文件一般自动生效）。"""
    return _lazy_wiki().refresh()


def _help_missing():
    """`T.help.missing()` —— 待编写清单（按 kind/category 分组）。"""
    return _lazy_wiki().missing()


def _help_check():
    """`T.help.check()` —— 词条健康度：解析失败/未知 ID/中英缺边/死链/覆盖率。"""
    return _lazy_wiki().check()


def _help_contrib(item_id=None):
    """`T.help.contrib('op:mul')` —— 有词条打印路径，没有就生成 TXT 骨架。"""
    return _lazy_wiki().contrib(item_id)


#: 挂在函数上（不是模块级新名字），`T.help.check()` 这样调用最自然。
help.refresh = _help_refresh
help.missing = _help_missing
help.check = _help_check
help.contrib = _help_contrib
