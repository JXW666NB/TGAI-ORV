# -*- coding: utf-8 -*-
"""`.tgop` —— 通用算子插件包：**一个文件装下所有厂商的实现**。

## 形态

`.tgop` 就是一个 zip（改后缀），里面固定有一份 `manifest.json`：

```
gelu.tgop
├── manifest.json          # 必需：算子名、arity、每个后端对应的文件/内核名
├── gelu.cu                # CUDA / HIP / MUSA 三家复用（源码级同形）
├── gelu.c                 # 通用 C（CPU JIT）
├── gelu.spv               # Vulkan 预编译 SPIR-V
└── gelu_py.py             # 兜底 Python 实现（必须给，保证任何机器都能跑）
```

manifest 示例：

```json
{
  "schema": 1,
  "name": "gelu",
  "arity": 1,
  "backends": {
    "cuda":   {"source": "gelu.cu", "fwd": "gelu_fwd", "bwd": "gelu_bwd"},
    "hip":    {"source": "gelu.cu", "fwd": "gelu_fwd", "bwd": "gelu_bwd"},
    "musa":   {"source": "gelu.cu", "fwd": "gelu_fwd", "bwd": "gelu_bwd"},
    "vulkan": {"spirv": "gelu.spv", "entry": "main"},
    "cpu":    {"c": "gelu.c"},
    "python": {"module": "gelu_py.py", "callable": "gelu"}
  }
}
```

## 为什么是这种设计

* **一个文件、多家厂商**：用户只发一个 `.tgop`，NVIDIA / AMD / 海光 / 摩尔线程 /
  Vulkan / CPU 的用户都能用；各自只解自己那一档。
* **CUDA 源码三家复用**：HIP（hipRTC）与 MUSA（mcc）语法同形，所以 `gelu.cu` 一份就够；
  昇腾 Ascend C / 寒武纪 BANG 语言不同，需要各自新增一个 `ascend`/`bang` 键（格式留好）。
* **python 档强制要求**：任何机器都至少有正确实现，绝不"装了跑不了"。

## 选取顺序

`cuda -> hip -> musa -> vulkan -> cpu -> python`，缺失或不可用就往下掉。
"""
import json
import os
import zipfile

SCHEMA = 1
_ORDER = ("cuda", "hip", "musa", "ascend", "bang", "vulkan", "cpu", "python")
REQUIRED_BACKEND = "python"


class BundleError(RuntimeError):
    """插件包不合法或本机无可用实现（消息里一定带修法）。"""


def _load_manifest(zf):
    names = zf.namelist()
    if "manifest.json" not in names:
        raise BundleError(
            "插件包缺少 manifest.json。修法：用 `python tools/tgop_pack.py <目录> -o x.tgop` 生成，"
            "或手写一份（见 docs 里的示例）")
    try:
        m = json.loads(zf.read("manifest.json").decode("utf-8-sig"))  # 容忍 Windows 编辑器的 BOM
    except Exception as exc:                            # noqa: BLE001
        raise BundleError("manifest.json 不是合法 JSON：%s。修法：检查语法" % exc)
    if int(m.get("schema", 0)) != SCHEMA:
        raise BundleError("插件包 schema=%r 不受支持（引擎支持 %d）。修法：升级引擎或重新打包"
                          % (m.get("schema"), SCHEMA))
    if not m.get("name"):
        raise BundleError("manifest 缺少 name。修法：填算子名，例如 \"gelu\"")
    if int(m.get("arity", 0)) not in (1, 2):
        raise BundleError("manifest.arity 必须是 1 或 2，收到 %r。修法：一元算子填 1"
                          % (m.get("arity"),))
    bk = m.get("backends") or {}
    if REQUIRED_BACKEND not in bk:
        raise BundleError(
            "manifest.backends 必须包含 python 兜底实现（否则没有 GPU 的机器装了跑不了）。"
            "修法：加 \"python\": {\"module\": \"x_py.py\", \"callable\": \"fn\"}")
    return m


def _pick(manifest, available, names=None):
    """挑第一个「本机可用 且 包里引用的文件确实存在」的后端。

    只查可用性不查文件是不够的：包里写了 `vulkan` 却没带 `.spv` 时，
    选到它会在加载阶段抛 KeyError。这里直接顺延到下一个后端。
    """
    names = set(names or ())
    for b in _ORDER:
        if b not in manifest["backends"] or b not in available:
            continue
        spec = manifest["backends"][b] or {}
        refs = [spec[k] for k in ("source", "c", "spirv", "bwd_spirv", "module")
                if isinstance(spec, dict) and spec.get(k)]
        if names and any(r not in names for r in refs):
            continue
        return b
    return None


def _python_impl(zf, spec, name):
    mod_file = spec.get("module")
    fn_name = spec.get("callable") or name
    if not mod_file:
        raise BundleError("python 档缺少 module。修法：写 {\"module\": \"x_py.py\", "
                          "\"callable\": \"fn\"}")
    src = zf.read(mod_file).decode("utf-8-sig")  # 插件作者常用 Windows 编辑器，带 BOM
    ns = {"__name__": "tgai_tgop_%s" % name}
    exec(compile(src, mod_file, "exec"), ns)           # noqa: S102  插件包是用户自己的代码
    fn = ns.get(fn_name)
    if not callable(fn):
        raise BundleError("python 档里找不到可调用对象 %r（文件 %s）。修法：确认函数名"
                          % (fn_name, mod_file))
    return fn


def load(path, verbose=True):
    """加载一个 `.tgop` 插件包，返回 (算子名, 选中的后端)。"""
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到插件包：%s。修法：检查路径（扩展名应为 .tgop）" % path)
    from tgai_go.train import ops_cuda, ops_native, ops_plugin

    with zipfile.ZipFile(path) as zf:
        m = _load_manifest(zf)
        name, arity = m["name"], int(m.get("arity", 1))
        available = ops_native.supported()
        chosen = _pick(m, available, zf.namelist())
        if chosen is None:
            raise BundleError(
                "插件 %r 里没有本机可用且文件齐全的后端（本机：%s，包里：%s）。"
                "修法：补一个 python 档，或换一台有对应加速卡的机器"
                % (name, available, sorted(m["backends"])))
        spec = m["backends"][chosen]

        if chosen == "cuda":
            src = zf.read(spec["source"]).decode("utf-8")
            mod = ops_cuda.load_source(src, cu_name=spec["source"])
            fwd = spec.get("fwd") or (name + "_fwd")
            bwd = spec.get("bwd") or (name + "_bwd")
            bwd2 = spec.get("bwd2")
            ops_cuda.register_elementwise(name, mod, fwd, bwd, bwd2, arity=arity)
        elif chosen == "cpu":
            from tgai_go.train import ops_cpu
            src = zf.read(spec["c"]).decode("utf-8-sig")
            h = ops_cpu.load_source(src, name="tgop_" + name)
            fwd = spec.get("fwd") or (name + "_fwd")
            bwd = spec.get("bwd") or (name + "_bwd")
            ops_cpu.register_elementwise(name, h, fwd, bwd, spec.get("bwd2"), arity=arity)
        elif chosen == "vulkan":
            from tgai_go.train import ops_vk_ops
            fwd = ops_vk_ops.spv_of(zf.read(spec["spirv"]))
            bwd = (ops_vk_ops.spv_of(zf.read(spec["bwd_spirv"]))
                   if spec.get("bwd_spirv") else fwd)
            ops_vk_ops.register_elementwise(name, fwd, bwd, arity=arity)
        elif chosen == "python":
            ops_plugin.register(name, _python_impl(zf, spec, name))
        else:
            raise BundleError(
                "后端 %r 的加载器还没实现（引擎当前只实现了 cuda / cpu / vulkan / python）。"
                "修法：临时用 python 档，或在 issue 里说明需要 %r" % (chosen, chosen))
    if verbose:
        print("[tgop] 加载 %s：本机可用=%s，选中=%s" % (os.path.basename(path),
                                                     available, chosen))
    return name, chosen


def inspect(path):
    """只看包里有什么（不加载、不编译）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "找不到插件包：%s。修法：检查路径（扩展名 .tgop），"
            "或用 tools/tgop_pack.py 从目录生成" % path)
    with zipfile.ZipFile(path) as zf:
        m = _load_manifest(zf)
        files = [n for n in zf.namelist() if n != "manifest.json"]
    return {"name": m["name"], "arity": int(m.get("arity", 1)),
            "backends": sorted((m.get("backends") or {})), "files": sorted(files)}
