# -*- coding: utf-8 -*-
"""`tgai_go.train` —— PyTorch 风格的训练 API（第一切片，CPU/numpy 后端）。

设计目标（见 docs/pytorch_compat_api.md）：
    开发者把 `import torch` 改成 `import tgai_go.train as torch`，
    再改函数前缀，就能迁移 —— 从而不吃 PyTorch 的显存税。

本切片（已自证）：
    * 反向模式自动微分（numpy 后端），带广播还原
    * nn.Module / Parameter / Linear / Embedding / RMSNorm / Sequential / ModuleList
    * optim.AdamW（与 torch 语义一致）/ optim.Muon（Newton-Schulz 正交化）
    * functional.cross_entropy / softmax / silu
    * state_dict() / load_state_dict()

M1 CUDA（见 train/cuda.py，set_device("cuda") 之后新建的张量在 GPU 上）：
    * 显存 + cublasSgemm 的 2D matmul（前向和两个反向）
    * functional.cross_entropy（NVRTC 编译内嵌的 tgai_ce.cu；ignore_index 仍走 CPU）

M2 CUDA（同一份公共 API，不改判分台）：
    * nn.RMSNorm：rmsnorm_fwd / rmsnorm_bwd
    * functional.silu 以及同形状 mul：silu_mul_* 与 ewise_mul
    * nn.Embedding（重复下标走 kernel 里的 atomicAdd）
    * 判分台用来反传的全归约 sum

M3 CUDA（同一份 CausalSelfAttention 公式，不改判分台）：
    * reshape（C 顺序重解释）、最后两维交换
    * 3D matmul：按 batch 循环已验收的 cublasSgemm
    * 可广播的 add（因果掩码）、最后一维 softmax
    * 因此 nn.CausalSelfAttention 整层在 GPU 上

M5 CPU（同一份公共 API，浮点一律 fp32）：
    * 工厂：tensor / zeros / ones / full / randn / rand / arange / from_numpy / manual_seed
    * dtype（float32 / int64 等）、device、cuda.is_available
    * permute / transpose(dim0, dim1) / contiguous / view / size / dim
    * 模块级 exp / log / sqrt / abs / clamp / cat / stack / matmul / bmm / einsum
      / max / min / topk / where / save / load
    * matmul 支持任意前导维，因此 nn.Linear 接受 ND 输入

M5 CUDA（判分台 `--cuda` 那 6 个洞；值在 GPU 上算，主机只算形状/步长/偏差校正这几个标量）：
    * 广播 add 的反向：按轴求和，不再把梯度拉回主机
    * 最后两维 matmul + 前导维广播（3D x 2D 的 Linear）。2D、同 batch 的 3D 仍走原来的 sgemm/bmm
    * permute，以及 transpose(dim0, dim1) 的非 2D 路径（反向是逆置换）
    * 最后一维 log_softmax
    * AdamW.step：参数和动量都在设备上。CUDA 张量的 data 是 None，不是「没有梯度」

M4 CUDA（优化器更新留在显存上。主机只算偏差校正、学习率这些标量）：
    * AdamW：weight decay、一阶/二阶动量、偏差校正后的参数更新，一次 kernel 原地写完
    * Muon：动量和 Nesterov 在 kernel 里。Newton-Schulz 的 X@X.T、(X@X.T)@(X@X.T)、
      Y@X 都在设备上做，累加顺序是顺序 FMA（sgemm_mode 1），和 NumPy float32 的
      `@` 逐位相同。cublasSgemm_v2（pedantic）单次相对误差约 3e-8，五轮迭代后
      参数误差超过 1e-7，所以这里不用它。转置守卫在设备上做 transpose2d。
      ||X|| 的平方和在设备上顺序累加，只把这一个标量降回主机做 float32 sqrt

MoE（nn.MoE，CPU / CUDA 同一份模块，激活不整段拉回主机）：
    * gate="sigmoid"：top-k（logit 更大者优先，平局取更小下标），权重 = sigmoid(logit)。
      top_k=1 与 tgai_moeblk.cu 的路由一致：argmax、平局取小下标、门控 = sigmoid。
    * gate="softmax"：先对全部 E 个 logit 做 softmax，再取同一个 top-k，权重就是这些
      softmax 值，不再在 top-k 内归一。反向走标准 softmax VJP，未选中的 logit
      会通过分母拿到梯度；sigmoid 则只有被选中的专家行有路由梯度。
    * 专家：h = silu(x @ gate.T) * (x @ up.T)，y = h @ down.T，再按门控加权求和。
      参数名 router.weight [E,d]、experts.{e}.gate/up.weight [ff,d]、
      experts.{e}.down.weight [d,ff]。
    * CUDA 只把每个专家的 token 计数（E+1 个 int32）读回主机，用来决定发不发 GEMM。
      空专家不发 GEMM，梯度保持 None（等于没贡献）。

尚未做：
    * M5 里仍只在 CPU 上的算子：exp / log / abs / clamp / cat / stack / einsum
      / max / min / topk / where（CUDA 上仍是 NotImplementedError）
    * 默认重算（activation checkpoint）
    * 默认 SDPA 仍物化 P。dh=64 且 causal 的低显存路径要显式 lowmem=True
      （functional.sdpa_lowmem / scaled_dot_product_attention(..., lowmem=True)），
      其它 dh 或非 causal 回退到物化实现。判分台 Dh=8，不会走到这条路径
    * 与 PyTorch 逐位相同的随机数、float64 存储、torch.save 的 zip 格式
"""
from __future__ import annotations

import os
import sys as _sys_mod
import numpy as np

# 模块最后会把 max 绑成 torch.max。这两处要的是内置 max，先留住。
_builtin_max = max
#: 本模块把 `min` 遮蔽成了张量 API（`_api_min`），所以内置 min 必须留一个别名，
#: 否则标量比较会被误当成张量运算（在 CUDA 上会抛 "max/min 尚未实现"）。
_builtin_min = min


def _f32(x):
    """numpy 2 把 Python float 乘进 float32 数组之前做的那一次舍入。"""
    return float(np.float32(x))


def _f32bits(u):
    return np.array([u], np.uint32).view(np.float32)[0]


_LOG2E32 = _f32bits(0x3FB8AA3B)
_LN232 = _f32bits(0x3F317218)
_EXP_C = (
    _f32bits(0x39500D01),
    _f32bits(0x3AB60B61),
    _f32bits(0x3C088889),
    _f32bits(0x3D2AAAAB),
    _f32bits(0x3E2AAAAB),
    np.float32(0.5),
    np.float32(1.0),
    np.float32(1.0),
)
# 1/12! .. 1/0!，|x|<=2.5 时用这条光滑泰勒，有限差分才稳。
_EXP_TAYLOR = (
    _f32bits(0x310F76C8),
    _f32bits(0x32D7322C),
    _f32bits(0x3493F27E),
    _f32bits(0x3638EF1D),
    _f32bits(0x37D00D01),
    _f32bits(0x39500D01),
    _f32bits(0x3AB60B61),
    _f32bits(0x3C088889),
    _f32bits(0x3D2AAAAB),
    _f32bits(0x3E2AAAAB),
    np.float32(0.5),
    np.float32(1.0),
    np.float32(1.0),
)
_EXP_SMOOTH = _f32bits(0x40200000)  # 2.5


def _exp32_taylor(x):
    x = np.asarray(x, np.float32)
    p = np.broadcast_to(_EXP_TAYLOR[0], np.shape(x)).astype(np.float32).copy()
    for c in _EXP_TAYLOR[1:]:
        p = np.float32(np.float32(p * x) + c)
    return p


def _exp32_wide(x):
    x = np.asarray(x, np.float32)
    x = np.clip(x, np.float32(-80.0), np.float32(80.0))
    nf = np.rint(np.float32(x * _LOG2E32)).astype(np.float32)
    t = np.float32(x - np.float32(nf * _LN232))
    p = np.broadcast_to(_EXP_C[0], np.shape(x)).astype(np.float32).copy()
    for c in _EXP_C[1:]:
        p = np.float32(np.float32(p * t) + c)
    bits = p.view(np.uint32).view(np.int32)
    n = nf.astype(np.int32)
    return (bits + n * np.int32(8388608)).view(np.uint32).view(np.float32)


def _exp32(x):
    """与 CUDA exp32_bits 逐位相同。|x|<=2.5 用泰勒，外面用换底。"""
    x = np.asarray(x, np.float32)
    smooth = np.abs(x) <= _EXP_SMOOTH
    # 两边都算，再用 where 选。NaN 的比较是 False，走 wide。
    return np.where(smooth, _exp32_taylor(x), _exp32_wide(x)).astype(np.float32)


def _log32(x):
    """与 CUDA log32_bits 逐位相同。"""
    x = np.asarray(x, np.float32)
    bits = x.view(np.uint32)
    e = ((bits >> 23) & np.uint32(255)).astype(np.int32) - np.int32(127)
    m = ((bits & np.uint32(0x7FFFFF)) | np.uint32(0x3F800000)).view(np.float32)
    num = np.float32(m - np.float32(1.0))
    den = np.float32(m + np.float32(1.0))
    z = np.float32(num / den)
    z2 = np.float32(z * z)
    z3 = np.float32(z2 * z)
    z5 = np.float32(z3 * z2)
    z7 = np.float32(z5 * z2)
    z9 = np.float32(z7 * z2)
    third, fifth = _f32bits(0x3EAAAAAB), _f32bits(0x3E4CCCCD)
    seventh, ninth = _f32bits(0x3E124925), _f32bits(0x3DE38E39)
    s = z
    s = np.float32(s + np.float32(z3 * third))
    s = np.float32(s + np.float32(z5 * fifth))
    s = np.float32(s + np.float32(z7 * seventh))
    s = np.float32(s + np.float32(z9 * ninth))
    ln_m = np.float32(np.float32(2.0) * s)
    return np.float32(ln_m + np.float32(e.astype(np.float32) * _LN232))


def _exp32_deriv(x):
    """exp32 的值，以及它对输入的导数。和 CUDA exp32_pair 同一顺序。"""
    x = np.asarray(x, np.float32)
    x = np.clip(x, np.float32(-80.0), np.float32(80.0))
    nf = np.rint(np.float32(x * _LOG2E32)).astype(np.float32)
    t = np.float32(x - np.float32(nf * _LN232))
    p = np.broadcast_to(_EXP_C[0], np.shape(x)).astype(np.float32).copy()
    dp = np.zeros(np.shape(x), np.float32)
    for c in _EXP_C[1:]:
        dp = np.float32(np.float32(dp * t) + p)
        p = np.float32(np.float32(p * t) + c)
    n = nf.astype(np.int32)
    one_bits = np.float32(1.0).view(np.int32)
    scale = (one_bits + n * np.int32(8388608)).view(np.uint32).view(np.float32)
    y = (p.view(np.uint32).view(np.int32) + n * np.int32(8388608)).view(np.uint32).view(np.float32)
    dy = np.float32(dp * scale)
    return y, dy


def _silu32_pair(v):
    """与 CUDA silu32_pair 同一顺序。返回 (y, dy/dv)。"""
    v = np.asarray(v, np.float32)
    neg = np.float32(v * np.float32(-1.0))
    s, ds_dneg = _exp32_deriv(neg)
    # 用 exp 的数学导数 s，而不是多项式的分段导数。有限差分对这个更稳，
    # GPU 的 silu32_pair 用同一条。
    ds = np.float32(s * np.float32(-1.0))
    den = np.float32(s + np.float32(1.0))
    inv = np.float32(np.float32(1.0) / den)
    y = np.float32(v * inv)
    ninv = np.float32(np.float32(np.float32(-1.0) * inv) * inv)
    dinv = np.float32(ninv * ds)
    d = np.float32(inv + np.float32(v * dinv))
    return y, d


def _ce32(logits, tgt):
    """与 CUDA ce_fwd / ce_bwd 同一套 exp32、log32 和左折叠。"""
    logits = np.ascontiguousarray(logits, np.float32)
    tgt = np.asarray(tgt, np.int64).reshape(-1)
    n, v = logits.shape
    inv_n = np.float32(np.float32(1.0) / np.float32(n))
    acc = np.float32(0.0)
    dl = np.empty((n, v), np.float32)
    one = np.float32(1.0)
    for i in range(int(n)):
        row = logits[i]
        m = np.float32(row[0])
        for j in range(1, int(v)):
            m = np.float32(np.maximum(m, np.float32(row[j])))
        ex = np.empty(int(v), np.float32)
        s = np.float32(0.0)
        for j in range(int(v)):
            e = np.float32(_exp32(np.float32(np.float32(row[j]) - m)))
            ex[j] = e
            s = np.float32(s + e)
        inv_sum = np.float32(one / s)
        term = np.float32(np.float32(m + np.float32(_log32(s))) - np.float32(row[int(tgt[i])]))
        acc = np.float32(acc + term)
        ti = int(tgt[i])
        for j in range(int(v)):
            p = np.float32(ex[j] * inv_sum)
            if j == ti:
                p = np.float32(p - one)
            dl[i, j] = np.float32(p * inv_n)
    loss = np.float32(acc / np.float32(n))
    return loss, dl


def _cross_entropy_fused(logits, tgt):
    loss_v, dl = _ce32(logits.data, tgt)
    out = Tensor(np.asarray(loss_v, np.float32), requires_grad=logits.requires_grad, _parents=(logits,))

    def bw(g, grads):
        if not logits.requires_grad:
            return
        scale = np.asarray(g, np.float32).reshape(-1)[0]
        grad = dl if scale == np.float32(1.0) else np.float32(np.float32(scale) * dl)
        _acc(grads, logits, grad)

    out._backward = bw
    return out

__all__ = [
    "Tensor", "Parameter", "no_grad", "set_device", "get_device",
    "Module", "Linear", "Embedding", "RMSNorm", "Sequential", "ModuleList",
    "CausalSelfAttention", "MoE",
    "functional", "nn", "optim",
    "tensor", "zeros", "ones", "full", "randn", "rand", "arange", "from_numpy",
    "manual_seed", "cat", "stack", "matmul", "bmm", "einsum", "permute",
    "repeat_interleave",
    "exp", "log", "sqrt", "abs", "clamp", "max", "min", "topk", "where",
    "save", "load",
    "dtype", "float32", "float64", "int32", "int64", "long",
    "device", "cuda",
]

# 默认 cpu。只有 set_device("cuda") 成功之后，新建张量才走 GPU。
# CPU 路径不加载 cuda.py。
_DEVICE = "cpu"


def get_device():
    return _DEVICE


_FUTURE_BACKENDS = {
    "vulkan": "Vulkan 后端目前只跑通了引擎推理侧的矩阵乘（docs/PRODUCTION_READINESS.md 2.2），训练运行时未实现",
    "rocm": "ROCm/HIP 后端尚未开始（docs/PRODUCTION_READINESS.md 2.3）",
    "hip": "ROCm/HIP 后端尚未开始（docs/PRODUCTION_READINESS.md 2.3）",
    "musa": "MUSA（摩尔线程）后端尚未开始",
    "metal": "Metal 后端不存在；macOS 上请用 cpu",
}


try:
    from .dist import DataParallel       # 多卡数据并行（world=1 时全是空操作）
except Exception:                       # noqa: BLE001
    DataParallel = None


def set_device(device):
    """切换后端：'cuda'（可带 :0）或 'cpu'。

    「一行切换」的承诺要靠这里：不可用的后端必须给出**明确原因**，
    不能神秘失败。cuda 会立刻初始化驱动 / cuBLAS / NVRTC，失败则抛出。
    """
    global _DEVICE
    if not isinstance(device, str):
        raise TypeError("set_device 需要字符串，收到 %r" % (device,) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    name = device.strip().lower()
    if name.startswith("cuda:"):
        name = "cuda"                      # 与 torch 习惯一致，卡号当前不生效（单卡）
    if name == "cpu":
        _DEVICE = "cpu"
        return
    if name == "cuda":
        _cuda()
        _DEVICE = "cuda"
        return
    if name in _FUTURE_BACKENDS:
        raise NotImplementedError(
            "set_device(%r) 暂不可用：%s。\n当前可用后端：'cuda'、'cpu'。"
            % (device, _FUTURE_BACKENDS[name]) + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    raise ValueError(
        "set_device 只接受 'cuda' 或 'cpu'，收到 %r。\n已知但未实现的后端：%s"
        % (device, "、".join(sorted(_FUTURE_BACKENDS))) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")


_CUDA_INST = None
_CUDA_TOKEN = -1


def _cuda():
    """取当前后端实例（P0 抽象后统一走 backend 工厂，默认 CUDA）。

    热路径上只做一次整数比较：backend 的缓存令牌变了（register/reset）才重建。
    """
    global _CUDA_INST, _CUDA_TOKEN
    from . import backend as _backend
    tok = _backend.cache_token()
    if _CUDA_INST is None or tok != _CUDA_TOKEN:
        _CUDA_INST = _backend.get(check=False)
        _CUDA_TOKEN = tok
    return _CUDA_INST


HINT_MATMUL = "（怎么办：矩阵乘要求 A.shape[-1] == B.shape[0]；若在算 attention 的 q@k^T，用 bmm_nt 而不是先转置）"
HINT_BCAST = "（怎么办：用 .reshape 显式对齐维度，或检查是不是漏了 keepdim=True / 多套了一层 batch 维）"
HINT_CE = "（怎么办：目标必须在 [0, V)；要忽略的位置用 ignore_index（默认 -100）而不是别的数）"


class DType:
    """对应 torch.dtype。浮点运算仍一律存成 float32，见模块说明。"""

    def __init__(self, name, numpy_dtype, floating):
        self.name = name
        self.numpy_dtype = np.dtype(numpy_dtype)
        self.is_floating_point = bool(floating)

    def __repr__(self):
        return "torch." + self.name

    def __eq__(self, other):
        if isinstance(other, DType):
            return self.numpy_dtype == other.numpy_dtype and self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash((self.name, str(self.numpy_dtype)))


dtype = DType
float32 = DType("float32", np.float32, True)
float64 = DType("float64", np.float64, True)
int32 = DType("int32", np.int32, False)
int64 = DType("int64", np.int64, False)
long = int64
_DTYPES = (float32, float64, int32, int64)


def _as_dtype(dt):
    if dt is None:
        return None
    if isinstance(dt, DType):
        return dt
    if dt is int:
        return int64
    if dt is float:
        return float64
    try:
        np_dt = np.dtype(dt)
    except TypeError:
        raise TypeError("不认识的 dtype: %r" % (dt,) + "。修法：本引擎张量统一 float32；需要低精度请用 rt.fp16_gemm，而不是改 dtype")
    for item in _DTYPES:
        if item.numpy_dtype == np_dt:
            return item
    return DType(str(np_dt), np_dt, np.issubdtype(np_dt, np.floating))


def _is_float_dtype(dt):
    wrapped = _as_dtype(dt)
    return wrapped is None or wrapped.is_floating_point


def _numpy_dtype(dt):
    return _as_dtype(dt).numpy_dtype


def _wrap_dtype(np_dt):
    np_dt = np.dtype(np_dt)
    for item in _DTYPES:
        if item.numpy_dtype == np_dt:
            return item
    return DType(str(np_dt), np_dt, np.issubdtype(np_dt, np.floating))


class Device:
    """对应 torch.device。index 只做记录；真正建张量时只用 type（cpu / cuda）。"""

    def __init__(self, type, index=None):
        if isinstance(type, Device):
            if index is not None:
                raise RuntimeError("device() 不能同时从已有 device 再给 index" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
            self.type = type.type
            self.index = type.index
            return
        text = str(type).strip()
        if ":" in text:
            name, idx_s = text.split(":", 1)
            parsed = int(idx_s)
            if index is not None and int(index) != parsed:
                raise RuntimeError("device 字符串里的 index 和参数冲突" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
            self.type = name
            self.index = parsed
        else:
            self.type = text
            self.index = None if index is None else int(index)

    def __repr__(self):
        if self.index is None:
            return "device(type=%r)" % self.type
        return "device(type=%r, index=%s)" % (self.type, self.index)

    def __str__(self):
        if self.index is None:
            return self.type
        return "%s:%s" % (self.type, self.index)

    def __eq__(self, other):
        if isinstance(other, Device):
            return self.type == other.type and self.index == other.index
        if isinstance(other, str):
            return str(self) == other or self.type == other
        return NotImplemented


device = Device


def _device_arg(dev):
    if dev is None:
        return None
    if isinstance(dev, Device):
        return dev.type
    if isinstance(dev, str):
        if ":" in dev:
            return dev.split(":", 1)[0]
        return dev
    raise TypeError("device 需要 str 或 torch.device，收到 %r" % (dev,) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")


class _CudaAPI:
    """对应 torch.cuda 的只读查询。不在 import 时加载驱动。"""

    @staticmethod
    def is_available():
        try:
            _cuda()
        except Exception:
            return False
        return True

    @staticmethod
    def device_count():
        return 1 if _CudaAPI.is_available() else 0

    @staticmethod
    def current_device():
        if not _CudaAPI.is_available():
            raise RuntimeError("CUDA 不可用" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        return 0


cuda = _CudaAPI()

# ─────────────────────────── 自动微分 ───────────────────────────
_GRAD_ENABLED = [True]

# ═══════════════════════════════════════════════════════════════════════
# 执行策略：显存 与 速度 的取舍（用户可切）
# ═══════════════════════════════════════════════════════════════════════
# 设计原则：**同一份代码，四组预设**，不复制实现。算子读这里的值决定行为。
# 实测基准（8 层、d=512、seq2048、top_k=2、共享专家；PyT 同构 = 4,655 MiB / 0.515 s）：
#   speed   物化 + 不检查点  7,693 MiB / 0.53 s   ← 速度已与 PyT 持平，代价是 1.65× 显存
#   extreme 物化 + 检查点    4,299 MiB / 0.98 s   ← 0.92× 显存、1.9× 慢（当前最优折中）
#   memory  块化 + 检查点    3,998 MiB / 2.12 s   ← 0.86× 显存，换 4× 时间
#   standard = extreme（用户判断：可被 2 取代；保留名字以备将来分化）
_POLICY_PRESETS = {
    "standard": dict(attn="materialized", checkpoint=True, attn_tile=512,
                     cache_masks=True, inplace_acc=True, fused_rope=True,
                     empty_cache_every=0),
    "speed":    dict(attn="materialized", checkpoint=False, attn_tile=1024,
                     cache_masks=True, inplace_acc=True, fused_rope=True,
                     empty_cache_every=0),
    "extreme":  dict(attn="materialized", checkpoint=True, attn_tile=512,
                     cache_masks=True, inplace_acc=True, fused_rope=True,
                     empty_cache_every=0),
    "memory":   dict(attn="flash", checkpoint=True, attn_tile=128,
                     cache_masks=True, inplace_acc=False, fused_rope=True,
                     empty_cache_every=1),
}
_POLICY = [dict(_POLICY_PRESETS["standard"])]
_POLICY_NAME = ["standard"]


def set_policy(name):
    """切换执行策略。`name` ∈ {"standard","speed","extreme","memory"}。

    返回切换后的策略字典（调用方可读，但不要就地改；要改请用 `set_policy_value`）。
    """
    if name not in _POLICY_PRESETS:
        raise ValueError(
            "未知策略 %r。修法：用 set_policy('standard'|'speed'|'extreme'|'memory')"
            % (name,))
    _POLICY[0] = dict(_POLICY_PRESETS[name])
    _POLICY_NAME[0] = name
    return dict(_POLICY[0])


def get_policy():
    """当前策略名与取值（只读副本）。"""
    return _POLICY_NAME[0], dict(_POLICY[0])


def set_policy_value(key, value):
    """微调单个策略项（例如只把 attn_tile 调到 256）。返回新策略。"""
    if key not in _POLICY[0]:
        raise ValueError(
            "策略里没有 %r。修法：可用项 %s" % (key, sorted(_POLICY[0])))
    _POLICY[0][key] = value
    return dict(_POLICY[0])


def policy(key, default=None):
    """供算子内部读取策略值。"""
    return _POLICY[0].get(key, default)


class no_grad:
    """对应 torch.no_grad()。"""

    def __enter__(self):
        self._prev = _GRAD_ENABLED[0]
        _GRAD_ENABLED[0] = False
        return self

    def __exit__(self, *exc):
        _GRAD_ENABLED[0] = self._prev
        return False


def _broadcast_shape(sa, sb):
    """右对齐广播后的形状，规则与 numpy 相同，最高 8 维。"""
    sa = tuple(int(s) for s in sa)
    sb = tuple(int(s) for s in sb)
    ndim = len(sa) if len(sa) > len(sb) else len(sb)
    if ndim > 8:
        raise NotImplementedError("CUDA 广播最高 8 维，收到 %s 与 %s" % (sa, sb) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
    pa = (1,) * (ndim - len(sa)) + sa
    pb = (1,) * (ndim - len(sb)) + sb
    out = []
    for x, y in zip(pa, pb):
        if x == y or x == 1:
            out.append(y if x == 1 else x)
        elif y == 1:
            out.append(x)
        else:
            raise ValueError("形状不能广播: %s vs %s" % (sa, sb) + HINT_BCAST + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
    return tuple(out)


def _unbroadcast(grad, shape):
    """把广播后的梯度还原到原形状（torch 的 sum_to_size 语义）。"""
    if grad.shape == shape:
        return grad
    # 去掉多出来的前导维度
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 对长度为 1 的维度求和并保持维度
    for axis, n in enumerate(shape):
        if n == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


def _norm_shape(shape, total):
    """把 reshape 里的 -1 解析成具体维度（与 torch 语义一致）。

    PyTorch 用户天天写 `x.reshape(-1, D)`；以前这里直接把 -1 传给运行时，
    会报 "reshape 不支持负数维度"。最多允许一个 -1。
    """
    shape = tuple(int(x) for x in shape)
    neg = [i for i, v in enumerate(shape) if v == -1]
    if not neg:
        return shape
    if len(neg) > 1:
        raise ValueError("reshape 最多只能有一个 -1，收到 %s" % (shape,) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
    known = 1
    for v in shape:
        if v != -1:
            known *= v
    if known <= 0 or total % known:
        raise ValueError("reshape 无法把 %d 个元素拆成 %s" % (total, shape) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
    out = list(shape)
    out[neg[0]] = total // known
    return tuple(out)


class Tensor:
    """最小可用的张量：数据 + 梯度 + 反向闭包。"""

    __slots__ = ("data", "grad", "requires_grad", "_parents", "_bwd", "device", "_gpu",
                 "_lazy_T", "_lazy_shape")
    __array_priority__ = 1000        # 让 ndarray <op> Tensor 走我们的 __r*__

    @property
    def _backward(self):
        """反向闭包。**受梯度开关控制**：no_grad 下赋值被丢弃。

        原因：算子构造出输出后会 `out._backward = bw`，而 `bw` 捕获了输入张量。
        若这里存下来，保留一个输出就把整条输入链钉住 —— 实测 no_grad 下每保留一个
        131 KB 的输出，live 涨约 1.4 MB（祖先链）；flash 反向因此随 seq 二次增长
        （75 → 228 → 869 MiB）。PyTorch 里对应的是 `GradMode` 关掉后根本不记录 grad_fn。
        """
        try:
            return self._bwd
        except AttributeError:
            return None

    @_backward.setter
    def _backward(self, v):
        self._bwd = v if _GRAD_ENABLED[0] else None

    def __init__(self, data, requires_grad=False, _parents=(), _backward=None, device=None, dtype=None):
        if device is not None and not isinstance(device, str):
            device = _device_arg(device)
        dev = _DEVICE if device is None else device
        self.device = dev
        self._gpu = None
        self.grad = None
        self.requires_grad = bool(requires_grad)
        self._parents = _parents
        self._backward = _backward
        self._lazy_T = None
        self._lazy_shape = None
        if dev == "cpu":
            # 整数/布尔只当显式 dtype 给出时才保留。默认路径与原来一样，强制 float32。
            if dtype is not None and not _is_float_dtype(dtype):
                if requires_grad:
                    raise RuntimeError("只有浮点张量可以 requires_grad" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
                self.data = np.asarray(data, dtype=_numpy_dtype(dtype))
                return
            self.data = np.asarray(data, dtype=np.float32)
            return
        if dev != "cuda":
            raise ValueError("Tensor device 只接受 cpu/cuda，收到 %r" % (dev,) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        arr = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
        self.data = None
        self._gpu = _cuda().h2d(arr)

    @classmethod
    def _from_gpu(cls, gpu, requires_grad=False, _parents=(), _backward=None):
        obj = cls.__new__(cls)
        obj.device = "cuda"
        obj._gpu = gpu
        obj.data = None
        obj.grad = None
        # ★ 照 PyTorch 的语义（torch/csrc/autograd/VariableTypeUtils.h:49）：
        #   `if (!requires_grad || !GradMode::is_enabled()) return false;`
        #   没开梯度模式时一律不要求梯度 ⇒ 上层算子不会建图 ⇒ no_grad 才真的省内存。
        #   原先只信调用方传的值，导致 `with T.no_grad()` 形同虚设（实测：图照样建，
        #   显存与建图模式完全相同，检查点因此两次失败）。
        # ★ 没开梯度模式时，**连 `_parents` 都不要记录**。
        #   只把 requires_grad 置 False 是不够的：张量仍持有输入引用，
        #   于是"保留一个输出"就把整条输入链钉住不放。
        #   实测：no_grad 下保留 200 个 131 KB 的输出，live 涨 279.8 MB（每个拖 ~1.4 MB 的祖先）；
        #   局部改法（保留 1 个/对）会让 flash 反向的峰值随 seq **二次**增长（75→228→869 MiB）。
        _grad_on = _GRAD_ENABLED[0]
        obj.requires_grad = bool(requires_grad) and _grad_on
        obj._parents = _parents if (_grad_on and obj.requires_grad) else ()
        obj._backward = _backward
        obj._lazy_T = None
        obj._lazy_shape = None
        return obj

    def _cpu_only(self, op):
        if self.device == "cuda":
            raise NotImplementedError("CUDA 后端还没有 %s" % op + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")

    # ---- 属性 ----
    @property
    def shape(self):
        lazy = getattr(self, "_lazy_shape", None)
        if lazy is not None:
            return lazy
        if self.device == "cuda":
            return self._gpu.shape
        return self.data.shape

    @property
    def ndim(self):
        if self.device == "cuda":
            return len(self.shape)
        return self.data.ndim

    @property
    def dtype(self):
        if self.data is None:
            return float32
        return _wrap_dtype(self.data.dtype)

    def numel(self):
        if self.device == "cuda":
            n = 1
            for s in self.shape:
                n *= int(s)
            return int(n)
        return self.data.size

    def numpy(self):
        if self.device == "cuda":
            return self.to_numpy()
        return self.data

    def __repr__(self):
        if self.device == "cuda":
            return "Tensor(shape=%s, device=cuda, requires_grad=%s)" % (self.shape, self.requires_grad)
        return "Tensor(shape=%s, requires_grad=%s)" % (self.data.shape, self.requires_grad)

    # ---- 反向 ----
    def backward(self, grad=None):
        if self.device == "cuda":
            return self._backward_cuda(grad)
        if not self.requires_grad:
            raise RuntimeError("调用 backward 的张量不参与求导" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        if grad is None:
            grad = np.ones_like(self.data)
        # 拓扑排序（迭代，避免深图递归爆栈）
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in seen:
                    stack.append((p, False))
        grads = {id(self): np.asarray(grad, dtype=np.float32)}
        for node in reversed(topo):
            g = grads.get(id(node))
            if g is None:
                continue
            is_leaf = node._backward is None
            if is_leaf:
                # 只有叶子对外保留 .grad（PyTorch 语义）；中间节点不再建一份全尺寸梯度
                if node.grad is None:
                    node.grad = np.array(g, copy=True)
                else:
                    node.grad = node.grad + g
            elif _GRAD_ENABLED[0]:
                node._backward(g, grads)
            # ★ 消费完立刻释放：否则所有中间梯度在反向期间常驻，峰值≈2 倍
            if not is_leaf:
                grads.pop(id(node), None)

    def zero_grad(self):
        self.grad = None

    def detach(self):
        # CPU：和原来一样拷贝 data。CUDA：data 是 None，经 to_numpy 拉回主机。
        if self.device == "cuda":
            return Tensor(self.to_numpy().copy())
        return Tensor(self.data.copy())

    # ---- 算子 ----
    def __add__(self, other):
        if self.device == "cuda" or (
                isinstance(other, Tensor) and getattr(other, "device", None) == "cuda"):
            return self._add_cuda(other)
        other = _as_tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=(self.requires_grad or other.requires_grad),
                     _parents=(self, other))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, _unbroadcast(g, self.shape))
            if other.requires_grad:
                _acc(grads, other, _unbroadcast(g, other.shape))
        out._backward = bw
        return out

    __radd__ = __add__

    def __mul__(self, other):
        if self.device == "cuda" or (
                isinstance(other, Tensor) and getattr(other, "device", None) == "cuda"):
            return self._mul_cuda(other)
        other = _as_tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=(self.requires_grad or other.requires_grad),
                     _parents=(self, other))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, _unbroadcast(g * other.data, self.shape))
            if other.requires_grad:
                _acc(grads, other, _unbroadcast(g * self.data, other.shape))
        out._backward = bw
        return out

    def _add_cuda(self, other):
        if not isinstance(other, Tensor):
            other = _as_tensor(other)
        if self.device != "cuda" or other.device != "cuda":
            raise RuntimeError("add 不能混用 CPU 和 CUDA 张量" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        rt = _cuda()
        buf = rt.add(self._gpu, other._gpu)
        rg = self.requires_grad or other.requires_grad
        out = Tensor._from_gpu(buf, requires_grad=rg, _parents=(self, other))
        a_shape, b_shape = tuple(self.shape), tuple(other.shape)
        out_shape = tuple(buf.shape)

        def bw(g, grads):
            if self.requires_grad:
                src = g if a_shape == out_shape else rt.unbroadcast(g, a_shape)
                _acc(grads, self, src)
            if other.requires_grad:
                src = g if b_shape == out_shape else rt.unbroadcast(g, b_shape)
                _acc(grads, other, src)
        out._backward = bw
        return out

    def _scale_cuda(self, scalar):
        rt = _cuda()
        buf = rt.scale(self._gpu, float(scalar))
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.scale(g, float(scalar)))
        out._backward = bw
        return out

    def _mul_cuda(self, other):
        rt = _cuda()
        if not isinstance(other, Tensor):
            if isinstance(other, (int, float, np.floating, np.integer)) and not isinstance(other, bool):
                return self._scale_cuda(float(other))
            other = _as_tensor(other)
        if self.device != "cuda" or other.device != "cuda":
            raise RuntimeError("mul 不能混用 CPU 和 CUDA 张量" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        a_shape = tuple(int(s) for s in self.shape)
        b_shape = tuple(int(s) for s in other.shape)
        a, b = self._gpu, other._gpu
        if a_shape == b_shape:
            buf = rt.mul(a, b)
            out_shape = a_shape
        else:
            out_shape = _broadcast_shape(a_shape, b_shape)
            aa = a if a_shape == out_shape else rt.broadcast_to(a, out_shape)
            bb = b if b_shape == out_shape else rt.broadcast_to(b, out_shape)
            buf = rt.mul(aa, bb)
        rg = self.requires_grad or other.requires_grad
        out = Tensor._from_gpu(buf, requires_grad=rg, _parents=(self, other))

        def bw(g, grads):
            if self.requires_grad:
                rhs = b if b_shape == out_shape else rt.broadcast_to(b, out_shape)
                prod = rt.mul(g, rhs)
                src = prod if a_shape == out_shape else rt.unbroadcast(prod, a_shape)
                _acc(grads, self, src)
            if other.requires_grad:
                lhs = a if a_shape == out_shape else rt.broadcast_to(a, out_shape)
                prod = rt.mul(g, lhs)
                src = prod if b_shape == out_shape else rt.unbroadcast(prod, b_shape)
                _acc(grads, other, src)
        out._backward = bw
        return out

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-_as_tensor(other))

    def __rsub__(self, other):
        return _as_tensor(other) + (-self)

    def __truediv__(self, other):
        other = _as_tensor(other)
        return self * (other ** -1.0)

    def __matmul__(self, other):
        other = _as_tensor(other)
        if self.device == "cuda" or other.device == "cuda":
            if self.device != "cuda" or other.device != "cuda":
                raise RuntimeError("matmul 不能混用 CPU 和 CUDA 张量" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
            return self._matmul_cuda(other)
        rg = self.requires_grad or other.requires_grad
        if self.ndim == 2 and other.ndim == 2:
            out = Tensor(self.data @ other.data, requires_grad=rg, _parents=(self, other))
            a, b = self.data, other.data

            def bw(g, grads):
                if self.requires_grad:
                    _acc(grads, self, g @ b.T)
                if other.requires_grad:
                    _acc(grads, other, a.T @ g)
            out._backward = bw
            return out
        if self.ndim == 3 and other.ndim == 3:
            if self.shape[0] != other.shape[0]:
                raise ValueError("3D matmul 需要相同 batch: %s vs %s"
                                 % (self.shape, other.shape) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
            out = Tensor(np.matmul(self.data, other.data), requires_grad=rg, _parents=(self, other))
            a, b = self.data, other.data

            def bw(g, grads):
                if self.requires_grad:
                    _acc(grads, self, np.matmul(g, np.swapaxes(b, -1, -2)))
                if other.requires_grad:
                    _acc(grads, other, np.matmul(np.swapaxes(a, -1, -2), g))
            out._backward = bw
            return out
        # 任意前导维（nn.Linear 的 ND 输入走这里）。2D x 2D 和 3D x 3D 仍走上面的原路径。
        if self.ndim >= 1 and other.ndim >= 1:
            out = Tensor(np.matmul(self.data, other.data), requires_grad=rg, _parents=(self, other))
            a_data, b_data = self.data, other.data

            def bw(g, grads):
                da, db = _matmul_grad_numpy(g, a_data, b_data)
                if self.requires_grad:
                    _acc(grads, self, da)
                if other.requires_grad:
                    _acc(grads, other, db)
            out._backward = bw
            return out
        raise NotImplementedError("matmul 支持 2D/3D 以及带前导维的矩阵乘，收到 %dD x %dD"
                                   % (self.ndim, other.ndim) + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")

    def _matmul_cuda(self, other):
        rt = _cuda()
        if self.ndim == 3 and other.ndim == 3:
            if self.shape[0] != other.shape[0]:
                raise ValueError("3D matmul 需要相同 batch: %s vs %s"
                                 % (self.shape, other.shape) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
            buf = rt.bmm(self._gpu, other._gpu)
            rg = self.requires_grad or other.requires_grad
            out = Tensor._from_gpu(buf, requires_grad=rg, _parents=(self, other))
            a, b = self._gpu, other._gpu

            def bw(g, grads):
                if self.requires_grad:
                    _acc(grads, self, rt.bmm(g, rt.swap_last2(b)))
                if other.requires_grad:
                    _acc(grads, other, rt.bmm(rt.swap_last2(a), g))
            out._backward = bw
            return out
        src = getattr(other, "_lazy_T", None)
        if src is not None and self.ndim == 2 and getattr(src, "ndim", 0) == 2:
            M, K = int(self.shape[0]), int(self.shape[1])
            N, K2 = int(src.shape[0]), int(src.shape[1])
            if K != K2:
                raise ValueError("matmul K 不匹配: %s @ %s^T" % (self.shape, src.shape) + HINT_MATMUL + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
            if not rt._numpy_exact_gemm(M, N, K):
                # ★ fp16 前向 GEMM（V100 Tensor Core）：只在 rt.fp16_gemm 打开时走。
                #   前向用 fp16 计算（fp32 累加），**反向仍用 fp32 的 sgemm/gemm_tn**，
                #   所以梯度路径一个字节都没变，只有前向的输入被舍入到 fp16。
                #   实测（V100）：mlp 4.33x、lm_head 5.21x，relerr ~3e-04（fp16 舍入）。
                # 只对"赚"的形状开：K>=512 且 N>=512。实测 attn_kv(N=128) fp16 反而慢 0.64x。
                if (getattr(rt, "fp16_gemm", False) and K >= 512 and N >= 512):
                    # 零转换路径：权重影子只建一次（step 后失效）、激活 cast 进复用 scratch、
                    # 直接 cublasGemmEx（fp16 输入 + fp32 输出），跳过 _sgemm_f16 内部那两次转换。
                    w16 = _fp16_shadow(src._gpu)
                    a16 = _fp16_scratch(self._gpu)
                    rt.cast_f32_to_f16_into(self._gpu, a16)
                    buf = rt.gemm_f16_nt_raw(a16, w16, M, N, K)
                    try:
                        _FP16_HITS[0] += 1
                    except Exception:
                        pass
                else:
                    buf = rt.gemm_nt(self._gpu, src._gpu)
                rg = self.requires_grad or src.requires_grad
                out = Tensor._from_gpu(buf, requires_grad=rg, _parents=(self, src) if rg else ())
                a, w = self._gpu, src._gpu

                def bw(g, grads, _a=a, _w=w, _self=self, _src=src,
                       _M=M, _N=N, _K=K):
                    # dA = g @ W，dW = g^T @ A。不再经过物化的 W^T。
                    # fp16 版：g/a 转进复用 scratch，W 用按代缓存的影子；
                    # 形状不划算（结果维度 <512）或出错则回退 fp32。
                    use16 = (getattr(rt, "fp16_gemm", False)
                             and _N >= 512 and _K >= 512)
                    g16 = None
                    if use16:
                        try:
                            g16 = _fp16_scratch(g)
                            rt.cast_f32_to_f16_into(g, g16)
                        except Exception:
                            g16 = None
                    if _self.requires_grad:
                        if g16 is not None and _K >= 512:
                            try:
                                _acc(grads, _self,
                                     rt.gemm_f16_nn_raw(g16, _fp16_shadow(_w),
                                                        _M, _N, _K))
                            except Exception:
                                _acc(grads, _self, rt.sgemm(g, _w))
                        else:
                            _acc(grads, _self, rt.sgemm(g, _w))
                    if _src.requires_grad:
                        if g16 is not None:
                            try:
                                a16 = _fp16_scratch(_a)
                                rt.cast_f32_to_f16_into(_a, a16)
                                _acc(grads, _src,
                                     rt.gemm_f16_tn_raw(g16, a16, _M, _N, _K))
                            except Exception:
                                _acc(grads, _src, rt.gemm_tn(g, _a))
                        else:
                            _acc(grads, _src, rt.gemm_tn(g, _a))
                out._backward = bw
                return out
            buf_t = rt.transpose2d(src._gpu)
            other = Tensor._from_gpu(
                buf_t, requires_grad=src.requires_grad,
                _parents=(src,) if src.requires_grad else ())

            def bw_t(g, grads, _src=src):
                if _src.requires_grad:
                    _acc(grads, _src, rt.transpose2d(g))
            other._backward = bw_t
        if self.ndim != 2 or other.ndim != 2:
            if self.ndim >= 2 and other.ndim >= 2:
                return self._matmul_cuda_nd(other)
            raise NotImplementedError("CUDA matmul 目前只支持至少 2D，收到 %dD x %dD"
                                      % (self.ndim, other.ndim) + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
        buf = rt.sgemm(self._gpu, other._gpu)
        rg = self.requires_grad or other.requires_grad
        out = Tensor._from_gpu(buf, requires_grad=rg, _parents=(self, other))
        a, b = self._gpu, other._gpu

        def bw(g, grads):
            # dA = g @ B^T ，dB = A^T @ g。和 CPU 的 g @ b.T / a.T @ g 同一套形状。
            if self.requires_grad:
                _acc(grads, self, rt.gemm_nt(g, b))
            if other.requires_grad:
                _acc(grads, other, rt.gemm_tn(a, g))
        out._backward = bw
        return out

    def _materialize_lazy_T(self):
        """把惰性转置还原成一个真实节点（带标准 transpose 反向）。

        `.T` 默认是惰性的（只为 2D matmul 省一次拷贝）。N 维 matmul 读的是原始缓冲区，
        不认识惰性状态，所以这条路径必须先实体化 —— 否则形状对不上
        （实测症状：`linear_nd` 报 matmul K 不匹配 (2,3,16) @ (8,16)）。
        """
        rt = _cuda()
        base = self._lazy_T
        buf = rt.transpose2d(base._gpu)
        out = Tensor._from_gpu(buf, requires_grad=base.requires_grad, _parents=(base,))

        def bw(g, grads):
            if base.requires_grad:
                _acc(grads, base, rt.transpose2d(g))

        out._backward = bw
        return out

    def _matmul_cuda_nd(self, other):
        """最后两维 matmul，前导维广播。反向先乘再 unbroadcast 回原形状。"""
        rt = _cuda()
        # 惰性转置在这条路径上必须先落地（原缓冲区形状是转置前的）
        if getattr(self, "_lazy_T", None) is not None:
            self = self._materialize_lazy_T()
        if getattr(other, "_lazy_T", None) is not None:
            other = other._materialize_lazy_T()
        buf = rt.matmul_general(self._gpu, other._gpu)
        rg = self.requires_grad or other.requires_grad
        out = Tensor._from_gpu(buf, requires_grad=rg, _parents=(self, other))
        a, b = self._gpu, other._gpu
        a_shape = tuple(int(s) for s in self.shape)
        b_shape = tuple(int(s) for s in other.shape)

        def bw(g, grads):
            if self.requires_grad:
                da = rt.matmul_general(g, rt.swap_last2(b))
                if tuple(int(s) for s in da.shape) != a_shape:
                    da = rt.unbroadcast(da, a_shape)
                _acc(grads, self, da)
            if other.requires_grad:
                db = rt.matmul_general(rt.swap_last2(a), g)
                if tuple(int(s) for s in db.shape) != b_shape:
                    db = rt.unbroadcast(db, b_shape)
                _acc(grads, other, db)
        out._backward = bw
        return out

    def _transpose_cuda(self):
        if self.ndim != 2:
            raise NotImplementedError("CUDA transpose 目前只支持 2D，收到 %dD" % self.ndim + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
        # 不物化。matmul 看见 _lazy_T 就走 gemm_nt，省掉词表那么大的转置。
        # 真要读数值时 to_numpy 再转。小矩阵的精确 GEMM 会在 matmul 里物化。
        rt = _cuda()
        out = Tensor._from_gpu(self._gpu, requires_grad=self.requires_grad, _parents=(self,))
        out._lazy_T = self
        out._lazy_shape = (int(self.shape[1]), int(self.shape[0]))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.transpose2d(g))
        out._backward = bw
        return out

    def _backward_cuda(self, grad):
        if not self.requires_grad:
            raise RuntimeError("调用 backward 的张量不参与求导" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        rt = _cuda()
        if grad is None:
            g0 = rt.full(self.shape, 1.0)
        elif isinstance(grad, Tensor):
            if grad.device != "cuda":
                g0 = rt.h2d(np.asarray(grad.to_numpy(), dtype=np.float32).reshape(self.shape))
            else:
                g0 = grad._gpu
        elif isinstance(grad, np.ndarray) or isinstance(grad, (list, tuple, int, float)):
            g0 = rt.h2d(np.asarray(grad, dtype=np.float32).reshape(self.shape))
        else:
            # 设备缓冲（GpuBuf）：直接用，不要 np.asarray —— 那会去调 float(GpuBuf)。
            # `T.checkpoint` 的反向重算就是这条路（它手里只有缓冲）。
            g0 = grad
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in seen:
                    stack.append((p, False))
        grads = {id(self): g0}
        for node in reversed(topo):
            g = grads.get(id(node))
            if g is None:
                continue
            if getattr(node, "device", "cpu") != "cuda":
                raise RuntimeError("CUDA 反传遇到 CPU 张量" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
            # 只给叶子张量留 .grad，和 PyTorch 一样。中间激活的梯度以前每次都 clone
            # 一份，seq=8192 上这 100 多次拷贝既占时间也把显存顶过 13 GiB。
            # 叶子的 g 是父算子刚写出来的新缓冲，反向不再改它，可以直接挂上，不必再拷。
            if node._backward is None:
                if node.grad is None:
                    node.grad = Tensor._from_gpu(g, requires_grad=False)
                else:
                    if not isinstance(node.grad, Tensor) or getattr(node.grad, "device", "cpu") != "cuda":
                        raise RuntimeError("CUDA 张量的 .grad 必须是 CUDA Tensor" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
                    node.grad = Tensor._from_gpu(rt.add(node.grad._gpu, g), requires_grad=False)
            if node._backward is not None:
                if _GRAD_ENABLED[0]:
                    node._backward(g, grads)
                # ★ 消费完立刻释放该节点的输出梯度：否则所有中间梯度在反向期间常驻，
                #   峰值 ≈ 激活 + 同尺寸梯度 = 2 倍（子智能体台账：中间梯度 11.23 GiB）。
                #   PyTorch 对应 node.h::release_variables。叶子已在上面挂进 .grad，不删。
                grads.pop(id(node), None)
        # 捕获期间不能同步（cuStreamSynchronize 在 capture 中返回 rc=900）。
        # 图重放本身会保证顺序，所以捕获时跳过这一步是安全的。
        if not getattr(rt, "_capturing", False):
            rt.sync()

    def swap_last2(self):
        """交换最后两维（attention 里对 k 用）。"""
        if self.device == "cuda":
            return self._swap_last2_cuda()
        out = Tensor(np.swapaxes(self.data, -1, -2),
                     requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            _acc(grads, self, np.swapaxes(g, -1, -2))
        out._backward = bw
        return out

    def _swap_last2_cuda(self):
        if self.ndim < 2:
            raise NotImplementedError("swap_last2 至少要 2 维，收到 %dD" % self.ndim + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
        rt = _cuda()
        buf = rt.swap_last2(self._gpu)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            _acc(grads, self, rt.swap_last2(g))
        out._backward = bw
        return out

    def __pow__(self, p):
        if self.device == "cuda":
            return self._pow_cuda(p)
        out = Tensor(self.data ** p, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            _acc(grads, self, g * p * (self.data ** (p - 1)))
        out._backward = bw
        return out

    def _pow_cuda(self, p):
        """标量指数。前向 powf，反向 g * p * x**(p-1)，与 CPU 同一公式。"""
        if isinstance(p, Tensor):
            raise NotImplementedError("CUDA pow 的指数目前只支持标量，不支持张量" + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
        if isinstance(p, (bool, np.bool_)):
            raise TypeError("pow 的指数不能是 bool" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        if not isinstance(p, (int, float, np.floating, np.integer)):
            raise NotImplementedError("CUDA pow 的指数目前只支持标量，收到 %r" % (type(p).__name__,) + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
        pf = float(p)
        rt = _cuda()
        x = self._gpu
        buf = rt.pow_scalar(x, pf)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))
        in_shape = tuple(int(s) for s in self.shape)

        def bw(g, grads, _p=pf, _x=x, _shape=in_shape):
            if not self.requires_grad:
                return
            # p==0 时 x**(p-1) 会变成无穷，梯度按定义是 0。
            if _p == 0.0:
                _acc(grads, self, rt.zeros(_shape))
                return
            xp = rt.pow_scalar(_x, _p - 1.0)
            _acc(grads, self, rt.scale(rt.mul(g, xp), _p))
        out._backward = bw
        return out

    def exp(self):
        if self.device == "cuda":
            return self._exp_cuda()
        e = np.exp(self.data)
        out = Tensor(e, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            _acc(grads, self, g * e)
        out._backward = bw
        return out

    def _exp_cuda(self):
        rt = _cuda()
        buf = rt.exp(self._gpu)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))
        y = buf

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.mul(g, y))
        out._backward = bw
        return out

    # ---- 0.9.16：把原本 CPU-only 的算子接到 C 端内核 ----
    def _abs_cuda(self):
        rt = _cuda()
        x = self._gpu
        buf = rt.abs(x)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.abs_bwd(x, g))
        out._backward = bw
        return out

    def _cos_cuda(self):
        rt = _cuda()
        x = self._gpu
        buf = rt.cos(x)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad,
                               _parents=(self,))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.cos_bwd(x, g))
        out._backward = bw
        return out

    def _sin_cuda(self):
        rt = _cuda()
        x = self._gpu
        buf = rt.sin(x)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad,
                               _parents=(self,))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.sin_bwd(x, g))
        out._backward = bw
        return out

    def _relu_cuda(self):
        rt = _cuda()
        x = self._gpu
        buf = rt.relu(x)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.relu_bwd(x, g))
        out._backward = bw
        return out

    def _clamp_cuda(self, lo, hi):
        rt = _cuda()
        x = self._gpu
        lo = float("-inf") if lo is None else float(lo)
        hi = float("inf") if hi is None else float(hi)
        buf = rt.clamp(x, lo, hi)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.clamp_bwd(x, g, lo, hi))
        out._backward = bw
        return out

    def _log_cuda(self):
        rt = _cuda()
        x = self._gpu
        buf = rt.log(x)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.div(g, x))
        out._backward = bw
        return out

    def log(self):
        if self.device == "cuda":
            return self._log_cuda()
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            _acc(grads, self, g / self.data)
        out._backward = bw
        return out

    def sum(self, axis=None, keepdims=False):
        if self.device == "cuda":
            if axis is None and not keepdims:
                return self._sum_cuda()
            return self._sum_axis_cuda(axis, keepdims)
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad, _parents=(self,))
        shape = self.shape

        def bw(g, grads):
            gg = g
            if axis is not None and not keepdims:
                gg = np.expand_dims(gg, axis)
            _acc(grads, self, np.broadcast_to(gg, shape).copy())
        out._backward = bw
        return out

    def _sum_cuda(self):
        rt = _cuda()
        buf = rt.sum_all(self._gpu)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))
        in_shape = self.shape

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.full(in_shape, rt.read_scalar(g)))
        out._backward = bw
        return out

    def _sum_axis_cuda(self, axis, keepdims):
        """单轴求和，或 keepdims 的全归约。反向把梯度广播回原形状。"""
        rt = _cuda()
        shape = tuple(int(s) for s in self.shape)
        nd = len(shape)
        if isinstance(axis, (tuple, list)):
            raise NotImplementedError("CUDA sum 目前只支持单轴或全归约，收到 %r" % (axis,) + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
        if axis is None:
            buf = rt.sum_all(self._gpu)
            if keepdims and nd:
                buf = rt.reshape(buf, (1,) * nd)
            out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))

            def bw_all(g, grads):
                if self.requires_grad:
                    _acc(grads, self, rt.full(shape, rt.read_scalar(g)))
            out._backward = bw_all
            return out
        ax = int(axis)
        if ax < 0:
            ax += nd
        if nd == 0 or ax < 0 or ax >= nd:
            raise ValueError("sum 轴 %s 越界（shape=%s）" % (axis, shape) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
        buf = rt.sum_axis(self._gpu, ax, keepdims=bool(keepdims))
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))
        dropped = not bool(keepdims)

        def bw(g, grads, _ax=ax, _drop=dropped, _shape=shape):
            if not self.requires_grad:
                return
            gg = g
            if _drop:
                gs = [int(s) for s in g.shape]
                gs.insert(_ax, 1)
                gg = rt.reshape(g, tuple(gs))
            if tuple(int(s) for s in gg.shape) != _shape:
                gg = rt.broadcast_to(gg, _shape)
            _acc(grads, self, gg)
        out._backward = bw
        return out

    def mean(self, axis=None, keepdims=False, dim=None, keepdim=None):
        # 接受 torch 拼写（dim/keepdim），便于'改个导入就迁移'
        if dim is not None:
            axis = dim
        if keepdim is not None:
            keepdims = keepdim
        if self.device == "cuda":
            if isinstance(axis, (tuple, list)):
                raise NotImplementedError("CUDA mean 目前只支持单轴或全归约，收到 %r" % (axis,) + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
            shape = tuple(int(s) for s in self.shape)
            if axis is None:
                n = 1
                for s in shape:
                    n *= int(s)
            else:
                ax = int(axis)
                if ax < 0:
                    ax += len(shape)
                if not shape or ax < 0 or ax >= len(shape):
                    raise ValueError("mean 轴 %s 越界（shape=%s）" % (axis, shape) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
                n = int(shape[ax])
            if n == 0:
                raise ValueError("mean 的归约长度是 0（shape=%s, axis=%s）" % (shape, axis) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
            return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(n))
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = _norm_shape(shape, int(np.prod(self.shape)))
        if self.device == "cuda":
            return self._reshape_cuda(shape)
        new = self.data.reshape(shape)
        if not np.issubdtype(self.data.dtype, np.floating):
            return Tensor(new, dtype=self.dtype, requires_grad=False)
        out = Tensor(new, requires_grad=self.requires_grad, _parents=(self,))
        old = self.shape

        def bw(g, grads):
            _acc(grads, self, g.reshape(old))
        out._backward = bw
        return out

    def _reshape_cuda(self, shape):
        rt = _cuda()
        buf = rt.reshape(self._gpu, shape)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))
        old = self.shape

        def bw(g, grads):
            _acc(grads, self, rt.reshape(g, old))
        out._backward = bw
        return out

    def transpose(self, dim0=None, dim1=None):
        # 不传参数：原来的完整转置（.T 走这里）。2D 上与 transpose(0, 1) 相同。
        if dim0 is None and dim1 is None:
            if self.device == "cuda":
                return self._transpose_cuda()
            out = Tensor(self.data.T, requires_grad=self.requires_grad, _parents=(self,))

            def bw(g, grads):
                _acc(grads, self, g.T)
            out._backward = bw
            return out
        if dim0 is None or dim1 is None:
            raise TypeError("transpose() 要么不传参数，要么同时给 dim0 和 dim1" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        if self.device == "cuda":
            nd = self.ndim
            a = int(dim0) + (nd if int(dim0) < 0 else 0)
            b = int(dim1) + (nd if int(dim1) < 0 else 0)
            if nd == 2 and {a, b} == {0, 1}:
                return self._transpose_cuda()
            # 更高维走 permute。下面和 CPU 共用轴交换。
        nd = self.ndim
        a = int(dim0)
        b = int(dim1)
        if a < 0:
            a += nd
        if b < 0:
            b += nd
        axes = list(range(nd))
        axes[a], axes[b] = axes[b], axes[a]
        return self.permute(*axes)

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        nd = self.ndim
        axes = []
        for d in dims:
            d = int(d)
            if d < 0:
                d += nd
            axes.append(d)
        if sorted(axes) != list(range(nd)):
            raise ValueError("permute 必须是 0..%d 的一个排列，收到 %s" % (nd - 1, axes) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        if self.device == "cuda":
            return self._permute_cuda(tuple(axes))
        moved = np.transpose(self.data, axes)
        if not np.issubdtype(self.data.dtype, np.floating):
            return Tensor(moved, dtype=self.dtype, requires_grad=False)
        out = Tensor(moved, requires_grad=self.requires_grad, _parents=(self,))
        inv = tuple(int(i) for i in np.argsort(axes))

        def bw(g, grads):
            _acc(grads, self, np.transpose(g, inv))
        out._backward = bw
        return out

    def _permute_cuda(self, axes):
        rt = _cuda()
        buf = rt.permute(self._gpu, axes)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))
        inv = [0] * len(axes)
        for i, src in enumerate(axes):
            inv[src] = i
        inv = tuple(inv)

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, rt.permute(g, inv))
        out._backward = bw
        return out

    def contiguous(self):
        if self.device == "cuda":
            return self
        if self.data.flags.c_contiguous:
            return self
        packed = np.ascontiguousarray(self.data)
        if not np.issubdtype(self.data.dtype, np.floating):
            return Tensor(packed, dtype=self.dtype, requires_grad=False)
        out = Tensor(packed, requires_grad=self.requires_grad, _parents=(self,))

        def bw(g, grads):
            _acc(grads, self, np.asarray(g, dtype=np.float32).reshape(self.shape))
        out._backward = bw
        return out

    def view(self, *shape):
        # 近似：非连续时先拷贝。PyTorch 的 view 在非连续上会直接报错。
        if self.device == "cuda":
            return self.reshape(*shape)
        if self.data.flags.c_contiguous:
            return self.reshape(*shape)
        return self.contiguous().reshape(*shape)

    def size(self, dim=None):
        if dim is None:
            return self.shape
        d = int(dim)
        if d < 0:
            d += self.ndim
        return int(self.shape[d])

    def dim(self):
        return int(self.ndim)

    def to(self, device):
        """把张量搬到指定设备（`"cpu"` / `"cuda"`）。

        注意：这是**叶子级**搬迁，不保留 autograd 链路 —— 用于常量、输入、参考模型的 logp 等；
        不要把需要梯度的中间量 `.to()` 后继续反传。
        """
        dev = str(device)
        if dev == self.device:
            return self
        if dev == "cpu":
            return Tensor(np.asarray(self.to_numpy(), np.float32),
                          requires_grad=self.requires_grad)
        rt = _cuda()
        buf = rt.h2d(np.asarray(self.data, np.float32))
        return Tensor._from_gpu(buf, requires_grad=self.requires_grad)

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")

    def abs(self):
        if self.device == "cuda":
            return self._abs_cuda()
        out = Tensor(np.abs(self.data), requires_grad=self.requires_grad, _parents=(self,))
        sign = np.sign(self.data).astype(np.float32)

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, g * sign)
        out._backward = bw
        return out

    def cos(self):
        """余弦。为 RoPE 一类位置编码准备（P100 报告指出的迁移缺口）。"""
        if self.device == "cuda":
            return self._cos_cuda()
        out = Tensor(np.cos(self.data), requires_grad=self.requires_grad,
                     _parents=(self,))
        s = np.sin(self.data).astype(np.float32)

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, -g * s)
        out._backward = bw
        return out

    def sin(self):
        """正弦。为 RoPE 一类位置编码准备。"""
        if self.device == "cuda":
            return self._sin_cuda()
        out = Tensor(np.sin(self.data), requires_grad=self.requires_grad,
                     _parents=(self,))
        c = np.cos(self.data).astype(np.float32)

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, g * c)
        out._backward = bw
        return out

    def sqrt(self):
        return self ** 0.5

    def clamp(self, min=None, max=None):
        if self.device == "cuda":
            return self._clamp_cuda(min, max)
        lo = _scalar_or_none(min, "clamp.min")
        hi = _scalar_or_none(max, "clamp.max")
        y = self.data
        if lo is not None:
            y = np.maximum(y, np.float32(lo))
        if hi is not None:
            y = np.minimum(y, np.float32(hi))
        out = Tensor(y, requires_grad=self.requires_grad, _parents=(self,))
        src = self.data

        def bw(g, grads):
            if not self.requires_grad:
                return
            mask = np.ones(src.shape, dtype=np.float32)
            if lo is not None:
                mask = np.where(src < lo, np.float32(0), mask)
            if hi is not None:
                mask = np.where(src > hi, np.float32(0), mask)
            _acc(grads, self, g * mask)
        out._backward = bw
        return out

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, idx):
        if self.device == "cuda":
            return self._getitem_cuda(idx)
        out = Tensor(self.data[idx], requires_grad=self.requires_grad, _parents=(self,))
        shape = self.shape

        def bw(g, grads):
            full = np.zeros(shape, dtype=np.float32)
            if _has_array_index(idx):
                np.add.at(full, idx, g)      # 重复下标要累加（Embedding）
            else:
                full[idx] += g               # 纯切片
            _acc(grads, self, full)
        out._backward = bw
        return out

    def _cuda_logical_storage(self):
        """与 self.shape 同行主序的设备缓冲。lazy .T 的 _gpu 还是转置前那块。"""
        lazy = getattr(self, "_lazy_T", None)
        if lazy is None:
            return self._gpu
        return _cuda().transpose2d(lazy._gpu)

    def _getitem_cuda(self, idx):
        rt = _cuda()
        arr = _int_index_array(idx)
        if arr is not None:
            n = int(self.shape[0]) if len(self.shape) else 0
            # 负数按 numpy 语义折回；越界立刻报错，不要下发到设备
            arr = np.where(arr < 0, arr + n, arr)
            if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= n):
                raise IndexError(
                    "数组索引越界：取值范围 [0, %d)，收到 [%d, %d]"
                    % (n, int(arr.min()), int(arr.max())) +
                    "。修法：检查索引数组，或确认首轴长度是 %d" % n)
            return self._array_index_cuda(arr)
        out_shape, base, divs, strides = _slice_gather_plan(self.shape, idx)
        buf = rt.index_gather(self._cuda_logical_storage(), out_shape, base, divs, strides)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))
        in_shape = tuple(int(s) for s in self.shape)

        def bw(g, grads):
            if self.requires_grad:
                # g 是切出来的形状。清零后只加回被选中的位置，其余保持 0。
                _acc(grads, self, rt.index_scatter_add(g, in_shape, base, divs, strides))
        out._backward = bw
        return out

    def _array_index_cuda(self, arr):
        """首轴整型数组索引的设备实现。`arr` 是**已归一化**（无负数、已查越界）的 int64 数组。

        前向走融合核；反向走主机遍历（`np.add.at` 才能正确处理重复下标）。
        反向比融合核慢，但 correctness 优先；embedding 场景请用 `nn.Embedding`。
        """
        rt = _cuda()
        shape = tuple(int(x) for x in self.shape)
        if not shape:
            raise NotImplementedError(
                "0 维张量不支持数组索引。修法：先 reshape 成至少一维")

        n_src = shape[0]
        tail = shape[1:]
        row_size = 1
        for d in tail:
            row_size *= int(d)
        n_rows = int(arr.size)

        idx_buf = rt.h2d(arr.astype("int32"), dtype=__import__("numpy").int32)
        src = self._cuda_logical_storage()
        if len(tail) == 0:
            buf = rt.gather_flat(src, idx_buf, n_src)
            out_shape = (n_rows,)
        else:
            buf = rt.gather_rows(src, idx_buf, n_src, row_size, n_rows)
            out_shape = (n_rows,) + tail
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad,
                               _parents=(self,))
        if tuple(int(x) for x in out.shape) != out_shape:
            out = out.reshape(out_shape)

        def bw(g, grads):
            if not self.requires_grad:
                return
            # 主机中转：g 可能是多维，先摊平成 (n_rows, row_size)
            g_np = np.asarray(g.to_numpy(), np.float32)
            full = np.zeros(shape, dtype=np.float32)
            if row_size == 1 and len(tail) == 0:
                np.add.at(full, arr, g_np.reshape(-1))
            else:
                np.add.at(full, arr, g_np.reshape((n_rows,) + tail))
            _acc(grads, self, rt.h2d(full))
        out._backward = bw
        return out

    def repeat_interleave(self, repeats, dim=None):
        return _repeat_interleave_impl(self, repeats, dim)

    def _repeat_interleave_cuda(self, repeats, dim):
        rt = _cuda()
        out_shape, base, divs, strides = _repeat_gather_plan(self.shape, repeats, dim)
        buf = rt.index_gather(self._cuda_logical_storage(), out_shape, base, divs, strides)
        out = Tensor._from_gpu(buf, requires_grad=self.requires_grad, _parents=(self,))
        in_shape = tuple(int(s) for s in self.shape)

        def bw(g, grads):
            if self.requires_grad:
                # 连续重复的那几份梯度加回同一个源位置。
                # rt.deterministic=True 时走 repeat_reduce（按 j 升序归约、无原子），
                # 保证同一输入两次运行梯度逐位相同。
                if getattr(rt, "deterministic", False):
                    _acc(grads, self, rt.repeat_reduce(g, int(repeats), in_shape, dim))
                else:
                    _acc(grads, self,
                         rt.index_scatter_add(g, in_shape, base, divs, strides))
        out._backward = bw
        return out

    def maximum(self, other):
        other = _as_tensor(other)
        if self.device == "cuda" or other.device == "cuda":
            # CUDA 上不能用 .data 做掩码（CUDA 张量没有主机镜像）。
            # 用 0.5*((x+y) + |x−y|) 组合实现；并列时梯度各 0.5（与 torch 的并列约定略有差异，
            # 已在 docs/REVIEW 里注明）。NaN 输入会自然传播为 NaN。
            return (self + other + (self - other).abs()) * 0.5
        mask = (self.data > other.data).astype(np.float32)
        out = Tensor(np.maximum(self.data, other.data),
                     requires_grad=(self.requires_grad or other.requires_grad),
                     _parents=(self, other))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, _unbroadcast(g * mask, self.shape))
            if other.requires_grad:
                _acc(grads, other, _unbroadcast(g * (1.0 - mask), other.shape))
        out._backward = bw
        return out

    def minimum(self, other):
        """对应 `torch.minimum`。CPU 用掩码；CUDA 用 0.5*((x+y) − |x−y|)。"""
        other = _as_tensor(other)
        if self.device == "cuda" or other.device == "cuda":
            return (self + other - (self - other).abs()) * 0.5
        mask = (self.data < other.data).astype(np.float32)
        out = Tensor(np.minimum(self.data, other.data),
                     requires_grad=(self.requires_grad or other.requires_grad),
                     _parents=(self, other))

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, _unbroadcast(g * mask, self.shape))
            if other.requires_grad:
                _acc(grads, other, _unbroadcast(g * (1.0 - mask), other.shape))
        out._backward = bw
        return out
    def __rtruediv__(self, other):
        """`2.0 / x` —— 之前直接 TypeError。用已有的除法反传，不新增内核。"""
        num = Tensor(np.full(tuple(self.shape), float(other), np.float32))
        return num / self

    def _cmp(self, other, op):
        """比较运算统一实现：返回 **numpy bool 数组**。

        为什么不是 Tensor：我们的 Tensor 只装浮点，装不了 bool；
        而 `where(mask, a, b)` 和 `x[mask]` 都是直接吃 numpy 数组的。
        CUDA 张量这里会 D2H（有拷贝开销），只在控制流里用一次通常可以接受。
        """
        rhs = other.to_numpy() if isinstance(other, Tensor) else other
        return op(self.to_numpy(), rhs)

    def __gt__(self, other):
        return self._cmp(other, np.greater)

    def __ge__(self, other):
        return self._cmp(other, np.greater_equal)

    def __lt__(self, other):
        return self._cmp(other, np.less)

    def __le__(self, other):
        return self._cmp(other, np.less_equal)

    def masked_fill(self, mask, value):
        """`x.masked_fill(mask, v)` —— 掩码处置为 v，其余不变（梯度只在未掩码处流）。"""
        m = mask.to_numpy() if isinstance(mask, Tensor) else np.asarray(mask, dtype=bool)
        if m.shape != tuple(self.shape):
            raise ValueError(
                "masked_fill: mask 形状 %s 与输入 %s 不一致。修法：先 broadcast_to 或 reshape"
                % (tuple(m.shape), tuple(self.shape)))
        if self.device == "cuda":
            # 原实现直接 np.where(..., self.data)，而 CUDA 张量的 self.data 是 None
            # ⇒ 会静默产出 nan（实测：CUDA 上返回 nan 而不是报错）。
            # 这里改成组合实现 x*(1−m) + v*m：只用已有算子，梯度自动只在未掩码处流。
            mt = _as_tensor(np.ascontiguousarray(m.astype(np.float32))).to("cuda")
            return self * (1.0 - mt) + mt * float(value)
        out = Tensor(np.where(m, np.float32(value), self.data),
                     requires_grad=self.requires_grad, _parents=(self,))
        keep = (~m).astype(np.float32)

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, g * keep)
        out._backward = bw
        return out

    def tanh(self):
        """CUDA 上走融合内核；CPU 走组合式（2*sigmoid(2x)-1，numpy 侧无所谓开销）。"""
        if self.device == "cuda":
            return _tanh_cuda(self)
        from . import nn_ops as _O
        return _O.tanh(self)

    def gelu(self, approximate="tanh"):
        from . import nn_ops as _O
        return _O.gelu(self, approximate=approximate)

    def relu(self):
        """max(x,0)。当前是 numpy 实现 ⇒ CUDA 上明确报错（不静默算错）。"""
        if self.device == "cuda":
            return self._relu_cuda()
        out = Tensor(np.maximum(self.data, np.float32(0.0)),
                     requires_grad=self.requires_grad, _parents=(self,))
        keep = (self.data > 0).astype(np.float32)

        def bw(g, grads):
            if self.requires_grad:
                _acc(grads, self, g * keep)
        out._backward = bw
        return out

    # ---- 便利 ----
    def scale(self, c):
        return self * c

    def to_numpy(self):
        if self.device == "cuda":
            src = getattr(self, "_lazy_T", None)
            if src is not None:
                return _cuda().d2h(_cuda().transpose2d(src._gpu))
            return _cuda().d2h(self._gpu)
        return self.data


def _acc(grads, node, g):
    if getattr(node, "device", "cpu") == "cuda":
        rt = _cuda()
        old = grads.get(id(node))
        grads[id(node)] = g if old is None else rt.add(old, g)
        return
    g = np.asarray(g, dtype=np.float32)
    old = grads.get(id(node))
    grads[id(node)] = g if old is None else old + g


def _as_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def _int_index_array(idx):
    """若不是「单个整型数组充当首轴索引」就返回 None。

    只认：`x[arr]` 或 `x[arr, :]`（arr 是一维整型）。其余形态交给原有切片计划。
    负数按 numpy 语义折回，越界在这里就报错 —— 别让它跑到设备上算成错误位置。
    """
    import numpy as _np

    if isinstance(idx, tuple):
        if len(idx) != 2 or not isinstance(idx[1], slice):
            return None
        if not (idx[1].start is None and idx[1].stop is None
                and idx[1].step is None):
            return None
        idx = idx[0]
    if isinstance(idx, slice) or idx is Ellipsis:
        return None
    if isinstance(idx, list):
        if not idx or not all(isinstance(v, (int, _np.integer)) for v in idx):
            return None
        idx = _np.asarray(idx, dtype=_np.int64)
    if isinstance(idx, Tensor):
        if getattr(idx, "device", "cpu") == "cuda":
            idx = _np.asarray(idx.to_numpy())
        else:
            idx = _np.asarray(idx.data)
    if not isinstance(idx, _np.ndarray):
        return None
    if idx.dtype.kind not in "iu":
        return None
    if idx.ndim != 1:
        return None
    return idx.astype(_np.int64, copy=False)


def _has_array_index(idx):
    """判断索引里是否含数组（需要 add.at 的重复累加语义）。"""
    items = idx if isinstance(idx, tuple) else (idx,)
    for it in items:
        if isinstance(it, np.ndarray):
            return True
        if isinstance(it, (list, tuple)):
            return True
    return False


def _contig_strides(shape):
    raw = [0] * len(shape)
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        raw[i] = stride
        stride *= int(shape[i])
    return raw


def _is_int_index(it):
    if isinstance(it, (bool, np.bool_)):
        return False
    return isinstance(it, (int, np.integer))


def _slice_gather_plan(shape, idx):
    """基础索引收成 gather 计划：(out_shape, base, divs, strides)。

    输出线性下标拆成 coord 后，源下标 =
        base + sum (coord[d] // divs[d]) * strides[d]
    切片的 div 恒为 1，strides 已经乘上 step（可以为负）。
    整数下标折进 base，并且从输出形状里丢掉那一维。
    """
    shape = tuple(int(s) for s in shape)
    nd = len(shape)
    if nd > 8:
        raise NotImplementedError("切片最高支持 8 维，收到 %d 维" % nd + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    if not isinstance(idx, tuple):
        idx = (idx,)
    n_ellipsis = 0
    consumed = 0
    for it in idx:
        if it is Ellipsis:
            n_ellipsis += 1
            continue
        if it is None:
            raise NotImplementedError("切片暂不支持 None（newaxis）" + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
        if _is_int_index(it) or isinstance(it, slice):
            consumed += 1
            continue
        raise NotImplementedError("切片暂不支持索引类型 %s" % type(it).__name__ + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    if n_ellipsis > 1:
        raise IndexError("Ellipsis 只能出现一次" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    if n_ellipsis == 0:
        if consumed > nd:
            raise IndexError("索引维数 %d 超过 ndim %d" % (consumed, nd) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        items = list(idx) + [slice(None)] * (nd - consumed)
    else:
        fill = nd - consumed
        if fill < 0:
            raise IndexError("索引维数 %d 超过 ndim %d" % (consumed, nd) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        items = []
        for it in idx:
            if it is Ellipsis:
                items.extend([slice(None)] * fill)
            else:
                items.append(it)
    strides_in = _contig_strides(shape)
    out_shape = []
    divs = []
    strides = []
    base = 0
    axis = 0
    for it in items:
        if axis >= nd:
            raise IndexError("索引维数超过 ndim %d" % nd + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        length = int(shape[axis])
        src_st = int(strides_in[axis])
        if _is_int_index(it):
            ii = int(it)
            if ii < 0:
                ii += length
            if ii < 0 or ii >= length:
                raise IndexError("索引 %d 越界（维 %d 长度 %d）" % (int(it), axis, length) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
            base += ii * src_st
            axis += 1
            continue
        start, stop, step = it.indices(length)
        count = len(range(start, stop, step))
        out_shape.append(count)
        divs.append(1)
        strides.append(int(step) * src_st)
        base += int(start) * src_st
        axis += 1
    if axis != nd:
        raise IndexError("索引没有覆盖全部维度（%d/%d）" % (axis, nd) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
    if len(out_shape) > 8:
        raise NotImplementedError("切片结果最高支持 8 维" + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    return tuple(out_shape), int(base), tuple(divs), tuple(strides)


def _repeat_gather_plan(shape, repeats, dim):
    """repeat_interleave 的 gather 计划。被重复的那一维 div=repeats。"""
    shape = tuple(int(s) for s in shape)
    nd = len(shape)
    if nd == 0:
        raise RuntimeError("标量不能 repeat_interleave，先 reshape 成 1 维" + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
    if nd > 8:
        raise NotImplementedError("repeat_interleave 最高支持 8 维" + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    repeats = int(repeats)
    if repeats < 0:
        raise RuntimeError("repeats 必须 >= 0，收到 %d" % repeats + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    axis = int(dim)
    if axis < 0:
        axis += nd
    if axis < 0 or axis >= nd:
        raise IndexError("dim %s 越界（ndim=%d）" % (dim, nd) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    out = list(shape)
    out[axis] = int(shape[axis]) * repeats
    strides_in = _contig_strides(shape)
    divs = []
    strides = []
    for i, st in enumerate(strides_in):
        if i == axis:
            # 输出该维为 0 时不会发射 kernel，div 不能是 0。
            divs.append(repeats if repeats > 0 else 1)
            strides.append(int(st))
        else:
            divs.append(1)
            strides.append(int(st))
    return tuple(out), 0, tuple(divs), tuple(strides)


def _matmul_grad_numpy(g, a, b):
    """np.matmul 的反向。返回与 a、b 同形状的 (da, db)。"""
    a_shape = a.shape
    b_shape = b.shape
    a_1 = a.ndim == 1
    b_1 = b.ndim == 1
    g = np.asarray(g, dtype=np.float32)
    if a_1:
        a = a.reshape(1, a.shape[0])
    if b_1:
        b = b.reshape(b.shape[0], 1)
    if a_1 and b_1:
        g2 = g.reshape(1, 1)
    elif a_1:
        g2 = np.expand_dims(g, axis=-2)
    elif b_1:
        g2 = np.expand_dims(g, axis=-1)
    else:
        g2 = g
    da = np.matmul(g2, np.swapaxes(b, -1, -2))
    db = np.matmul(np.swapaxes(a, -1, -2), g2)
    if a_1:
        da = np.squeeze(da, axis=-2)
    if b_1:
        db = np.squeeze(db, axis=-1)
    da = _unbroadcast(np.asarray(da, dtype=np.float32), a_shape)
    db = _unbroadcast(np.asarray(db, dtype=np.float32), b_shape)
    return da, db


def _scalar_or_none(v, name):
    if v is None:
        return None
    if isinstance(v, Tensor):
        if v.numel() != 1:
            raise NotImplementedError("%s 目前只支持标量" % name + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
        return float(np.asarray(v.to_numpy()).reshape(-1)[0])
    if isinstance(v, (bool, np.bool_)):
        raise TypeError("%s 不接受布尔值" % name + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    return float(v)


def _as_index_array(target):
    if isinstance(target, Tensor):
        target = target.to_numpy()
    return np.asarray(target, dtype=np.int64)


def _normalize_size(size):
    if len(size) == 1 and isinstance(size[0], (tuple, list, np.ndarray)):
        return tuple(int(x) for x in size[0])
    return tuple(int(x) for x in size)


def _infer_dtype(data):
    if isinstance(data, Tensor):
        return data.dtype
    if isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.floating):
            return float32
        if data.dtype == np.int32:
            return int32
        if np.issubdtype(data.dtype, np.integer):
            return int64
        raise TypeError("tensor 暂不支持 dtype=%s" % data.dtype + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    arr = np.asarray(data)
    if arr.dtype == np.bool_:
        raise TypeError("bool 张量尚未实现" + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    if np.issubdtype(arr.dtype, np.floating):
        return float32
    if np.issubdtype(arr.dtype, np.integer):
        return int64
    return float32


def tensor(data, dtype=None, device=None, requires_grad=False):
    if isinstance(data, Tensor):
        data = data.to_numpy()
    if dtype is None:
        dtype = _infer_dtype(data)
    else:
        dtype = _as_dtype(dtype)
    dev = _device_arg(device)
    if _is_float_dtype(dtype):
        return Tensor(data, requires_grad=requires_grad, device=dev)
    if requires_grad:
        raise RuntimeError("只有浮点张量可以 requires_grad" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    return Tensor(data, requires_grad=False, device=dev, dtype=dtype)


def _constant(shape, value, dt, device, requires_grad):
    dev = _device_arg(device)
    if dt is None or _is_float_dtype(dt):
        data = np.full(shape, value, dtype=np.float32)
        return Tensor(data, requires_grad=requires_grad, device=dev)
    if requires_grad:
        raise RuntimeError("只有浮点张量可以 requires_grad" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    data = np.full(shape, value, dtype=_numpy_dtype(dt))
    return Tensor(data, requires_grad=False, device=dev, dtype=_as_dtype(dt))


def zeros(*size, dtype=None, device=None, requires_grad=False):
    return _constant(_normalize_size(size), 0, dtype, device, requires_grad)


def ones(*size, dtype=None, device=None, requires_grad=False):
    return _constant(_normalize_size(size), 1, dtype, device, requires_grad)


def full(size, fill_value, dtype=None, device=None, requires_grad=False):
    if isinstance(size, (int, np.integer)) and not isinstance(size, bool):
        shape = (int(size),)
    else:
        shape = tuple(int(x) for x in size)
    return _constant(shape, fill_value, dtype, device, requires_grad)


def randn(*size, dtype=None, device=None, requires_grad=False):
    if dtype is not None and not _is_float_dtype(dtype):
        raise TypeError("randn 只支持浮点 dtype" + "。修法：本引擎张量统一 float32；需要低精度请用 rt.fp16_gemm，而不是改 dtype")
    data = np.random.standard_normal(_normalize_size(size)).astype(np.float32)
    return Tensor(data, requires_grad=requires_grad, device=_device_arg(device))


def rand(*size, dtype=None, device=None, requires_grad=False):
    if dtype is not None and not _is_float_dtype(dtype):
        raise TypeError("rand 只支持浮点 dtype" + "。修法：本引擎张量统一 float32；需要低精度请用 rt.fp16_gemm，而不是改 dtype")
    data = np.random.random(_normalize_size(size)).astype(np.float32)
    return Tensor(data, requires_grad=requires_grad, device=_device_arg(device))


def arange(start, end=None, step=1, dtype=None, device=None, requires_grad=False):
    if end is None:
        end = start
        start = 0
    if dtype is None:
        vals = (start, end, step)
        integral = all(
            isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
            for v in vals
        )
        dtype = int64 if integral else float32
    else:
        dtype = _as_dtype(dtype)
    dev = _device_arg(device)
    if _is_float_dtype(dtype):
        data = np.arange(start, end, step, dtype=np.float32)
        return Tensor(data, requires_grad=requires_grad, device=dev)
    if requires_grad:
        raise RuntimeError("只有浮点张量可以 requires_grad" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    data = np.arange(start, end, step, dtype=_numpy_dtype(dtype))
    return Tensor(data, requires_grad=False, device=dev, dtype=dtype)


def manual_seed(seed):
    """种子打进全局 numpy RNG。nn.Linear / Embedding 的初始化也吃这一个 RNG。

    近似：没有 torch.Generator，也没有单独的 CUDA 随机流。
    两次 randn 在同一种子下一致，但数值不会和 PyTorch 逐位相同。
    """
    np.random.seed(int(seed))
    return int(seed)


def from_numpy(ndarray):
    """近似：总是拷贝，不和调用方共享内存。浮点收成 float32。"""
    if not isinstance(ndarray, np.ndarray):
        raise TypeError("from_numpy 需要 numpy.ndarray，收到 %s" % type(ndarray).__name__ + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    if ndarray.dtype == np.bool_:
        raise TypeError("bool 张量尚未实现" + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    if np.issubdtype(ndarray.dtype, np.floating):
        return Tensor(np.asarray(ndarray, dtype=np.float32).copy())
    if ndarray.dtype == np.int32:
        return Tensor(ndarray.copy(), dtype=int32)
    if np.issubdtype(ndarray.dtype, np.integer):
        return Tensor(np.asarray(ndarray, dtype=np.int64).copy(), dtype=int64)
    raise TypeError("from_numpy 暂不支持 dtype=%s" % ndarray.dtype + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")


def _cat_check(ts, dim):
    """拼接轴，以及每个输入的形状。除拼接轴外必须一致。"""
    nd = int(ts[0].ndim)
    shapes = []
    for i, t in enumerate(ts):
        if int(t.ndim) != nd:
            raise ValueError(
                "cat 各张量维度数必须相同，张量 0 是 %d 维 %s，张量 %d 是 %d 维 %s"
                % (nd, tuple(int(s) for s in ts[0].shape),
                   i, int(t.ndim), tuple(int(s) for s in t.shape)) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
        shapes.append(tuple(int(s) for s in t.shape))
    if nd == 0:
        raise ValueError("cat 不支持 0 维张量（形状 %s）" % (shapes[0],) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
    axis = int(dim)
    if axis < 0:
        axis += nd
    if axis < 0 or axis >= nd:
        raise ValueError("cat 轴 %s 越界（ndim=%d）" % (dim, nd) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    ref = shapes[0]
    for i, sh in enumerate(shapes[1:], 1):
        for d, (a, b) in enumerate(zip(ref, sh)):
            if d != axis and a != b:
                raise ValueError(
                    "cat 除拼接轴 %d 外形状必须一致：张量 0 %s，张量 %d %s"
                    % (axis, ref, i, sh) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
    return axis, shapes


def _cat_cuda(ts, dim):
    """沿任意轴拼接。前向按槽拷贝，反向把上游梯度按同样的槽切回去。"""
    axis, shapes = _cat_check(ts, dim)
    rt = _cuda()
    sizes = [int(sh[axis]) for sh in shapes]
    out_shape = list(shapes[0])
    total = 0
    for sz in sizes:
        total += int(sz)
    out_shape[axis] = int(total)
    out_shape = tuple(out_shape)
    srcs = []
    for t, sh in zip(ts, shapes):
        src = t._cuda_logical_storage()
        got = tuple(int(s) for s in src.shape)
        if got != sh:
            raise RuntimeError("cat 设备布局 %s 与逻辑形状 %s 不一致" % (got, sh) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
        if src.dtype != np.float32:
            raise TypeError("CUDA cat 目前只支持 float32，收到 %s" % (src.dtype,) + "。修法：本引擎张量统一 float32；需要低精度请用 rt.fp16_gemm，而不是改 dtype")
        srcs.append(src)
    dst = rt.empty(out_shape, dtype=np.float32)
    # 逻辑存储若是临时转置，必须活过默认流上的拷贝核。
    dst.keepalive = tuple(srcs)
    off = 0
    for src, sz in zip(srcs, sizes):
        if sz:
            rt.cat_place(src, dst, axis, off)
        off += int(sz)
    rg = any(t.requires_grad for t in ts)
    out = Tensor._from_gpu(dst, requires_grad=rg, _parents=tuple(ts))
    saved_sizes = tuple(int(s) for s in sizes)
    saved_shapes = tuple(shapes)

    def bw(g, grads, _ts=ts, _sizes=saved_sizes, _axis=axis, _shapes=saved_shapes):
        off_b = 0
        for t, sz, sh in zip(_ts, _sizes, _shapes):
            if t.requires_grad:
                part = rt.cat_extract(g, _axis, off_b, sz, sh)
                _acc(grads, t, part)
            off_b += int(sz)
    out._backward = bw
    return out


def cat(tensors, dim=0):
    ts = [_as_tensor(t) for t in tensors]
    if not ts:
        raise ValueError("cat 至少需要一个张量" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    if any(getattr(t, "device", "cpu") == "cuda" for t in ts):
        if any(getattr(t, "device", "cpu") != "cuda" for t in ts):
            raise RuntimeError(
                "cat 不能混用 CPU 和 CUDA 张量，收到 %s"
                % ", ".join(str(getattr(t, "device", "cpu")) for t in ts) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        return _cat_cuda(ts, dim)
    axis = int(dim)
    if axis < 0:
        axis += ts[0].ndim
    data = np.concatenate([t.data for t in ts], axis=axis)
    floating = all(np.issubdtype(t.data.dtype, np.floating) for t in ts)
    if not floating:
        return Tensor(data, dtype=ts[0].dtype, requires_grad=False)
    rg = any(t.requires_grad for t in ts)
    out = Tensor(data, requires_grad=rg, _parents=tuple(ts))
    sizes = [int(t.shape[axis]) for t in ts]

    def bw(g, grads):
        parts = np.split(np.asarray(g, dtype=np.float32), np.cumsum(sizes)[:-1], axis=axis)
        for t, part in zip(ts, parts):
            if t.requires_grad:
                _acc(grads, t, part)
    out._backward = bw
    return out


def stack(tensors, dim=0):
    ts = [_as_tensor(t) for t in tensors]
    if not ts:
        raise ValueError("stack 至少需要一个张量" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    if any(getattr(t, "device", "cpu") == "cuda" for t in ts):
        raise NotImplementedError("CUDA stack 尚未实现" + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    axis = int(dim)
    data = np.stack([t.data for t in ts], axis=axis)
    floating = all(np.issubdtype(t.data.dtype, np.floating) for t in ts)
    if not floating:
        return Tensor(data, dtype=ts[0].dtype, requires_grad=False)
    rg = any(t.requires_grad for t in ts)
    out = Tensor(data, requires_grad=rg, _parents=tuple(ts))

    def bw(g, grads):
        parts = np.split(np.asarray(g, dtype=np.float32), len(ts), axis=axis)
        for t, part in zip(ts, parts):
            if t.requires_grad:
                _acc(grads, t, np.squeeze(part, axis=axis))
    out._backward = bw
    return out


def matmul(input, other):
    return _as_tensor(input) @ _as_tensor(other)


def bmm(input, mat2):
    a = _as_tensor(input)
    b = _as_tensor(mat2)
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("bmm 需要两个 3D 张量，收到 %s x %s" % (a.shape, b.shape) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
    return a @ b


def exp(input):
    return _as_tensor(input).exp()


def log(input):
    return _as_tensor(input).log()


def sqrt(input):
    return _as_tensor(input).sqrt()


def _api_abs(input):
    return _as_tensor(input).abs()


def _api_clamp(input, min=None, max=None):
    return _as_tensor(input).clamp(min=min, max=max)


def _reduce_values_grad(data, grad, axis, keepdim, pick):
    idx = pick(data, axis=axis)
    full = np.zeros(data.shape, dtype=np.float32)
    g = np.asarray(grad, dtype=np.float32)
    idx_exp = np.expand_dims(idx, axis)
    if not keepdim:
        g = np.expand_dims(g, axis)
    np.put_along_axis(full, idx_exp, g, axis=axis)
    return full


def _reduce_pair(t, dim, keepdim, value_fn, index_fn):
    if getattr(t, "device", "cpu") == "cuda":
        raise NotImplementedError("CUDA max/min 尚未实现" + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    floating = np.issubdtype(t.data.dtype, np.floating)
    if dim is None:
        flat = t.data.reshape(-1)
        flat_i = int(index_fn(flat))
        raw = flat[flat_i]
        if not floating:
            return Tensor(np.array(raw), dtype=t.dtype, requires_grad=False)
        out = Tensor(np.array(raw, dtype=np.float32), requires_grad=t.requires_grad, _parents=(t,))

        def bw(g, grads):
            if not t.requires_grad:
                return
            full = np.zeros(t.shape, dtype=np.float32)
            full.reshape(-1)[flat_i] = np.asarray(g, dtype=np.float32).reshape(-1)[0]
            _acc(grads, t, full)
        out._backward = bw
        return out
    axis = int(dim)
    if axis < 0:
        axis += t.ndim
    values = value_fn(t.data, axis=axis, keepdims=keepdim)
    indices = index_fn(t.data, axis=axis)
    if keepdim:
        indices = np.expand_dims(indices, axis)
    it = Tensor(indices, dtype=int64)
    if not floating:
        return Tensor(values, dtype=t.dtype, requires_grad=False), it
    vt = Tensor(values, requires_grad=t.requires_grad, _parents=(t,))

    def bw(g, grads):
        if t.requires_grad:
            _acc(grads, t, _reduce_values_grad(t.data, g, axis, keepdim, index_fn))
    vt._backward = bw
    return vt, it


def _api_max(input, dim=None, keepdim=False):
    return _reduce_pair(_as_tensor(input), dim, keepdim, np.max, np.argmax)


def _api_min(input, dim=None, keepdim=False):
    return _reduce_pair(_as_tensor(input), dim, keepdim, np.min, np.argmin)


def topk(input, k, dim=-1, largest=True, sorted=True):
    """近似：并列时名次跟 numpy.argpartition，不保证和 PyTorch 的稳定排序一致。"""
    t = _as_tensor(input)
    if getattr(t, "device", "cpu") == "cuda":
        raise NotImplementedError("CUDA topk 尚未实现" + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    k = int(k)
    axis = int(dim)
    if axis < 0:
        axis += t.ndim
    n = int(t.shape[axis])
    if k < 0 or k > n:
        raise ValueError("topk 的 k=%d 超出维度长度 %d" % (k, n) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
    data = t.data
    if k == 0:
        shape = list(data.shape)
        shape[axis] = 0
        idx = np.zeros(shape, dtype=np.int64)
        values = np.zeros(shape, dtype=np.float32 if np.issubdtype(data.dtype, np.floating) else data.dtype)
    else:
        if largest:
            part = np.argpartition(data, n - k, axis=axis)
            idx = np.take(part, np.arange(n - k, n), axis=axis)
        else:
            part = np.argpartition(data, k - 1, axis=axis)
            idx = np.take(part, np.arange(0, k), axis=axis)
        values = np.take_along_axis(data, idx, axis=axis)
        if sorted:
            key = -values if largest else values
            order = np.argsort(key, axis=axis)
            idx = np.take_along_axis(idx, order, axis=axis)
            values = np.take_along_axis(values, order, axis=axis)
    floating = np.issubdtype(data.dtype, np.floating)
    saved_idx = np.array(idx, dtype=np.int64, copy=True)
    it = Tensor(saved_idx, dtype=int64)
    if not floating:
        return Tensor(values, dtype=t.dtype, requires_grad=False), it
    vt = Tensor(np.asarray(values, dtype=np.float32), requires_grad=t.requires_grad, _parents=(t,))

    def bw(g, grads):
        if not t.requires_grad:
            return
        full = np.zeros(data.shape, dtype=np.float32)
        if k:
            np.put_along_axis(full, saved_idx, np.asarray(g, dtype=np.float32), axis=axis)
        _acc(grads, t, full)
    vt._backward = bw
    return vt, it


def where(condition, x=None, y=None):
    if isinstance(condition, Tensor):
        cond = condition.to_numpy()
    else:
        cond = np.asarray(condition)
    if x is None and y is None:
        return tuple(Tensor(np.asarray(a, dtype=np.int64), dtype=int64) for a in np.nonzero(cond))
    if x is None or y is None:
        raise TypeError("where 需要只给 condition，或者同时给 x 和 y" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    xt = _as_tensor(x)
    yt = _as_tensor(y)
    if getattr(xt, "device", "cpu") == "cuda" or getattr(yt, "device", "cpu") == "cuda":
        raise NotImplementedError("CUDA where 尚未实现" + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    mask = np.asarray(cond, dtype=bool)
    floating = np.issubdtype(xt.data.dtype, np.floating) or np.issubdtype(yt.data.dtype, np.floating)
    if not floating:
        return Tensor(np.where(mask, xt.data, yt.data), dtype=xt.dtype, requires_grad=False)
    data = np.where(mask, xt.data, yt.data).astype(np.float32)
    rg = xt.requires_grad or yt.requires_grad
    out = Tensor(data, requires_grad=rg, _parents=(xt, yt))

    def bw(g, grads):
        gg = np.asarray(g, dtype=np.float32)
        if xt.requires_grad:
            _acc(grads, xt, _unbroadcast(np.where(mask, gg, np.float32(0)), xt.shape))
        if yt.requires_grad:
            _acc(grads, yt, _unbroadcast(np.where(mask, np.float32(0), gg), yt.shape))
    out._backward = bw
    return out


def _split_einsum(equation, operands):
    eq = equation.replace(" ", "")
    if "..." in eq:
        raise NotImplementedError("einsum 暂不支持省略号" + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    if "->" in eq:
        lhs, rhs = eq.split("->")
    else:
        lhs = eq
        counts = {}
        for ch in lhs:
            if ch.isalpha():
                counts[ch] = counts.get(ch, 0) + 1
        rhs = "".join(ch for ch in sorted(counts) if counts[ch] == 1)
    labels = lhs.split(",") if lhs else []
    if len(labels) != len(operands):
        raise ValueError("einsum 操作数个数 %d 和方程 %r 不一致" % (len(operands), equation) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    for lab in labels + [rhs]:
        letters = [c for c in lab if c.isalpha()]
        if len(letters) != len(set(letters)):
            raise NotImplementedError("einsum 暂不支持重复字母: %r" % equation + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
        if any(not c.isalpha() for c in lab):
            raise NotImplementedError("einsum 暂不支持字母以外的下标: %r" % equation + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    return labels, rhs


def _einsum_vjp(labels, rhs, operands, grad, index):
    target = labels[index]
    others_l = [labels[j] for j in range(len(operands)) if j != index]
    others_a = [operands[j].data for j in range(len(operands)) if j != index]
    provided = set(rhs)
    for lab in others_l:
        provided.update(lab)
    missing = [c for c in target if c not in provided]
    known = "".join(c for c in target if c in provided)
    g = np.asarray(grad, dtype=np.float32)
    if not missing:
        if rhs == "" and not others_l:
            return np.broadcast_to(g.reshape(()), operands[index].shape).astype(np.float32).copy()
        if rhs == "":
            gi = np.einsum(",".join(others_l) + "->" + target, *others_a) * g.reshape(())
        else:
            gi = np.einsum(",".join([rhs] + others_l) + "->" + target, g, *others_a)
        return np.asarray(gi, dtype=np.float32).reshape(operands[index].shape)
    if known == "" and not others_l and rhs == "":
        return np.broadcast_to(g.reshape(()), operands[index].shape).astype(np.float32).copy()
    if rhs == "":
        if known == "":
            partial = g.reshape(())
        else:
            partial = np.asarray(np.einsum(",".join(others_l) + "->" + known, *others_a), dtype=np.float32)
            partial = partial * g.reshape(())
    elif known == "":
        partial = np.asarray(np.einsum(",".join([rhs] + others_l) + "->", g, *others_a), dtype=np.float32)
    else:
        partial = np.asarray(
            np.einsum(",".join([rhs] + others_l) + "->" + known, g, *others_a), dtype=np.float32)
    sizes = {c: n for c, n in zip(target, operands[index].shape)}
    ones_labels = "".join(missing)
    ones = np.ones(tuple(sizes[c] for c in ones_labels), dtype=np.float32)
    if known == "":
        gi = np.einsum(ones_labels + "->" + target, ones) * np.asarray(partial, dtype=np.float32).reshape(())
    else:
        gi = np.einsum(known + "," + ones_labels + "->" + target, np.asarray(partial, dtype=np.float32), ones)
    return np.asarray(gi, dtype=np.float32).reshape(operands[index].shape)


def einsum(equation, *operands):
    """近似：不支持省略号和重复字母。浮点结果收成 float32。"""
    ops = [_as_tensor(o) for o in operands]
    if any(getattr(o, "device", "cpu") == "cuda" for o in ops):
        raise NotImplementedError("CUDA einsum 尚未实现" + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    if not ops or not all(np.issubdtype(o.data.dtype, np.floating) for o in ops):
        raise NotImplementedError("einsum 目前只支持浮点张量" + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    labels, rhs = _split_einsum(equation, ops)
    for o, lab in zip(ops, labels):
        if o.ndim != len(lab):
            raise ValueError("einsum %r 的字母数和形状 %s 不一致" % (lab, o.shape) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
    out_data = np.einsum(",".join(labels) + "->" + rhs, *[o.data for o in ops])
    rg = any(o.requires_grad for o in ops)
    out = Tensor(np.asarray(out_data, dtype=np.float32), requires_grad=rg, _parents=tuple(ops))

    def bw(g, grads):
        for i, o in enumerate(ops):
            if o.requires_grad:
                _acc(grads, o, _einsum_vjp(labels, rhs, ops, g, i))
    out._backward = bw
    return out


def _pack(obj):
    if isinstance(obj, Tensor):
        return {
            "__tgai__": "tensor",
            "data": np.array(obj.to_numpy(), copy=True),
            "requires_grad": bool(obj.requires_grad),
        }
    if isinstance(obj, Module):
        return {"__tgai__": "state_dict", "data": obj.state_dict()}
    if isinstance(obj, dict):
        return {"__tgai__": "dict", "data": {str(k): _pack(v) for k, v in obj.items()}}
    if isinstance(obj, tuple):
        return {"__tgai__": "tuple", "data": [_pack(v) for v in obj]}
    if isinstance(obj, list):
        return {"__tgai__": "list", "data": [_pack(v) for v in obj]}
    if isinstance(obj, np.ndarray):
        return {"__tgai__": "ndarray", "data": np.array(obj, copy=True)}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise TypeError("save 不支持 %s。这不是 torch.save 的 zip 格式。" % type(obj).__name__ + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")


def _unpack(obj):
    if not isinstance(obj, dict) or "__tgai__" not in obj:
        return obj
    kind = obj["__tgai__"]
    if kind == "tensor":
        data = obj["data"]
        if np.issubdtype(data.dtype, np.floating):
            return Tensor(data, requires_grad=bool(obj.get("requires_grad", False)))
        return Tensor(data, dtype=_wrap_dtype(data.dtype), requires_grad=False)
    if kind == "ndarray":
        return obj["data"]
    if kind == "dict":
        return {k: _unpack(v) for k, v in obj["data"].items()}
    if kind == "list":
        return [_unpack(v) for v in obj["data"]]
    if kind == "tuple":
        return tuple(_unpack(v) for v in obj["data"])
    if kind == "state_dict":
        return {k: np.array(v, copy=True) for k, v in obj["data"].items()}
    raise TypeError("未知存档类型 %s" % kind + "。修法：本引擎张量统一 float32；需要低精度请用 rt.fp16_gemm，而不是改 dtype")


def _move_loaded(obj, dev):
    if isinstance(obj, Tensor) and obj.data is not None and np.issubdtype(obj.data.dtype, np.floating):
        return Tensor(obj.to_numpy(), requires_grad=obj.requires_grad, device=dev)
    if isinstance(obj, dict):
        return {k: _move_loaded(v, dev) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_loaded(v, dev) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_loaded(v, dev) for v in obj)
    return obj


def save(obj, f):
    """近似：pickle 一份带标记的 numpy，不是 torch.save 的 zip，两边不能互读。"""
    import pickle
    payload = {"__tgai_save__": 1, "obj": _pack(obj)}
    if hasattr(f, "write"):
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return
    with open(f, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load(f, map_location=None):
    import pickle
    if hasattr(f, "read"):
        payload = pickle.load(f)
    else:
        with open(f, "rb") as fh:
            payload = pickle.load(fh)
    if not isinstance(payload, dict) or payload.get("__tgai_save__") != 1:
        raise TypeError("不是 tgai_go.train.save 写出的文件（不能读 torch.save 的 zip）" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    obj = _unpack(payload["obj"])
    dest = None
    if isinstance(map_location, Device):
        dest = map_location.type
    elif isinstance(map_location, str):
        dest = map_location.split(":", 1)[0]
    if dest not in (None, "cpu"):
        obj = _move_loaded(obj, dest)
    return obj


# ─────────────────────────── functional ───────────────────────────
class _Functional:
    @staticmethod
    def tanh(x: Tensor) -> Tensor:
        """组合式（2*sigmoid(2x)-1）⇒ CPU/CUDA 都能用。"""
        from . import nn_ops as _O
        return _O.tanh(x)

    @staticmethod
    def gelu(x: Tensor, approximate: str = "tanh") -> Tensor:
        from . import nn_ops as _O
        return _O.gelu(x, approximate=approximate)

    @staticmethod
    def relu(x: Tensor) -> Tensor:
        """max(x,0)。CUDA 上走 Tensor.relu 会明确报错（当前是 numpy 实现）。"""
        if isinstance(x, Tensor):
            return x.relu()
        return Tensor(np.maximum(np.asarray(x, np.float32), 0.0))

    @staticmethod
    def silu(x: Tensor) -> Tensor:
        if getattr(x, "device", "cpu") == "cuda":
            return _silu_cuda(x)
        y, d = _silu32_pair(x.data)
        out = Tensor(y, requires_grad=x.requires_grad, _parents=(x,))

        def bw(g, grads):
            if x.requires_grad:
                _acc(grads, x, np.float32(np.asarray(g, np.float32) * d))
        out._backward = bw
        return out

    @staticmethod
    def softmax(x: Tensor, axis=-1, dim=None) -> Tensor:
        if dim is not None:
            axis = dim
        if getattr(x, "device", "cpu") == "cuda":
            return _softmax_cuda(x, axis)
        m = x.data.max(axis=axis, keepdims=True)
        z = x - Tensor(m)                     # 常数，不参与求导
        e = z.exp()
        return e * (e.sum(axis=axis, keepdims=True) ** -1.0)

    @staticmethod
    def log_softmax(x: Tensor, axis=-1, dim=None) -> Tensor:
        if dim is not None:
            axis = dim
        if getattr(x, "device", "cpu") == "cuda":
            return _log_softmax_cuda(x, axis)
        m = x.data.max(axis=axis, keepdims=True)
        z = x - Tensor(m)
        e = z.exp()
        return z - e.sum(axis=axis, keepdims=True).log()

    @staticmethod
    def scaled_dot_product_attention(q, k, v, causal=False, is_causal=None,
                                   scale=None, lowmem=False, attn_mask=None):
        """与 torch 的 `F.scaled_dot_product_attention` 兼容的入口。

        两种输入都可：
          * `[B*H, S, Dh]` —— 内部展平形式（原有用法，保持不变）
          * `[B, H, S, Dh]` —— **torch 的写法**，自动展平并在返回时还原

        其它参数与 `_sdpa_impl` 完全一致。`attn_mask` 为加性掩码（0 / -inf）。
        """
        rank = len(tuple(q.shape))
        if rank != 4:
            return _Functional._sdpa_impl(q, k, v, causal=causal,
                                          is_causal=is_causal,
                                          scale=scale, lowmem=lowmem,
                                          attn_mask=attn_mask)
        B, H, S, Dh = (int(q.shape[0]), int(q.shape[1]),
                        int(q.shape[2]), int(q.shape[3]))
        qf = q.reshape(B * H, S, Dh)
        kf = k.reshape(int(k.shape[0]) * int(k.shape[1]), int(k.shape[2]),
                          int(k.shape[3]))
        vf = v.reshape(int(v.shape[0]) * int(v.shape[1]), int(v.shape[2]),
                          int(v.shape[3]))
        out = _Functional._sdpa_impl(qf, kf, vf, causal=causal,
                                     is_causal=is_causal,
                                     scale=scale, lowmem=lowmem,
                                     attn_mask=attn_mask)
        return out.reshape(B, H, S, Dh)

    @staticmethod
    def _sdpa_impl(q: Tensor, k: Tensor, v: Tensor, causal=False, is_causal=None,
                                     scale=None, lowmem=False, attn_mask=None):
        """q,k,v: [B*H, S, Dh]（已展平 head）。causal=True 时做下三角掩码。

        lowmem=True 且设备是 CUDA、dh==64、causal 时，走不物化 P 的内核。
        其它情况（含 dh!=64、非 causal、CPU）回退到下面的物化实现。默认 lowmem=False。

        `attn_mask`：加性掩码；**给出时强制走物化路径**（lowmem 内核不接受掩码）。
        """
        if is_causal is not None:
            causal = bool(is_causal)
        dh = int(q.shape[-1])
        if scale is None:
            scale = 1.0 / np.sqrt(dh)
        else:
            scale = float(scale)
        if attn_mask is None and lowmem and _sdpa_lowmem_ok(q, k, v, causal, dh):
            return _sdpa_lowmem(q, k, v, scale)
        return _sdpa_materialized(q, k, v, causal, scale, attn_mask=attn_mask)

    @staticmethod
    def sdpa_lowmem(q: Tensor, k: Tensor, v: Tensor, causal=True, scale=None):
        """不物化 P 的 SDPA。dh!=64、非 causal 或非 CUDA 时回退到物化实现。"""
        return _Functional.scaled_dot_product_attention(
            q, k, v, causal=causal, scale=scale, lowmem=True)

    @staticmethod
    def sdpa_lowmem_ready():
        """提前编译 tgai_attn_nop.cu。计时前调用，避免把 NVRTC 算进 s/step。"""
        return _cuda().ensure_attn_nop()

    @staticmethod
    def linear_cross_entropy(x: Tensor, weight: Tensor, target, chunk: int = 1024):
        """分块算 logits + cross_entropy，避免把 [N, V] 全量物化。

        V 很大时（32779 / 96000）全量 logits 一步就 1 GiB 以上；按 token 分块后
        峰值只与 chunk 有关，总 FLOPs 不变。CE 逐行独立，所以按 (块行数/总行数) 加权累加即等价。
        反向由 autograd 串起来，每块的 dlogits 用完即释放、池可复用。
        """
        # 分块只为省显存（VRAM），CPU 侧没有这个动机，直接走普通路径，
        # 避免 0 维标量相乘在 CPU 广播路径上的边界问题（实测会踩 numpy axis 越界）。
        if getattr(x, "device", "cpu") != "cuda":
            return _Functional.cross_entropy(x @ weight.T, target)
        n = int(x.shape[0])
        if int(target.shape[0]) != n:
            raise ValueError("linear_cross_entropy 目标长度 %d != N=%d"
                             % (int(target.shape[0]), n) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
        c = int(chunk) if chunk else n
        if c <= 0:
            c = n
        total = None
        for s0 in range(0, n, c):
            e0 = s0 + c
            if e0 > n:          # 注意：模块级 min 已被张量 API 占用，这里不能裸写 min
                e0 = n
            lg = x[s0:e0] @ weight.T                  # [块行数, V]
            part = _Functional.cross_entropy(lg, target[s0:e0])   # 块内均值
            term = part * (float(e0 - s0) / float(n))
            total = term if total is None else total + term
        return total

    @staticmethod
    def cross_entropy(logits: Tensor, target, ignore_index=-100):

        if isinstance(target, Tensor):
            target = target.to_numpy()
        """logits [N,V]，target [N]（int）。返回标量 loss（mean）。"""
        if getattr(logits, "device", "cpu") == "cuda":
            return _cross_entropy_cuda(logits, target, ignore_index)
        return _Functional._cross_entropy_cpu(logits, target, ignore_index)

    @staticmethod
    def _cross_entropy_cpu(logits, target, ignore_index=-100):
        n, v = logits.shape
        tgt = _as_index_array(target).reshape(-1)
        if np.any(tgt == ignore_index):
            lp = _Functional.log_softmax(logits, axis=-1)
            mask = tgt != ignore_index
            # ★ 先把 ignore 位置换成合法索引 0，否则 lp[arange, -100] 会越界
            safe = np.where(mask, tgt, 0)
            picked = lp[np.arange(n), safe]
            return (picked * Tensor(mask.astype(np.float32))).sum() * (
                -1.0 / _builtin_max(1, mask.sum()))
        return _cross_entropy_fused(logits, tgt)


functional = _Functional()


# ─────────────────── 梯度检查点（activation recompute） ───────────────────

# ---- fp16 权重影子（按"代"缓存，step() 后失效）----
_FP16_HITS = [0]
_FP16_SHADOW_GEN = [0]
_FP16_SHADOWS = {}


def fp16_shadow_gen():
    return _FP16_SHADOW_GEN[0]


def fp16_shadows_invalidate():
    """优化器 step() 之后调用：所有权重影子作废。"""
    _FP16_SHADOW_GEN[0] += 1


_FP16_SCRATCH = {}


def _fp16_scratch(src_buf):
    """按元素数复用一块 fp16 scratch（避免每次 GEMM 前都分配）。

    同一条流上 kernel 有序，所以下一行的 cast 一定在上一个 GEMM 之后执行 —— 复用安全。
    """
    n = int(src_buf.n)
    buf = _FP16_SCRATCH.get(n)
    if buf is None:
        rt = _cuda()
        buf = rt.empty((n,), dtype=__import__("numpy").float16)
        if len(_FP16_SCRATCH) > 64:
            _FP16_SCRATCH.clear()
        _FP16_SCRATCH[n] = buf
    return buf


def _fp16_shadow(buf):
    """取/建权重的 fp16 影子。键用 id(buf)+元素数，代不匹配就重建。"""
    key = (id(buf), int(buf.n))
    gen = _FP16_SHADOW_GEN[0]
    ent = _FP16_SHADOWS.get(key)
    if ent is not None and ent[0] == gen:
        return ent[1]
    rt = _cuda()
    sh = rt.cast_f32_to_f16(buf)
    if len(_FP16_SHADOWS) > 512:            # 上限，防止换模型时无限增长
        _FP16_SHADOWS.clear()
    _FP16_SHADOWS[key] = (gen, sh)
    return sh


def checkpoint(fn, *inputs):
    """用"重算"换显存：`out = checkpoint(fn, x)` 的数值与 `fn(x)` 完全一致，
    但前向不保留中间激活，反向时重跑一次 fn。

    典型用法：把 Transformer 的一层或一段包起来

        def block(h):
            return layer(h)
        h = T.checkpoint(block, h)

    限制（写清楚，避免误用）：
      * `fn` 必须返回**单个 Tensor**（多输出请自己包成 tuple 再拆，见下）；
      * `fn` 必须**可重复执行且确定**（含随机性的算子会让重算结果不一致 ——
        我们的算子里目前没有 dropout，但你自己加的插件若有随机性就不能用）；
      * 只有"需要梯度"的输入才会被追踪，纯 numpy 输入原样透传。
    """
    ts = [x for x in inputs if isinstance(x, Tensor)]
    # ★ 不要按 device 过滤：CUDA 张量也必须走"不建图 + 反向重算"，
    #   否则这个函数在 GPU 上等于 `fn(*inputs)`（静默不省显存）。
    need = [x for x in ts if x.requires_grad]
    if not need:
        return fn(*inputs)              # 没有需要梯度的输入：没必要折腾

    prev = _GRAD_ENABLED[0]
    _GRAD_ENABLED[0] = False
    try:
        with no_grad():
            out = fn(*inputs)
    finally:
        _GRAD_ENABLED[0] = prev

    if not isinstance(out, Tensor):
        raise TypeError(
            "checkpoint: fn 必须返回单个 Tensor，实际 %s。"
            "修法：多输出请包成 lambda，返回一个 Tensor（例如把 tuple 拼起来），"
            "或对每个输出分别调用 checkpoint" % type(out).__name__)

    # 关键：不连 _parents —— 中间激活因此可以在前向结束后被释放。
    #
    # ★ 2026-11-12 改（来自 P100 用户实测：同形状 PyTorch 能跑、ORV 开检查点仍 OOM）：
    #   这里原先**每次都 clone 一份输出**。代价是每段检查点多占一份段输出：
    #   20 层 / d=1280 / B=8 / S=2048 时每层 84 MB ⇒ **白白多占 1.7 GB**，
    #   正好是 16GB 卡上"torch 能跑、我们 OOM"的量级差。
    #
    #   PyTorch 的 `checkpoint` 是直接**返回输出张量本身**，不复制。我们也改成共享同一块
    #   设备缓冲：`out` 在 `no_grad` 下算出来，没有 `_backward`，把它原地升级成图节点即可。
    #   持有 `res` 就等于持有那块缓冲（`GpuBuf` 的 `_token` 保证池子不会回收），
    #   所以不存在"前向缓冲被复用后再读"的风险。
    #
    #   仍然保留"输出必须与输入不同"这一前提：若 fn 直接返回了某个输入，
    #   下面会显式报错，而不是把输入当成检查点输出（那会让重算的梯度加错地方）。
    aliased = any(getattr(x, "_gpu", None) is not None
                  and out._gpu is not None
                  and int(x._gpu.ptr) == int(out._gpu.ptr) for x in ts)
    if aliased:
        raise RuntimeError(
            "checkpoint: fn 直接返回了某个输入张量（ptr 相同）"
            "。修法：让 fn 返回新张量（例如 return x + delta 而不是 return x），"
            "否则重算时梯度会累加到错误的位置")
    res = out
    res.requires_grad = True
    # ★ 2026-11-12 第二次修（对齐 PyTorch `CheckpointFunction`，见 torch/utils/checkpoint.py）
    #
    #   上一版把 `res._parents` 清成 `()` 来切断中间激活 —— 这个方向错了，代价极大：
    #   外层图因此**在段与段之间断开**，反向时只能靠 `bw` 里的 `re.backward(g)` 递归往上一层钻，
    #   而上一层的输出 `_backward` 正是它自己的 `bw` ⇒ **一次调用把整条链全部重算**，
    #   所有段的激活同时驻留。实测（L=6/S=2048/B=2）：前向结束 0.133 GiB，
    #   但第一段反向一开始就是 8.474 GiB ≈ 6 层激活之和，反向峰值 10.8 GiB。
    #
    #   PyTorch 的做法是两件事一起做：
    #     ① **外层图保持连通**（输出节点仍指向真实输入），中间激活靠"没有父节点链"而释放；
    #     ② 反向重算时用 `detach_variable` 把输入变成**叶子**，嵌套反传就停在它们身上，
    #        再把 `detached.grad` 作为本节点的 backward **返回值**交还外层引擎。
    #
    #   对应到这里：parents 只指向"需要梯度的输入"，不再清空；`bw` 里用 detached 输入重算。
    res._parents = tuple(x for x in ts if x.requires_grad)

    def bw(g, grads):
        # 重算用 detached 输入：它们 `_backward is None`（叶子），
        # 嵌套反传走到这里就停，不会递归进上一段。
        ds = []
        fargs = []
        for a in inputs:
            if isinstance(a, Tensor) and a.requires_grad:
                if a.device == "cuda":
                    d = Tensor._from_gpu(a._gpu, requires_grad=True)
                else:
                    d = Tensor(a.data, requires_grad=True)
                d._parents = ()
                d._backward = None
                d.grad = None
                ds.append((a, d))
                fargs.append(d)
            else:
                fargs.append(a)

        prev2 = _GRAD_ENABLED[0]
        _GRAD_ENABLED[0] = True
        try:
            re = fn(*fargs)
        finally:
            _GRAD_ENABLED[0] = prev2
        re.backward(g)

        # 把 detached 输入的梯度交还外层引擎（= PyTorch 的 `return (None, None) + grads`）。
        for orig, d in ds:
            gg = d.grad
            if gg is None:
                continue
            if orig.device == "cuda":
                _acc(grads, orig, gg._gpu if isinstance(gg, Tensor) else gg)
            else:
                _acc(grads, orig,
                     np.asarray(gg.data if isinstance(gg, Tensor) else gg,
                                dtype=np.float32))

    res._backward = bw
    return res


def checkpoint_sequential(functions, x, segments=None):
    """把一串函数（层）分成 `segments` 段，每段做一次 checkpoint（PyTorch 同签名）。

    `segments` 是**显存/速度的旋钮**：
      * `segments = len(functions)`：每层一个检查点 —— 存得最少、重算最频繁；
      * `segments = 1`：整段一个检查点 —— 只存段边界，重算一次跑完这段；
      * `None`（默认）等价于 1，与 PyTorch 的默认行为一致。
    数值与逐个调用完全相同（测试逐位对比），区别只在峰值显存与重算次数。
    """
    fns = list(functions)
    if not fns:
        return x
    n = len(fns)
    # ★ 本模块把内建 min/max 覆盖成了**张量算子**，所以这里不能用 min()/max()：
    #   `min(int, int)` 会走到张量算子上去（实测炸在这个坑里）。用比较硬写。
    seg = 1 if segments is None else int(segments)
    if seg < 1:
        seg = 1
    elif seg > n:
        seg = n
    per = (n + seg - 1) // seg                 # 每段几层（向上取整）
    for i in range(0, n, per):
        group = fns[i:i + per]

        def run(h, group=group):
            for f in group:
                h = f(h)
            return h

        x = checkpoint(run, x)
    return x


def _attn_scores(q, k, scale):
    """scores = (q @ k^T) * scale，[BH,S,DH] → [BH,S,S]。

    用 cuBLAS 的 op=T 直接算转置乘：省掉 k 的一次真拷贝（每层每步一次，全步 64 次）。
    反向也不拷贝：dq = dS @ k（NN），dk = dS^T @ q（TN）。
    """
    rt = _cuda()
    bh, s, dh = (int(x) for x in q.shape)
    buf = rt.sgemm_nt_strided_batched(q._gpu, k._gpu, bh, s, s, dh)
    if scale != 1.0:
        buf = rt.scale(buf, float(scale))
    need = bool(q.requires_grad or k.requires_grad)
    out = Tensor._from_gpu(buf, requires_grad=need, _parents=(q, k))

    def bw(g, grads):
        dq = rt.sgemm_strided_batched(g, k._gpu, bh, s, s, dh)   # [S,S] @ [S,DH] -> [S,DH]
        dk = rt.sgemm_tn_strided_batched(g, q._gpu, bh, s, dh, s)
        if scale != 1.0:
            dq = rt.scale(dq, float(scale))
            dk = rt.scale(dk, float(scale))
        if q.requires_grad:
            _acc(grads, q, dq)
        if k.requires_grad:
            _acc(grads, k, dk)

    if need:
        out._backward = bw
    return out


# 因果掩码缓存：S=512 时一个 mask 是 59 MB。
# ★ 原来每层每步都 `Tensor(mask)` 重建一次 —— 在 cuda 默认设备下那是**每层一次 59 MB H2D**，
#   24 层就是 1.4 GB/前向（实测：forward 0.419 s 里很大一块是这个）。掩码是常量，缓存即可。
_CAUSAL_MASK_CACHE = {}


def _causal_mask(S, device):
    """返回缓存的上三角 -1e30 掩码（常量，不需要梯度）。"""
    key = (int(S), str(device))
    t = _CAUSAL_MASK_CACHE.get(key)
    if t is None:
        m = np.triu(np.full((int(S), int(S)), -1e30, dtype=np.float32), k=1)
        t = Tensor(m, requires_grad=False, device=device)
        if len(_CAUSAL_MASK_CACHE) > 8:      # 内存上限：最多缓存 8 种长度
            _CAUSAL_MASK_CACHE.clear()
        _CAUSAL_MASK_CACHE[key] = t
    return t


def _softmax_causal_enabled():
    """TG_SOFTMAX_CAUSAL=1 时启用掩码内联 softmax（默认关：现版内核比通用路径慢）。"""
    import os
    return os.environ.get("TG_SOFTMAX_CAUSAL", "").strip().lower() in ("1", "true", "on", "yes")


def _softmax_causal_cuda(x):
    """causal softmax（掩码内联）：一行只看 j<=pos，省掉 mask 张量与一次加法。

    内核原地写，所以先 clone；反向用 attn_mask_softmax_bwd（被掩掉的列 P=0，
    dS 自动为 0，不需要再乘掩码）。
    """
    rt = _cuda()
    y = rt.clone(x._gpu)
    rt.attn_mask_softmax_(y, causal=True)
    out = Tensor._from_gpu(y, requires_grad=x.requires_grad, _parents=(x,))

    def bw(g, grads):
        if x.requires_grad:
            _acc(grads, x, rt.attn_mask_softmax_bwd(g, y))
    out._backward = bw
    return out


def _sdpa_materialized(q, k, v, causal, scale, attn_mask=None):
    """物化 scores/P 的 SDPA。默认路径，也是 lowmem 不合格时的回退。

    `attn_mask`：**加性**掩码（0 可见 / -inf 不可见），形状可广播到 `[BH, S, S]`。
    打包训练必须用它：块对角掩码让不同文档互不可见（详见 `v3_masks`）。
    """
    # ★ 转置乘走 cuBLAS 的 op=T，省掉 k 的一次真拷贝（旧写法 q @ k.swap_last2()）
    if (getattr(q, "device", "cpu") == "cuda" and q.ndim == 3
            and tuple(q.shape) == tuple(k.shape)):
        scores = _attn_scores(q, k, scale)
    else:
        scores = (q @ k.swap_last2()) * scale
    # 掩码内联的 causal softmax。
    # ★ 2026-11-14：内核已从"每行一个线程串行扫 k"改成"一块一行 + warp 归约"，
    #   在 S=2048 实测 0.0410 s vs 组合路径 0.0433 s（**快 6%**），S=512 仍略慢（0.96×）。
    #   所以默认按序列长度自动选：>=1024 用融合核，短序列保持组合路径。
    #   环境变量 TG_SOFTMAX_CAUSAL 仍可强制（1=强制开，0=强制关）。
    _env = os.environ.get("TG_SOFTMAX_CAUSAL", "").strip().lower()
    if _env in ("1", "true", "on", "yes"):
        _use_fused_sm = True
    elif _env in ("0", "false", "off", "no"):
        _use_fused_sm = False
    else:
        _use_fused_sm = int(scores.shape[-1]) >= 1024
    if causal and _use_fused_sm and attn_mask is None and getattr(scores, "device", "cpu") == "cuda":
        p = _softmax_causal_cuda(scores)
    else:
        if causal:
            scores = scores + _causal_mask(scores.shape[-1], scores.device)
        if attn_mask is not None:
            # ★ 加性掩码：与因果掩码**叠加**（两者都可能出现）。
            #   打包训练用它实现块对角（不同文档互不可见）。
            scores = scores + attn_mask
        p = _Functional.softmax(scores, axis=-1)
    return p @ v


def _sdpa_lowmem_ok(q, k, v, causal, dh):
    """内核契约：CUDA、causal、dh==64、三维同形状 [BH, S, Dh]。"""
    if not causal or dh != 64:
        return False
    if getattr(q, "device", "cpu") != "cuda":
        return False
    if getattr(k, "device", "cpu") != "cuda" or getattr(v, "device", "cpu") != "cuda":
        return False
    if q.ndim != 3 or tuple(k.shape) != tuple(q.shape) or tuple(v.shape) != tuple(q.shape):
        return False
    bh, seq, _dh = q.shape
    return int(bh) > 0 and int(seq) > 0


def _sdpa_lowmem(q, k, v, scale):
    """[BH, S, Dh] -> 显式 permute 成 [S, BH, Dh] 再调内核，结果 permute 回来。

    B=1 时 BH 就是 H，布局正好是内核要的 [S, H, Dh]。B>1 时各头本来就不互相看，
    把 BH 当成 nh 同样成立。反向不读 P：前向记下每行 m/l，attn_bwd_flash 每个
    Q 块只扫一遍 K/V。
    """
    rt = _cuda()
    rt.ensure_attn_nop()
    # 走运行时 permute，不把这次拷贝接进 autograd，避免和自定义 VJP 重复反传。
    q_sk = rt.permute(q._gpu, (1, 0, 2))
    k_sk = rt.permute(k._gpu, (1, 0, 2))
    v_sk = rt.permute(v._gpu, (1, 0, 2))
    o_sk, m_sk, l_sk = rt.attn_nop_fwd_stats(q_sk, k_sk, v_sk, scale)
    o_hs = rt.permute(o_sk, (1, 0, 2))
    need = bool(q.requires_grad or k.requires_grad or v.requires_grad)
    out = Tensor._from_gpu(o_hs, requires_grad=need, _parents=(q, k, v) if need else ())
    if not need:
        return out

    def bw(g, grads):
        g_sk = rt.permute(g, (1, 0, 2))
        dq_sk, dk_sk, dv_sk = rt.attn_nop_bwd_flash(
            g_sk, q_sk, k_sk, v_sk, o_sk, m_sk, l_sk, scale)
        if q.requires_grad:
            _acc(grads, q, rt.permute(dq_sk, (1, 0, 2)))
        if k.requires_grad:
            _acc(grads, k, rt.permute(dk_sk, (1, 0, 2)))
        if v.requires_grad:
            _acc(grads, v, rt.permute(dv_sk, (1, 0, 2)))

    out._backward = bw
    return out


def _tanh_cuda(x):
    """tanh 走融合内核（单次分配）；反向用 `1 − y²`，复用前向输出。

    原先没有 tanh 内核，靠 `2·sigmoid(2x)−1` 组合，实测 16 MB 输入吃掉 176 MiB（11×），
    是 `situ_glu_orv` 756 MiB 与 MoE 每层 450 MiB 的主要来源。
    """
    rt = _cuda()
    y = rt.tanh(x._gpu)
    out = Tensor._from_gpu(y, requires_grad=x.requires_grad, _parents=(x,))

    def bw(g, grads):
        if x.requires_grad:
            _acc(grads, x, rt.tanh_bwd(g, y))
    out._backward = bw
    return out


def _softmax_cuda(x, axis):
    """最后一维 softmax。反向用 y*(dy - sum(dy*y))，和 CPU 那条 exp/sum 链在 fp32 上差 ~1e-7。"""
    if axis != -1 and axis != x.ndim - 1:
        raise NotImplementedError("CUDA softmax 目前只支持最后一维，收到 axis=%r" % (axis,) + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    if x.ndim < 1:
        raise NotImplementedError("CUDA softmax 需要至少 1 维" + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    rt = _cuda()
    y = rt.softmax_lastdim(x._gpu)
    out = Tensor._from_gpu(y, requires_grad=x.requires_grad, _parents=(x,))

    def bw(g, grads):
        if x.requires_grad:
            _acc(grads, x, rt.softmax_lastdim_bwd(g, y))
    out._backward = bw
    return out


def _silu_cuda(x):
    """silu(x)。b 恒为 1，不再分配全 1，也不再分配用不到的 db。"""
    rt = _cuda()
    y = rt.clone(x._gpu)
    rt.silu_fwd_inplace(y)
    out = Tensor._from_gpu(y, requires_grad=x.requires_grad, _parents=(x,))

    def bw(g, grads):
        if not x.requires_grad:
            return
        _acc(grads, x, rt.silu_bwd_da(g, x._gpu))

    out._backward = bw
    return out


def _log_softmax_cuda(x, axis):
    """最后一维 log_softmax。反向是 g - softmax(x) * sum(g)。"""
    if axis != -1 and axis != x.ndim - 1:
        raise NotImplementedError("CUDA log_softmax 目前只支持最后一维，收到 axis=%r" % (axis,) + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    if x.ndim < 1:
        raise NotImplementedError("CUDA log_softmax 需要至少 1 维" + "。修法：看 docs/OP_COVERAGE.md 的支持矩阵；或用已有算子组合实现（ops_plugin.register）")
    rt = _cuda()
    y = rt.log_softmax_lastdim(x._gpu)
    out = Tensor._from_gpu(y, requires_grad=x.requires_grad, _parents=(x,))

    def bw(g, grads):
        if x.requires_grad:
            _acc(grads, x, rt.log_softmax_lastdim_bwd(g, y))
    out._backward = bw
    return out


def _cross_entropy_cuda(logits, target, ignore_index):
    """GPU 上的 mean CE。语义与 ce_fwd/ce_bwd 一致，不含 ignore_index。"""
    tgt = _as_index_array(target).reshape(-1)
    n, v = logits.shape
    if tgt.shape != (n,):
        raise ValueError("cross_entropy 目标长度 %s != N=%d" % (tgt.shape, n) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
    if np.any(tgt == ignore_index):
        return _cross_entropy_cuda_ignore(logits, tgt, ignore_index)
    if np.any((tgt < 0) | (tgt >= v)):
        raise ValueError("cross_entropy 目标越界，合法范围是 [0, %d)" % v + HINT_CE + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    rt = _cuda()
    tgt_buf = rt.h2d(tgt, dtype=np.int32, cache_slot="ce_tgt")
    loss_buf = rt.ce_fwd(logits._gpu, tgt_buf, n, v)
    out = Tensor._from_gpu(loss_buf, requires_grad=logits.requires_grad, _parents=(logits,))

    def bw(g, grads):
        if not logits.requires_grad:
            return
        # 根梯度几乎总是主机填好的 1。先读已知值，再发射 ce_bwd，
        # 这样就算真要回读，也不会把 ce_bwd 自己等进去。
        scale = rt.read_scalar(g)
        gpu = logits._gpu
        # 大词表才原地写。llama_parity 会在 backward 之后读 logits，小张量必须保住原值。
        # 视图也不能原地：那会改掉被 reshape 共享的上游激活。
        owned = getattr(gpu, "_base", None) is None and getattr(gpu, "_token", None) is not None
        if owned and n * v >= (1 << 20):
            dl = rt.ce_bwd(gpu, tgt_buf, n, v, out=gpu)
            if scale != 1.0:
                dl = rt.scale_inplace(dl, scale)
        else:
            dl = rt.ce_bwd(gpu, tgt_buf, n, v)
            if scale != 1.0:
                dl = rt.scale(dl, scale)
        _acc(grads, logits, dl)

    out._backward = bw
    return out


_CE_SEL_CACHE = {}


def _ce_selection(n, rows):
    """选择矩阵 S[nvalid,n] 与它的转置缓存（训练时 mask 每步相同，缓存后只建一次）。"""
    key = (int(n), int(rows.size), int(rows[0]) if rows.size else -1,
           int(rows[-1]) if rows.size else -1, hash(rows.tobytes()))
    hit = _CE_SEL_CACHE.get(key)
    if hit is not None:
        return hit
    nv = int(rows.size)
    S = np.zeros((nv, n), np.float32)
    S[np.arange(nv), rows] = 1.0
    ST = np.ascontiguousarray(S.T)
    # ★ 缓存 GPU 张量，否则每步都要把 40+40 MB 的选择矩阵传上显存（实测慢 2.4 倍）
    S_t = Tensor(S)
    ST_t = Tensor(ST)
    if len(_CE_SEL_CACHE) > 4:
        _CE_SEL_CACHE.clear()
    _CE_SEL_CACHE[key] = (S_t, ST_t)
    return S_t, ST_t


def _cross_entropy_cuda_ignore(logits, tgt, ignore_index):
    """★ GPU 版：用融合 gather/scatter 只处理有效行。

    演进（每一步都实测过）：
      v1 把整块 logits 拷到 CPU：7 MB D2H + 7 MB H2D，占整步 13%（8.8 ms）；
      v2 选择矩阵 S[nvalid,n] @ logits：数值正确，但那是 2×7 GFLOP 的 GEMM（k=4992），
         V100 上 CE 仍要 9.9 ms；
      v3（本版）复用 emb_fwd/emb_bwd（本质是「按 id 取行 / 散射累加」），
         只搬 nvalid×v 个元素，不再做长 K GEMM。

    注意：不能用「无效行填 safe 目标」的偷懒法 —— 那些行的 CE 会被算进均值，
    loss 会偏大（实测 8.09 vs 4.69），梯度却是对的，很容易被漏掉。
    """
    rt = _cuda()
    n, v = logits.shape
    tgt_arr = np.asarray(tgt).reshape(-1)
    mask = (tgt_arr != ignore_index)
    nvalid = int(mask.sum())
    if nvalid == 0:
        # 没有任何有效行时也要返回 **0 维标量**（与 nvalid>0 的路径一致），
        # 否则调用方 float(loss) 会报 "only 0-dimensional arrays can be converted"。
        # 梯度为 0：没有有效行就没有可学的信号。
        zero = rt.empty((), dtype=np.float32)
        out = Tensor._from_gpu(zero, requires_grad=logits.requires_grad, _parents=(logits,))

        def bw_noop(g, grads):
            return

        out._backward = bw_noop
        return out
    rows = np.flatnonzero(mask)
    ids_buf = rt.h2d(np.ascontiguousarray(rows, dtype=np.int32),
                     dtype=np.int32, cache_slot="ce_ids")
    lc_buf = rt.emb_fwd(logits._gpu, ids_buf, nvalid, v, n, (nvalid, v))
    tgt_buf = rt.h2d(np.ascontiguousarray(tgt_arr[rows], dtype=np.int32),
                     dtype=np.int32, cache_slot="ce_tgt")
    loss_buf = rt.ce_fwd(lc_buf, tgt_buf, nvalid, v)
    out = Tensor._from_gpu(loss_buf, requires_grad=logits.requires_grad, _parents=(logits,))

    def bw(g, grads):
        if not logits.requires_grad:
            return
        scale = rt.read_scalar(g)
        dlc = rt.ce_bwd(lc_buf, tgt_buf, nvalid, v)
        d = rt.emb_bwd(dlc, ids_buf, nvalid, v, (n, v))    # 无效行天然为 0
        if scale != 1.0:
            d = rt.scale(d, scale)
        _acc(grads, logits, d)

    out._backward = bw
    return out

class Parameter(Tensor):
    def __init__(self, data):
        # 与 torch.nn.Parameter 一致：也接受 Tensor 入参（迁移兼容）
        if isinstance(data, Tensor):
            data = data.to_numpy() if getattr(data, "device", "cpu") == "cuda" else data.data
        super().__init__(data, requires_grad=True)


def empty_gpu(shape, dtype=np.float32, requires_grad=False):
    """**只分配设备内存**、不建宿主 numpy 副本的 Tensor 构造。

    为什么必须有：`T.tensor(np_array)` 会先在宿主建数组再 h2d —— 3B 参数就是 12 GB 主机内存，
    与 ★3「不用到内存」直接冲突（实测：分配 3B fp32 权重时主机 OOM）。
    这里走 `rt.empty()`（纯 cuMemAlloc），`data` 保持 None ⇒ 主机内存不增长。
    """
    t = Tensor.__new__(Tensor)
    t.device = "cuda"
    t.data = None
    t.grad = None
    t.requires_grad = bool(requires_grad)
    t._parents = ()
    t._backward = None
    t._gpu = _cuda().empty(tuple(int(x) for x in shape), dtype)
    t._lazy_T = None
    t._lazy_shape = None
    return t


class Module:
    """对应 torch.nn.Module：参数注册、子模块、state_dict。"""

    def __init__(self):
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_mods", {})
        # 训练/推理模式：Dropout、BatchNorm 依赖它（与 torch.nn.Module 对齐）
        object.__setattr__(self, "training", True)

    def train(self, mode=True):
        """切换训练/推理模式，并递归到子模块。返回 self（可链式）。"""
        object.__setattr__(self, "training", bool(mode))
        for m in self._mods.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __setattr__(self, name, value):
        if name in ("_params", "_mods"):
            object.__setattr__(self, name, value)
            return
        if isinstance(value, Parameter):
            self._params[name] = value
        elif isinstance(value, Module):
            self._mods[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, *a, **kw):
        return self.forward(*a, **kw)

    def forward(self, *a, **kw):
        raise NotImplementedError

    def named_parameters(self, prefix=""):
        for k, p in self._params.items():
            yield prefix + k, p
        for k, m in self._mods.items():
            yield from m.named_parameters(prefix + k + ".")

    def parameters(self):
        """所有参数（**按对象身份去重**，与 PyTorch 一致）。

        为什么必须去重：绑定词嵌入（`tie_embedding`，GPT 类模型的常见做法）会让
        `embed.weight` 与 `lm_head.weight` 指向**同一个 Parameter**，它在 `named_parameters()`
        里会出现两次。若 `parameters()` 原样返回两份，优化器就会**每步把它更新两遍**
        ——实测导致与分层更新相差 7.2e-03，而且不报错。
        PyTorch 的 `parameters()` 同样会去掉重复参数，所以这里对齐它的语义。
        """
        out, seen = [], set()
        for _, p in self.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            out.append(p)
        return out

    def state_dict(self, prefix=""):
        out = {}
        for k, p in self._params.items():
            if getattr(p, "device", "cpu") == "cuda":
                out[prefix + k] = p.to_numpy().copy()
            else:
                out[prefix + k] = p.data.copy()
        for k, m in self._mods.items():
            out.update(m.state_dict(prefix + k + "."))
        return out

    def load_state_dict_stream(self, items, dtype=None):
        """**流式**装载：`items` 是 (name, ndarray) 的可迭代对象，逐个上传即释放。

        与 `load_state_dict(dict)` 的区别（3B 权重装载必用这个）：
        * 调用方无需把全部权重同时放进内存（dict 版本要求 12 GB 宿主张量，实测会 OOM）；
        * 覆盖已有 `_gpu` 时**先释放旧块**（否则构造期分配的随机权重会泄漏一半显存）。

        宿主峰值 = 单个张量；显存峰值 = 参数总量。任一张量缺失/多余/形状不符都直接报错。

        `dtype`：设备侧存储精度，`None` 保持历史行为 = fp32。
        传 `np.float16` 可把权重按 fp16 存放（2.999B 全参训练在 32 GB 卡上必须如此：
        fp32 权重+梯度+优化器状态 = 44.7 GiB 放不下，fp16 全套 = 22.35 GiB 才进得去）。
        **数值敏感处（RMSNorm / final_norm / RoPE 表 / head）应保持 fp32**，由调用方决定。
        """
        _dt = np.float32 if dtype is None else dtype
        params = {n: p for n, p in self.named_parameters()}
        seen = set()
        for name, arr in items:
            p = params.get(name)
            if p is None:
                raise KeyError("没有参数 %r。修法：用 named_parameters() 核对名字（%d 个可用）"
                               % (name, len(params)))
            arr = np.ascontiguousarray(arr, _dt)
            try:
                want = tuple(int(x) for x in p.shape)
            except Exception:                      # _gpu 已被释放：用构造时记下的形状
                want = tuple(int(x) for x in (getattr(p, "_lazy_shape", None) or arr.shape))
            if tuple(arr.shape) != want:
                raise ValueError("参数 %r 形状不符：给的是 %s，要的是 %s。修法：核对导出脚本"
                                 % (name, arr.shape, want))
            if getattr(p, "device", "cpu") == "cuda":
                rt = _cuda()
                old = getattr(p, "_gpu", None)
                p._gpu = rt.h2d(arr, dtype=_dt)
                p.data = None
                if old is not None and int(old.n) > 0:
                    try:
                        rt.free(int(old.ptr))
                    except Exception as exc:        # noqa: BLE001
                        raise RuntimeError("释放旧缓冲失败（%s）：%r。修法：检查是否被别的张量共享"
                                           % (name, exc)) from exc
            else:
                p.data = arr.copy()
            seen.add(name)
        missing = set(params) - seen
        if missing:
            raise KeyError("缺 %d 个参数，例如 %s。修法：补齐或确认模型结构与权重一致"
                           % (len(missing), sorted(missing)[:3]))
        return self

    def load_state_dict(self, sd):
        for name, p in self.named_parameters():
            if name not in sd:
                raise KeyError("缺少参数: %s" % name + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
            if getattr(p, "device", "cpu") == "cuda":
                arr = np.asarray(sd[name], dtype=np.float32).reshape(p.shape)
                p._gpu = _cuda().h2d(arr)
            else:
                p.data = np.asarray(sd[name], dtype=np.float32).reshape(p.data.shape)
        return self

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def __repr__(self):
        return "%s(params=%d)" % (type(self).__name__, self.num_params())


class Linear(Module):
    # 默认 bias=True，与 torch.nn.Linear 对齐（迁移兼容性）。
    # 之前默认 False：PyTorch 程序改导入后参数量会不一致，是迁移的障碍。
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        k = 1.0 / np.sqrt(in_features)
        self.weight = Parameter(np.random.randn(out_features, in_features).astype(np.float32) * k)
        self.bias = Parameter(np.zeros(out_features, np.float32)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


class Embedding(Module):
    def __init__(self, num_embeddings, dim, weight=None):
        """weight 给定时**直接挂上**，不再随机初始化一张新表。

        为什么要有这个参数：权重绑定（tie）的模型会把词嵌入同时当作 lm_head，
        典型写法是"建 Embedding，再把 weight 换成共享 Parameter"——
        那等于先白分配一张 V*D 的表（0.5B 模型 545 MB）再扔掉。
        用 weight= 可以直接复用已有 Parameter，既省显存也不会多一次随机初始化。
        """
        super().__init__()
        if weight is not None:
            if tuple(int(x) for x in weight.shape) != (int(num_embeddings), int(dim)):
                raise ValueError(
                    "Embedding: 传入的 weight 形状是 %s，与 (V=%d, D=%d) 不符。"
                    "修法：确认 cfg 的 V/D 与权重一致"
                    % (tuple(int(x) for x in weight.shape), num_embeddings, dim))
            self.weight = weight
            return
        self.weight = Parameter((np.random.randn(num_embeddings, dim) * 0.02).astype(np.float32))

    def forward(self, idx):
        # 接受 Tensor 或 numpy；CUDA 上走融合 gather 内核（emb_fwd，1 个 kernel）
        if isinstance(idx, Tensor):
            idx = idx.to_numpy() if getattr(idx, "device", "cpu") == "cuda" else idx.data
        idx = np.asarray(idx, dtype=np.int64)
        # CPU 路径原本没有校验，越界直接抛 numpy 的
        # "index 9 is out of bounds for axis 0 with size 5" —— 看不出是哪个算子、怎么修
        vocab = int(self.weight.shape[0])
        if idx.size and ((idx < 0) | (idx >= vocab)).any():
            bad = idx[(idx < 0) | (idx >= vocab)][:4].tolist()
            raise IndexError(
                "Embedding: 下标越界 %s，合法范围是 [0, %d)（词表大小 %d）。"
                "修法：确认 tokenizer 的 id 空间与该模型一致"
                "（Qwen2.5/Qwen3 词表都是 151936），或检查 --limit/切片是否压过 vocab"
                % (bad, vocab, vocab))
        if getattr(self.weight, "device", "cpu") == "cuda":
            return self._forward_cuda(idx)
        return self.weight[idx.reshape(-1)].reshape(idx.shape + (self.weight.shape[1],))

    def _forward_cuda(self, idx):
        rt = _cuda()
        flat = np.ascontiguousarray(idx, dtype=np.int64).reshape(-1)
        vocab = int(self.weight.shape[0])
        d = int(self.weight.shape[1])
        if flat.size and ((flat < 0) | (flat >= vocab)).any():
            raise IndexError("Embedding 下标越界，合法范围是 [0, %d)" % vocab + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        seq = int(flat.shape[0])
        ids = rt.h2d(flat.astype(np.int32), dtype=np.int32, cache_slot="emb_ids")
        out_shape = tuple(int(s) for s in idx.shape) + (d,)
        y = rt.emb_fwd(self.weight._gpu, ids, seq, d, vocab, out_shape)
        w = self.weight
        out = Tensor._from_gpu(y, requires_grad=w.requires_grad, _parents=(w,))

        def bw(g, grads):
            if not w.requires_grad:
                return
            _acc(grads, w, rt.emb_bwd(g, ids, seq, d, w.shape))

        out._backward = bw
        return out


def silu_mul(a, b):
    """out = silu(a) * b —— 融合实现（CUDA 上 fwd/bwd 各 1 个 kernel，组合实现要 5 个）。

    silu_mul_fwd 是原地写 a，所以先 clone 一份；反向要用**原始 a**，因此 a 本身不能被改。
    """
    if (getattr(a, "device", "cpu") != "cuda" or getattr(b, "device", "cpu") != "cuda"):
        inv = (Tensor(np.array([1.0], np.float32)) + (-a).exp()) ** -1.0
        return a * inv * b
    rt = _cuda()
    y = rt.silu_mul_fwd(rt.clone(a._gpu), b._gpu)
    out = Tensor._from_gpu(y, requires_grad=(a.requires_grad or b.requires_grad),
                           _parents=(a, b))

    def bw(g, grads):
        da, db = rt.silu_mul_bwd(g, a._gpu, b._gpu)
        if a.requires_grad:
            _acc(grads, a, da)
        if b.requires_grad:
            _acc(grads, b, db)

    out._backward = bw
    return out


# ---- M6：自研融合内核（交错式 RoPE / 原地残差加 / causal softmax）----
# 三个内核都是在 cuda.py 的 NVRTC 源码里追加的（_FUSED_KERNELS），这里只做 Python 侧接线。
# 设计要求：**默认安全** —— 任何时候内核不可用/抛异常，都退回原来的组合实现，只警告一次。
_FUSED_WARNED = {}
_ROPE_P_CACHE = {}


def _fused_kernels_on():
    """总开关：TG_NO_FUSED=1 时强制走组合实现（给 A/B 对拍和线上排障用）。"""
    try:
        import os
        return os.environ.get("TG_NO_FUSED", "").strip().lower() not in (
            "1", "true", "on", "yes")
    except Exception:
        return True


def _fused_warn_once(op, exc):
    if _FUSED_WARNED.get(op):
        return
    _FUSED_WARNED[op] = True
    try:
        import warnings
        warnings.warn("tgai_go.train.%s 融合内核不可用，已回退到组合实现（后续静默）：%s"
                      % (op, exc), RuntimeWarning, stacklevel=3)
    except Exception:
        pass


def _rope_perm(dh):
    """交错式 RoPE 的置换矩阵：缓存一份，反向/回退都要用。"""
    P = _ROPE_P_CACHE.get(dh)
    if P is not None:
        return P
    arr = np.zeros((dh, dh), np.float32)
    for j in range(dh):
        if j % 2 == 0:
            arr[j + 1, j] = -1.0
        elif j - 1 >= 0:
            arr[j - 1, j] = 1.0
    P = Tensor(arr)
    _ROPE_P_CACHE[dh] = P
    return P


def _rope_composed(x, cos, sin):
    """参考实现（与 examples 里的 rope() 逐位一致）：x*cos + (x@P)*sin。"""
    dh = int(x.shape[-1]) if x.ndim else 0
    if dh < 2 or dh % 2:
        raise ValueError("RoPE 需要最后一维为 >=2 的偶数，收到 %s" % (x.shape,) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
    return x * cos + (x @ _rope_perm(dh)) * sin


def _rope_rotate_np(g, cos, sin):
    """numpy 版 RoPE 反向（只给「前向成功但反向内核异常」这种极端情况兜底用）。

    与内核 fused_rope_bwd 同一套下标：dx[2k] = g[2k]*cos[2k] + g[2k+1]*sin[2k+1]、
    dx[2k+1] = g[2k+1]*cos[2k+1] - g[2k]*sin[2k]。
    """
    out = np.array(g, copy=True)
    out[..., 0::2] = g[..., 0::2] * cos[..., 0::2] + g[..., 1::2] * sin[..., 1::2]
    out[..., 1::2] = g[..., 1::2] * cos[..., 1::2] - g[..., 0::2] * sin[..., 0::2]
    return out


def rope_half(x, cos, sin, seq):
    """half-split RoPE（rotate_half：配对 d / d+dh/2）。CUDA 上前后向各 1 个 kernel。

    组合实现要 1 个 cat + 2 次乘 + 1 次加（反向 14 个 kernel），这里各 1 个。
    cos/sin 是 [seq, dh] 的常量表（模型里 reshape 成 [1,seq,dh] 也等价，元素数一致）。
    非 CUDA 或异常 → 回退组合实现，数值一致。
    """
    if (getattr(x, "device", "cpu") != "cuda"
            or getattr(cos, "device", "cpu") != "cuda"
            or getattr(sin, "device", "cpu") != "cuda"):
        return _rope_half_composed(x, cos, sin)
    try:
        rt = _cuda()
        y = rt.rope_half_ip(rt.clone(x._gpu), cos._gpu, sin._gpu, int(seq))
        rt._fn("rope_half_bwd")              # 先把反向符号解析出来，别等反向才炸
    except Exception as exc:
        _fused_warn_once("rope_half", exc)
        return _rope_half_composed(x, cos, sin)
    out = Tensor._from_gpu(y, requires_grad=x.requires_grad, _parents=(x,))
    if x.requires_grad:
        def bw(g, grads):
            _acc(grads, x, rt.rope_half_bwd(g, cos._gpu, sin._gpu, int(seq)))
        out._backward = bw
    return out


def _rope_half_composed(x, cos, sin):
    """half-split 的组合实现（CPU / 回退路径）。与 `rope_half` 数值一致。"""
    dh = int(x.shape[-1])
    h = dh // 2
    rot = cat([-x[..., h:], x[..., :h]], dim=-1)
    return x * cos + rot * sin


def fused_rope(x, cos, sin):
    """交错式 RoPE：out[2k]=x[2k]*cos-x[2k+1]*sin，out[2k+1]=x[2k]*sin+x[2k+1]*cos。

    CUDA 上 fwd/bwd 各 1 个 kernel（原地旋转 clone 出来的那份）；组合实现是
    1 个 matmul + 3 个 elementwise。cos/sin 是常量（不需要梯度）才走融合。
    任何异常 → 回退组合实现，只警告一次，绝不抛给调用方。
    """
    if not _fused_kernels_on():
        return _rope_composed(x, cos, sin)
    if (getattr(x, "device", "cpu") != "cuda"
            or getattr(cos, "device", "cpu") != "cuda"
            or getattr(sin, "device", "cpu") != "cuda"):
        return _rope_composed(x, cos, sin)
    if cos.requires_grad or sin.requires_grad:
        # cos/sin 可训练时反向要多算两份梯度，融合内核不覆盖 ⇒ 直接走组合实现。
        return _rope_composed(x, cos, sin)
    if not x.requires_grad:
        # 不需要梯度就不必 clone：内核本来就是原地，直接在自己的缓冲上转。
        try:
            rt = _cuda()
            y = rt.fused_rope_(x._gpu, cos._gpu, sin._gpu)
            return Tensor._from_gpu(y, requires_grad=False)
        except Exception as exc:
            _fused_warn_once("fused_rope", exc)
            return _rope_composed(x, cos, sin)
    try:
        rt = _cuda()
        y = rt.fused_rope_(rt.clone(x._gpu), cos._gpu, sin._gpu)
        # 前向内核拿到了就先把反向符号也解析出来，避免反向时才炸。
        rt._fn("fused_rope_bwd")
    except Exception as exc:
        _fused_warn_once("fused_rope", exc)
        return _rope_composed(x, cos, sin)
    out = Tensor._from_gpu(y, requires_grad=True, _parents=(x,))

    def bw(g, grads):
        if not x.requires_grad:
            return
        try:
            _acc(grads, x, rt.fused_rope_bwd(g, cos._gpu, sin._gpu))
        except Exception as exc:
            _fused_warn_once("fused_rope_bwd", exc)
            gg = rt.d2h(g)
            cc = rt.d2h(cos._gpu)
            ss = rt.d2h(sin._gpu)
            _acc(grads, x, rt.h2d(_rope_rotate_np(gg, cc, ss)))

    out._backward = bw
    return out


def fused_add_(x, y):
    """x += y。CUDA 且 x 不需要梯度时走原地内核（省一次分配）；否则退回 x + y。

    ★ x.requires_grad 时**不能**原地改 x 的缓冲：训练图里更早的算子（如 RMSNorm
    反向）在 backward 时还要读 x 的前向值，原地改会把梯度算错。这时退化成
    `x + y`（同一条 ewise_add，结果逐位相同，只是多一次分配）。
    """
    if (not _fused_kernels_on()
            or getattr(x, "device", "cpu") != "cuda"
            or getattr(y, "device", "cpu") != "cuda"
            or x.requires_grad
            or tuple(x.shape) != tuple(y.shape)):
        return x + y
    try:
        rt = _cuda()
        return Tensor._from_gpu(rt.add_(x._gpu, y._gpu), requires_grad=False)
    except Exception as exc:
        _fused_warn_once("fused_add_", exc)
        return x + y


def attn_mask_softmax(scores, causal=True, need_lse=False):
    """causal mask + softmax（原地），可选返回每行 logsumexp。

    只做 fused 前向 + 反向，不改变 autograd 图；CUDA 不可用时回退到组合实现。
    """
    if (not _fused_kernels_on() or getattr(scores, "device", "cpu") != "cuda"
            or not scores.requires_grad or int(scores.ndim) != 2
            or not causal):
        return _softmax_composed(scores, causal)
    try:
        rt = _cuda()
        p_buf, lse = rt.attn_mask_softmax_(rt.clone(scores._gpu), causal=True,
                                           want_lse=bool(need_lse))
        rt._fn("attn_mask_softmax_bwd")
    except Exception as exc:
        _fused_warn_once("attn_mask_softmax", exc)
        return _softmax_composed(scores, causal)
    out = Tensor._from_gpu(p_buf, requires_grad=True, _parents=(scores,))

    def bw(g, grads):
        if not scores.requires_grad:
            return
        try:
            _acc(grads, scores, rt.attn_mask_softmax_bwd(g, p_buf))
        except Exception as exc:
            _fused_warn_once("attn_mask_softmax_bwd", exc)
            gg = rt.d2h(g)
            pp = rt.d2h(p_buf)
            dot = (gg * pp).sum(axis=-1, keepdims=True)
            _acc(grads, scores, rt.h2d((pp * (gg - dot)).astype(np.float32)))

    out._backward = bw
    if need_lse:
        return out, (None if lse is None else Tensor._from_gpu(lse))
    return out


def _softmax_composed(scores, causal=True):
    x = scores
    if causal:
        k = int(x.shape[-1])
        n = int(x.numel())
        rows = n // k if k else 0
        mask = np.zeros((rows, k), np.float32)
        for r in range(rows):
            mask[r, r % k + 1:] = -np.inf
        x = x + Tensor(mask).reshape(tuple(x.shape))
    e = x.exp()
    return e * (e.sum(axis=-1, keepdims=True) ** -1.0)


class RMSNorm(Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = Parameter(np.ones(dim, np.float32))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # 形状校验放在最前（CPU/GPU 共用），否则 CPU 上是 numpy 的
        # "operands could not be broadcast together with shapes (3,4) (7,)" —— 看不出根因
        d_w = int(self.weight.numel())
        d_x = int(x.shape[-1]) if getattr(x, "ndim", 0) else 0
        if d_x != d_w:
            raise ValueError(
                "RMSNorm: 形状不匹配 —— 输入最后一维 %d，归一化维度 %d（x.shape=%s）。"
                "修法：RMSNorm 只归一化最后一维；Qwen3 的 q_norm/k_norm 是 head_dim"
                "（0.6B 为 128），不是 hidden（1024）"
                % (d_x, d_w, tuple(getattr(x, "shape", ()))))
        if getattr(x, "device", "cpu") == "cuda":
            return self._forward_cuda(x)
        var = (x * x).mean(axis=-1, keepdims=True)
        inv = (var + Tensor(np.array([self.eps], np.float32))) ** -0.5
        return x * inv * self.weight

    def _forward_cuda(self, x):
        rt = _cuda()
        if x.ndim < 1:
            raise ValueError("RMSNorm 需要至少 1 维" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        d = int(x.shape[-1])
        if int(self.weight.numel()) != d:
            raise ValueError("RMSNorm weight 长度 %d != 最后一维 %d" % (self.weight.numel(), d) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        seq = int(x.numel() // d) if d else 0
        eps = float(self.eps)
        y = rt.rmsnorm_fwd(x._gpu, self.weight._gpu, seq, d, eps)
        w = self.weight
        out = Tensor._from_gpu(
            y, requires_grad=(x.requires_grad or w.requires_grad), _parents=(x, w))

        def bw(g, grads):
            # rt.deterministic=True 时走"无 atomic"版本：dw 按列升序串行累加，
            # 保证同一输入两次运行得到逐位相同的梯度（默认关闭，快路径仍是原子版）。
            if getattr(rt, "deterministic", False) and w.requires_grad:
                dw = rt.empty((d,), dtype=__import__("numpy").float32)
                dx = rt.rmsnorm_bwd_det(g, x._gpu, w._gpu, seq, d, eps, dw)
            else:
                dx, dw = rt.rmsnorm_bwd(g, x._gpu, w._gpu, seq, d, eps)
            if x.requires_grad:
                _acc(grads, x, dx)
            if w.requires_grad:
                _acc(grads, w, dw)

        out._backward = bw
        return out


class CausalSelfAttention(Module):
    """causal 多头自注意力（对应引擎里的 attn 子层）。

    权重拆成独立的 q/k/v/out 四个 Linear，命名与 .TG 里的
    `blocks.{i}.attn.{q,k,v,out}_proj.weight` 一一对应，便于直出。
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model 必须能被 n_heads 整除" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = Linear(d_model, d_model, bias=False)
        self.k_proj = Linear(d_model, d_model, bias=False)
        self.v_proj = Linear(d_model, d_model, bias=False)
        self.out_proj = Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head

        def split(t):
            # [B,S,D] -> [B,S,H,Dh] -> permute -> [B,H,S,Dh] -> [B*H, S, Dh]
            #
            # **必须 permute**：从 [B,S,H,Dh] 直接 reshape 到 [B*H,S,Dh] 会把 head 与 seq
            # 两个轴混在一起（实测与逐头参照相差 9.6e-01，等于完全算错，而且不报错）。
            return t.reshape(B, S, H, Dh).permute(0, 2, 1, 3).reshape(B * H, S, Dh)

        q = split(self.q_proj(x.reshape(B * S, D)))
        k = split(self.k_proj(x.reshape(B * S, D)))
        v = split(self.v_proj(x.reshape(B * S, D)))
        o = _Functional.scaled_dot_product_attention(q, k, v, causal=True)
        o = o.reshape(B, H, S, Dh).permute(0, 2, 1, 3).reshape(B, S, H * Dh)
        return self.out_proj(o.reshape(B * S, D)).reshape(B, S, D)


class Sequential(Module):
    def __init__(self, *mods):
        super().__init__()
        self._seq = list(mods)
        for i, m in enumerate(self._seq):
            setattr(self, "m%d" % i, m)

    def forward(self, x):
        for m in self._seq:
            x = m(x)
        return x


class ModuleList(Module):
    def __init__(self, mods):
        super().__init__()
        self._list = list(mods)
        for i, m in enumerate(self._list):
            setattr(self, "m%d" % i, m)

    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __iter__(self):
        return iter(self._list)


def _sigmoid32(v):
    """与 moe_sigmoid 同一套 exp32：1 / (1 + exp32(-v))。"""
    v = np.asarray(v, dtype=np.float32)
    neg = np.multiply(v, np.float32(-1.0), dtype=np.float32)
    e = np.asarray(_exp32(neg), dtype=np.float32)
    den = np.add(np.float32(1.0), e, dtype=np.float32)
    return np.divide(np.float32(1.0), den, dtype=np.float32)


def _sigmoid_deriv32(y):
    """moeblk 用的 g*(1-g)，不是 exp32 多项式的导数。CPU / CUDA 同一式子。"""
    y = np.asarray(y, dtype=np.float32)
    neg = np.multiply(y, np.float32(-1.0), dtype=np.float32)
    one_m = np.add(np.float32(1.0), neg, dtype=np.float32)
    return np.multiply(y, one_m, dtype=np.float32)


def _sigmoid_cuda(x):
    rt = _cuda()
    buf = rt.moe_sigmoid(x._gpu)
    out = Tensor._from_gpu(buf, requires_grad=x.requires_grad, _parents=(x,))

    def bw(g, grads):
        if not x.requires_grad:
            return
        one = rt.full(tuple(int(s) for s in buf.shape), 1.0)
        deriv = rt.mul(buf, rt.add(one, rt.scale(buf, -1.0)))
        _acc(grads, x, rt.mul(g, deriv))

    out._backward = bw
    return out


def _sigmoid_tensor(x):
    if getattr(x, "device", "cpu") == "cuda":
        return _sigmoid_cuda(x)
    y = _sigmoid32(x.data)
    out = Tensor(y, requires_grad=x.requires_grad, _parents=(x,))

    def bw(g, grads):
        if x.requires_grad:
            _acc(grads, x, np.asarray(g, dtype=np.float32) * _sigmoid_deriv32(y))

    out._backward = bw
    return out


def _moe_topk_idx(logits, k):
    """logit 更大者优先，平局取更小下标。和 moe_topk kernel 同一套扫描。"""
    z = np.asarray(logits, dtype=np.float32)
    n, e = int(z.shape[0]), int(z.shape[1])
    k = int(k)
    idx = np.empty((n, k), dtype=np.int64)
    for t in range(n):
        row = z[t]
        chosen = []
        for s in range(k):
            best_e = -1
            best_v = np.float32(0.0)
            for j in range(e):
                used = False
                for prev in chosen:
                    if prev == j:
                        used = True
                        break
                if used:
                    continue
                v = np.float32(row[j])
                if best_e < 0 or v > best_v or (v == best_v and j < best_e):
                    best_e = j
                    best_v = v
            chosen.append(best_e)
            idx[t, s] = best_e
    return idx


def _moe_swiglu(rows, expert):
    gate = rows @ expert.gate.weight.T
    up = rows @ expert.up.weight.T
    hidden = functional.silu(gate) * up
    return hidden @ expert.down.weight.T


def _moe_scatter_cpu(n, d, tok, rows):
    tok = np.asarray(tok, dtype=np.int64)
    buf = np.zeros((int(n), int(d)), dtype=np.float32)
    if tok.size:
        np.add.at(buf, tok, np.asarray(rows.data, dtype=np.float32))
    out = Tensor(buf, requires_grad=rows.requires_grad, _parents=(rows,))

    def bw(g, grads):
        if rows.requires_grad:
            _acc(grads, rows, np.asarray(g, dtype=np.float32)[tok])

    out._backward = bw
    return out


def _moe_take_cuda(src, idx_buf, n, e, k):
    rt = _cuda()
    buf = rt.moe_gather_pairs(src._gpu, idx_buf, n, e, k)
    out = Tensor._from_gpu(buf, requires_grad=src.requires_grad, _parents=(src,))

    def bw(g, grads):
        if src.requires_grad:
            _acc(grads, src, rt.moe_scatter_pairs(g, idx_buf, n, e, k))

    out._backward = bw
    return out


def _moe_gather_rows_cuda(x, tok, begin, m, d):
    rt = _cuda()
    buf = rt.moe_gather_rows(x._gpu, tok, begin, m, d)
    out = Tensor._from_gpu(buf, requires_grad=x.requires_grad, _parents=(x,))

    def bw(g, grads):
        if x.requires_grad:
            _acc(grads, x, rt.moe_scatter_rows(g, tok, begin, m, int(x.shape[0]), d))

    out._backward = bw
    return out


def _moe_scatter_rows_cuda(n, d, tok, begin, m, rows):
    rt = _cuda()
    buf = rt.moe_scatter_rows(rows._gpu, tok, begin, m, n, d)
    out = Tensor._from_gpu(buf, requires_grad=rows.requires_grad, _parents=(rows,))

    def bw(g, grads):
        if rows.requires_grad:
            _acc(grads, rows, rt.moe_gather_rows(g, tok, begin, m, d))

    out._backward = bw
    return out


def _moe_gather_slots_cuda(w, tok, slot, begin, m, k):
    rt = _cuda()
    n = int(w.shape[0])
    buf = rt.moe_gather_slots(w._gpu, tok, slot, begin, m, k)
    out = Tensor._from_gpu(buf, requires_grad=w.requires_grad, _parents=(w,))

    def bw(g, grads):
        if w.requires_grad:
            _acc(grads, w, rt.moe_scatter_slots(g, tok, slot, begin, m, n, k))

    out._backward = bw
    return out


def _moe_row_scale_cuda(y, w):
    rt = _cuda()
    buf = rt.moe_row_scale(y._gpu, w._gpu)
    rg = y.requires_grad or w.requires_grad
    out = Tensor._from_gpu(buf, requires_grad=rg, _parents=(y, w))

    def bw(g, grads):
        if y.requires_grad:
            _acc(grads, y, rt.moe_row_scale(g, w._gpu))
        if w.requires_grad:
            _acc(grads, w, rt.moe_row_dot(g, y._gpu))

    out._backward = bw
    return out


class _MoEExpert(Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = Linear(d_model, d_ff, bias=False)
        self.up = Linear(d_model, d_ff, bias=False)
        self.down = Linear(d_ff, d_model, bias=False)


class _ExpertList(Module):
    """名字用 "0","1"，这样 state_dict 是 experts.0.gate.weight，不是 experts.m0。"""

    def __init__(self, experts):
        super().__init__()
        self._n = len(experts)
        for i, expert in enumerate(experts):
            setattr(self, str(i), expert)

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        i = int(i)
        if i < 0:
            i += self._n
        return getattr(self, str(i))

    def __iter__(self):
        for i in range(self._n):
            yield self[i]


def _moe_grouped_cuda(x, w, tok_buf, slot_buf, off, experts, n, d, k):
    """8 个专家的 GEMM 合成一次 cublasSgemmGroupedBatched。

    token 数不必相同，也不 pad：每组 group_size=1。小形状（自检）返回 None，
    仍走逐专家的原路径，保证和 CPU 的逐位/1e-5 判据不变。
    反向里 pad 不存在；没选中的专家不写梯度。
    """
    counts = [int(off[e + 1]) - int(off[e]) for e in range(len(experts))]
    active = [e for e, c in enumerate(counts) if c > 0]
    # ★ 2026-11-08：融合分组路径（cublasSgemmGroupedBatched）在**某专家行数 >64** 时
    # `ewise_mul_v4` 越界（实测 n=128 崩、n=8 正常）。在它修好之前**默认关闭**，
    # 走已验证的逐专家回退（数值一致，只是慢一些）。
    # 需要重新打开做性能对比时：设环境变量 TGAI_MOE_GROUPED=1。
    if os.environ.get("TGAI_MOE_GROUPED", "0") != "1":
        return None
    if not active or _builtin_max(counts) <= 64:
        return None
    rt = _cuda()
    rows = [rt.moe_gather_rows(x._gpu, tok_buf, int(off[e]), counts[e], d) for e in active]
    wg = [experts[e].gate.weight._gpu for e in active]
    wu = [experts[e].up.weight._gpu for e in active]
    wd = [experts[e].down.weight._gpu for e in active]
    gate = rt.grouped_nt(rows, wg)
    up = rt.grouped_nt(rows, wu)
    silu = []
    for g in gate:
        s = rt.clone(g)
        rt.silu_fwd_inplace(s)
        silu.append(s)
    hidden = [rt.mul(s, u) for s, u in zip(silu, up)]
    y_rows = rt.grouped_nt(hidden, wd)
    wws = []
    out_buf = rt.zeros((n, d), dtype=np.float32)
    for e, y in zip(active, y_rows):
        ww = rt.moe_gather_slots(w._gpu, tok_buf, slot_buf, int(off[e]), counts[e], k)
        wws.append(ww)
        scaled = rt.moe_row_scale(y, ww)
        rt.moe_scatter_rows_add(scaled, tok_buf, int(off[e]), counts[e], out_buf)
    parents = [x, w]
    for e in active:
        parents.append(experts[e].gate.weight)
        parents.append(experts[e].up.weight)
        parents.append(experts[e].down.weight)
    need = any(getattr(p, "requires_grad", False) for p in parents)
    out = Tensor._from_gpu(out_buf, requires_grad=need, _parents=tuple(parents) if need else ())
    if not need:
        return out
    # 盒子而不是默认参数：反传用完就能丢掉前向激活，后面的注意力反传才能复用这块池。
    saved_box = [(rows, gate, up, silu, hidden, y_rows, wws)]

    def bw(g, grads, _box=saved_box, _active=tuple(active), _counts=tuple(counts)):
        rows_s, gate_s, up_s, silu_s, hidden_s, y_s, ww_s = _box[0]
        d_scaled = [rt.moe_gather_rows(g, tok_buf, int(off[e]), _counts[e], d) for e in _active]
        d_y = [rt.moe_row_scale(ds, ww) for ds, ww in zip(d_scaled, ww_s)]
        d_ww = [rt.moe_row_dot(ds, y) for ds, y in zip(d_scaled, y_s)]
        d_scaled = None
        y_s = None
        ww_s = None
        d_hidden = rt.grouped_nn(d_y, wd)
        d_wd = rt.grouped_tn(d_y, hidden_s)
        hidden_s = None
        d_y = None
        d_s = [rt.mul(dh, u) for dh, u in zip(d_hidden, up_s)]
        d_up = [rt.mul(dh, s) for dh, s in zip(d_hidden, silu_s)]
        d_hidden = None
        up_s = None
        silu_s = None
        d_gate = [rt.silu_bwd_da(ds, gpre) for ds, gpre in zip(d_s, gate_s)]
        d_s = None
        gate_s = None
        d_wg = rt.grouped_tn(d_gate, rows_s)
        d_wu = rt.grouped_tn(d_up, rows_s)
        d_rows_g = rt.grouped_nn(d_gate, wg)
        d_rows_u = rt.grouped_nn(d_up, wu)
        d_gate = None
        d_up = None
        rows_s = None
        _box[0] = None
        if x.requires_grad:
            dx = rt.zeros((n, d), dtype=np.float32)
            for e, a, b in zip(_active, d_rows_g, d_rows_u):
                both = rt.add(a, b)
                rt.moe_scatter_rows_add(both, tok_buf, int(off[e]), _counts[e], dx)
            _acc(grads, x, dx)
        for e, dw in zip(_active, d_wg):
            if experts[e].gate.weight.requires_grad:
                _acc(grads, experts[e].gate.weight, dw)
        for e, dw in zip(_active, d_wu):
            if experts[e].up.weight.requires_grad:
                _acc(grads, experts[e].up.weight, dw)
        for e, dw in zip(_active, d_wd):
            if experts[e].down.weight.requires_grad:
                _acc(grads, experts[e].down.weight, dw)
        if w.requires_grad:
            acc_w = None
            for e, dw in zip(_active, d_ww):
                part = rt.moe_scatter_slots(dw, tok_buf, slot_buf, int(off[e]), _counts[e], n, k)
                acc_w = part if acc_w is None else rt.add(acc_w, part)
            if acc_w is not None:
                _acc(grads, w, acc_w)

    out._backward = bw
    # 反传里把 box[0] 置空就会把前向激活还回池；这里只是多留一个引用，防闭包被提前收掉。
    out_buf.keepalive = saved_box
    return out


class MoE(Module):
    """带两种门控的 MoE。专家按 token 分组，空专家不参与 GEMM。

    前向（每个 token）：
        z = x @ router.weight.T
        S = topk(z)                         # 更大的 logit 优先，平局取更小下标
        sigmoid:  y = Σ_{e in S} sigmoid(z_e) * Expert_e(x)
        softmax:  p = softmax(z);  y = Σ_{e in S} p_e * Expert_e(x)
        Expert:   h = silu(x @ gate.T) * (x @ up.T);  out = h @ down.T
    """

    def __init__(self, d_model, d_ff, n_experts, top_k=2, gate="softmax", qb_bias=None,
                 n_shared=0, shared_d_ff=None, balance="none", aux_free_lr=1e-3,
                 z_loss_coef=0.0, capacity_factor=None):
        super().__init__()
        d_model = int(d_model)
        d_ff = int(d_ff)
        n_experts = int(n_experts)
        top_k = int(top_k)
        if d_model < 1 or d_ff < 1:
            raise ValueError("MoE 的 d_model / d_ff 必须为正" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        if n_experts < 1:
            raise ValueError("MoE 至少要 1 个专家" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        if top_k < 1 or top_k > n_experts:
            raise ValueError("top_k=%d 必须落在 [1, %d]" % (top_k, n_experts) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        #: V2.7（quantile balancing）用 "softmax_topk"：**只对选中的 k 个**做 softmax
        #: 依据 tgai_cuda.cu:301-307「softmax 仅对选中的 k 个做」
        if gate not in ("softmax", "sigmoid", "softmax_topk"):
            raise ValueError("gate 只接受 softmax / sigmoid，收到 %r" % (gate,) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = gate
        #: V2.7 的负载均衡偏置 [n_experts]；前向里**加在 top-k 之前**（tgai_engine.c:1120）
        self.qb_bias = None if qb_bias is None else _as_tensor(qb_bias)
        self.router = Linear(d_model, n_experts, bias=False)
        self.experts = _ExpertList([
            _MoEExpert(d_model, d_ff) for _ in range(n_experts)
        ])
        #: **共享专家**（DeepSeekMoE / Llama 4 做法）：所有 token 恒定经过、**不参与路由**，
        #: 输出直接与路由输出**相加**。作用是吸收通用知识，让路由专家专注特化。
        #: 为什么必须有：现代主流 MoE（DeepSeek-V3 1 个、Llama 4 各 1 个）都带共享专家；
        #: 没有它，ORV 训不了这些结构。n_shared=0 时行为与加入该功能前完全一致。
        self.n_shared = int(n_shared)
        if self.n_shared < 0:
            raise ValueError(
                "n_shared 必须 ≥ 0，收到 %d" % self.n_shared +
                "。修法：共享专家数量是自然数；不需要共享专家时用默认值 0")
        self.shared_d_ff = int(shared_d_ff) if shared_d_ff else int(d_ff)
        if self.shared_d_ff < 1:
            raise ValueError(
                "shared_d_ff 必须 ≥ 1，收到 %d" % self.shared_d_ff +
                "。修法：共享专家可以不设（默认等于 d_ff），设了就要给正数")
        self.shared_experts = _ExpertList([
            _MoEExpert(d_model, self.shared_d_ff) for _ in range(self.n_shared)
        ])
        #: 负载均衡方案。
        #: "none"（默认）= 不均衡，与旧行为逐位相同；
        #: "aux_free" = DeepSeek-V3 的免辅助损失方案（arXiv 2408.15664）。
        if balance not in ("none", "aux_free"):
            raise ValueError(
                "balance 只接受 'none' / 'aux_free'，收到 %r" % (balance,) +
                "。修法：需要 V3 的免辅助损失平衡就写 'aux_free'，否则用默认 'none'")
        self.balance = balance
        self.aux_free_lr = float(aux_free_lr)
        #: 每个专家的选择偏置 b_i（**buffer，不是 Parameter**）。
        #: 只参与 top-k 选择，不进门控权重、不进梯度图、不由优化器更新。
        self.expert_bias = np.zeros(n_experts, dtype=np.float32)
        self._last_idx = None
        #: Z-loss 权重（ST-MoE, arXiv:2202.08906）。0 = 关闭（默认，与旧行为一致）。
        #: 开启后每次前向会把 `self.z_loss` 设成一个标量张量，
        #: **调用方需自己把它加到总 loss 上**（ORV 不替你改 loss，避免隐式行为）。
        self.z_loss_coef = float(z_loss_coef)
        self.z_loss = None
        #: 每专家容量因子 α（`C = ceil(α·T·k/N)`）。None = 不限制（默认，与旧行为一致）。
        #: 多卡 Expert Parallel 必须设；单卡可不设。超容量走**残差旁路**。
        if capacity_factor is not None:
            capacity_factor = float(capacity_factor)
            if capacity_factor <= 0:
                raise ValueError(
                    "capacity_factor 必须 > 0，收到 %r" % (capacity_factor,) +
                    "。修法：常用 1.0~1.25；不想限制就用默认 None")
        self.capacity_factor = capacity_factor

    def _add_shared(self, x, out):
        """把共享专家的输出加到路由输出上。

        在 `_forward_flat` 这一层做，所以 **CPU 与 CUDA 两条路径自动同时生效**
        —— 不会出现"一条实现、一条没实现"的数值不一致。
        """
        for e in self.shared_experts:
            out = out + _moe_swiglu(x, e)
        return out

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = _as_tensor(x)
        if x.ndim < 1:
            raise ValueError("MoE 输入至少 1 维" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        if int(x.shape[-1]) != self.d_model:
            raise ValueError(
                "MoE 最后一维 %d != d_model %d" % (int(x.shape[-1]), self.d_model) + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
        dev_x = getattr(x, "device", "cpu")
        dev_w = getattr(self.router.weight, "device", "cpu")
        if dev_x != dev_w:
            raise RuntimeError("MoE 参数在 %s，输入在 %s" % (dev_w, dev_x) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        n = 1
        for s in x.shape[:-1]:
            n *= int(s)
        if n > 2147483647 // self.top_k:
            raise ValueError("MoE token 数过大" + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        flat = x.reshape((n, self.d_model))
        y = flat * 0.0 if n == 0 else self._forward_flat(flat)
        if self.balance == "aux_free" and getattr(self, "training", True) and n > 0:
            # V3 的 bias 更新每步一次、在图外做（不进梯度图）
            self._update_expert_bias(n)
        return y.reshape(tuple(int(s) for s in x.shape))

    def _capacity(self, n):
        """每专家容量 `C = ceil(α · T · k / N)`；未开启返回 None。

        为什么需要：多卡 Expert Parallel 下必须静态分配缓冲，不能让某专家无限收 token。
        单卡其实可以不设；默认 None = 不限制（与旧行为一致）。

        超过容量的 token 在本层**拿不到该专家的贡献**（输出记为 0），
        但它在 block 里的残差 `x` 仍然流过去 —— 这就是 **residual bypass**，
        Switch / Mixtral / DeepSeek 的默认做法：丢的是"专家加工"，不是"整个 token"。
        """
        if self.capacity_factor is None:
            return None
        import math as _math
        cap = _math.ceil(float(self.capacity_factor) * float(n) * float(self.top_k)
                         / float(self.n_experts))
        return _builtin_max(1, int(cap))

    @staticmethod
    def _capacity_mask(idx, n, cap, n_experts, top_k):
        """返回 `[n, k]` 布尔掩码：True = 保留该 (token, 专家) 对。

        规则：**每个专家按 (token, slot) 升序保留前 cap 个**，其余丢弃。

        为什么用掩码而不是截断专家偏移（这是 0.9.40 的修法，已废弃）：

        截断偏移量依赖后端 compact 的顺序，CPU 与 CUDA 一旦不同，
        **丢的就不是同一批 token** —— 0.9.40 实测 CPU 丢 3 个、CUDA 丢 0 个，
        于是开容量限制时 CPU/CUDA 结果不一致。
        掩码只依赖 `idx`（两条路径都从同一份 logits 选出），所以天然一致。

        实现用**稳定排序**：`argsort(kind="stable")` 保证同一专家内部仍是 (t,s) 升序，
        与 CPU 逐 token 扫描的顺序完全相同。
        """
        flat = np.asarray(idx, np.int64).reshape(-1)
        nk = int(flat.size)
        if nk == 0:
            return np.ones((int(n), int(top_k)), bool)
        keep = np.ones(nk, bool)
        order = np.argsort(flat, kind="stable")
        counts = np.bincount(flat, minlength=int(n_experts))
        start = 0
        for c in counts:
            c = int(c)
            if c > cap:
                # 该专家超容量：丢掉排序后靠后的 c - cap 个
                keep[order[start + cap:start + c]] = False
            start += c
        return keep.reshape(int(n), int(top_k))

    def _router_z_loss(self, logits):
        """ST-MoE 的 Z-loss：`β · Σ_t (logsumexp(s_t))²`，压制 router logit 幅值。

        为什么要它：fp16 下 router logits 容易溢出（早期训练 router 权重摆动大），
        溢出会让 softmax 变成 one-hot 甚至 nan。Z-loss 惩罚 logit 的绝对大小，
        是 MoE 训练的数值稳定手段（arXiv:2202.08906）。

        实现技巧：`log_softmax(x)_j = x_j − lse`，所以 `x_j − log_softmax_j === lse`
        **对每一列都成立**。取均值就得到 lse，不必新写 logsumexp 内核，
        也不会因为 E 的大小而损失精度。
        """
        E = int(logits.shape[-1])
        lp = functional.log_softmax(logits, axis=-1)
        lse = (logits - lp).sum(axis=-1) / float(E)   # 每列都等于 lse
        return self.z_loss_coef * (lse * lse).sum()

    def _forward_flat(self, x):
        logits = x @ self.router.weight.T
        if getattr(self, "qb_bias", None) is not None:
            # 依据 tgai_engine.c:1120「router logits 先加本偏置，再做 top-k」
            logits = logits + self.qb_bias
        #: Z-loss（若开启）；调用方把它加到总 loss 上即可。None 表示未开启。
        self.z_loss = self._router_z_loss(logits) if self.z_loss_coef else None
        sel_logits = None
        if self.balance == "aux_free":
            # ★ aux-loss-free（DeepSeek-V3, arXiv 2408.15664）：
            #   sel_logits = logits + b_i 只用于**选专家**；
            #   门控权重仍用原始 logits（由 _combine_* 保证），b_i 不进梯度图。
            sel_logits = logits + _as_tensor(self.expert_bias)
        if getattr(x, "device", "cpu") == "cuda":
            out = self._combine_cuda(x, logits, sel_logits)
        else:
            out = self._combine_cpu(x, logits, sel_logits)
        return self._add_shared(x, out)

    def _update_expert_bias(self, n):
        """按 DeepSeek-V3 的符号更新调整 b_i（**不进梯度图、不由优化器更新**）。

        b_i ← b_i − u·sign(c_i − c̄_i)，c_i 是本步该专家收到的 token 数，
        c̄_i = T·k/N 是均匀分布的期望。过载则降 bias，欠载则升 bias。

        用 `_last_idx`（combine 里**已经算出来的**那份索引），不重算 router ——
        既省一次 GEMM，也保证"统计的确实是被执行的那次选择"。
        """
        idx = getattr(self, "_last_idx", None)
        if idx is None or n <= 0:
            return
        c = np.bincount(np.asarray(idx, np.int64).ravel(),
                        minlength=self.n_experts).astype(np.float64)
        c_bar = float(n * self.top_k) / float(self.n_experts)
        self.expert_bias = (self.expert_bias -
                            self.aux_free_lr * np.sign(c - c_bar)).astype(np.float32)

    def _weights_cpu(self, logits, idx):
        n, k = int(idx.shape[0]), int(idx.shape[1])
        rows = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None], (n, k)).copy()
        cols = np.asarray(idx, dtype=np.int64)
        if self.gate == "sigmoid":
            return _sigmoid_tensor(logits[rows, cols])
        if self.gate == "softmax_topk":
            # 只对选中的 k 个归一化（与产线一致）
            return functional.softmax(logits[rows, cols], axis=-1)
        return functional.softmax(logits, axis=-1)[rows, cols]

    def _combine_cpu(self, x, logits, sel_logits=None):
        n = int(x.shape[0])
        d = int(x.shape[1])
        # ★ aux-loss-free：**用 sel_logits 选专家，用 logits 算门控权重**。
        # 两者在普通模式下是同一个张量；aux-free 时 sel_logits = logits + bias，
        # bias 只影响"选谁"，绝不进门控权重，否则会污染主梯度。
        sel = logits if sel_logits is None else sel_logits
        idx = _moe_topk_idx(sel.data, self.top_k)
        self._last_idx = np.asarray(idx, np.int64)   # 供 aux-loss-free 统计负载，不重算
        cap = self._capacity(n)
        w = self._weights_cpu(logits, idx)
        if cap is not None:
            # 与 CUDA 路径**同一套掩码规则**（同一份 idx、同样的稳定排序），
            # 所以两条路径丢的是同一批 (token, 专家) 对。
            keep = self._capacity_mask(idx, n, cap, self.n_experts, self.top_k)
            w = w * _as_tensor(keep.astype(np.float32))
        acc = None
        for e, expert in enumerate(self.experts):
            tok = []
            slot = []
            for t in range(n):
                row = idx[t]
                for s in range(self.top_k):
                    if int(row[s]) == e:
                        tok.append(t)
                        slot.append(s)
            if not tok:
                continue
            # 注意：容量限制已由上面的**权重掩码**处理，这里**不能**再截断 tok/slot，
            # 否则会重复丢弃（两套规则叠加会让 CPU 与 CUDA 再次分叉）。
            tok_a = np.asarray(tok, dtype=np.int64)
            slot_a = np.asarray(slot, dtype=np.int64)
            y = _moe_swiglu(x[tok_a], expert)
            y = y * w[tok_a, slot_a].reshape((-1, 1))
            part = _moe_scatter_cpu(n, d, tok_a, y)
            acc = part if acc is None else acc + part
        if acc is None:
            return x * 0.0
        return acc

    def _combine_cuda(self, x, logits, sel_logits=None):
        rt = _cuda()
        n = int(x.shape[0])
        d = int(x.shape[1])
        k = self.top_k
        n_exp = self.n_experts
        # ★ aux-loss-free：idx 从 sel_logits（含 bias）选，**权重值必须从原始 logits 取**。
        # 所以这里只借 sel_logits 拿 idx；val_buf 来自它，只在不带 bias 时才可当权重用。
        aux_free = sel_logits is not None
        if aux_free:
            idx_buf, _ = rt.moe_topk(sel_logits._gpu, k)
            val_buf = rt.moe_gather_pairs(logits._gpu, idx_buf, n, n_exp, k)
        else:
            idx_buf, val_buf = rt.moe_topk(logits._gpu, k)
        # 供 aux-loss-free 统计负载（只在需要时拉回主机）
        if self.balance == "aux_free":
            self._last_idx = np.asarray(rt.d2h(idx_buf), np.int64).reshape(n, k)
        if self.gate == "sigmoid":
            sel = Tensor._from_gpu(
                val_buf, requires_grad=logits.requires_grad, _parents=(logits,))

            def bw_sel(g, grads, _idx=idx_buf, _rt=rt, _logits=logits):
                if _logits.requires_grad:
                    _acc(grads, _logits, _rt.moe_scatter_pairs(g, _idx, n, n_exp, k))

            sel._backward = bw_sel
            w = _sigmoid_tensor(sel)
        elif self.gate == "softmax_topk":
            # ★ 修复（2026-11-09 实测定位）：这里原先是
            #     w = softmax(logits 全部 n_exp 个) 再取 top-k
            #   而语义（CPU 路径 / 产线 tgai_cuda.cu:301-307 / PyTorch 参考）都是
            #   **只在选中的 k 个上做 softmax**。两者仅当某个专家占优时数值接近，
            #   一旦 logits 幅值变大就分叉（实测 scale=8 差 4.55e-02）。
            #   现在直接用 moe_topk 返回的选中值 val_buf 做 softmax（形状 [n,k]）。
            sel = Tensor._from_gpu(
                val_buf, requires_grad=logits.requires_grad, _parents=(logits,))

            def bw_sel_topk(g, grads, _idx=idx_buf, _rt=rt, _logits=logits):
                if _logits.requires_grad:
                    _acc(grads, _logits, _rt.moe_scatter_pairs(g, _idx, n, n_exp, k))

            sel._backward = bw_sel_topk
            w = functional.softmax(sel, axis=-1)
        else:
            w = _moe_take_cuda(functional.softmax(logits, axis=-1), idx_buf, n, n_exp, k)
        off_buf, tok_buf, slot_buf = rt.moe_compact(idx_buf, n, n_exp, k)
        # 只有 E+1 个计数回到主机，用来跳过空专家。激活留在设备上。
        off = np.asarray(rt.d2h(off_buf), dtype=np.int32).reshape(-1)
        cap = self._capacity(n)
        if cap is not None:
            # 容量限制用**门控权重掩码**实现（见 _capacity_mask 的说明）：
            # 把超容量的 (token, 专家) 对的权重置 0，等价于该对不参与路由，
            # 于是被丢的 token 自然走残差旁路。只依赖 idx ⇒ CPU/CUDA 丢同一批。
            idx_host = np.asarray(rt.d2h(idx_buf), np.int64).reshape(n, k)
            keep = self._capacity_mask(idx_host, n, cap, n_exp, k)
            mask_t = _as_tensor(keep.astype(np.float32))
            w = w * mask_t
        grouped = _moe_grouped_cuda(
            x, w, tok_buf, slot_buf, off, self.experts, n, d, k)
        if grouped is not None:
            return grouped
        acc = None
        for e, expert in enumerate(self.experts):
            begin = int(off[e])
            count = int(off[e + 1]) - begin
            if count <= 0:
                continue
            rows = _moe_gather_rows_cuda(x, tok_buf, begin, count, d)
            y = _moe_swiglu(rows, expert)
            ww = _moe_gather_slots_cuda(w, tok_buf, slot_buf, begin, count, k)
            y = _moe_row_scale_cuda(y, ww)
            part = _moe_scatter_rows_cuda(n, d, tok_buf, begin, count, y)
            acc = part if acc is None else acc + part
        if acc is None:
            return x * 0.0
        return acc


# ─────────────────────────── optim ───────────────────────────
def _bf16_step_boundary():
    """优化器每步开始：通知后端让 bf16 影子缓存失效（权重马上要变）。

    只有支持 bf16 的影子缓存的后端才有这个方法；没有就是空操作，
    对 fp32 路径零影响（一次函数调用 + 一次 getattr）。
    """
    try:
        rt = _cuda()
    except Exception:
        return
    f = getattr(rt, "_bf16_epoch_bump", None)
    if f is not None:
        f()


class _Optim:
    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr = lr
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


def _invalidate_fp16_after_step(fn):
    """装饰器：step() 执行后让 fp16 权重影子失效。"""
    def wrapped(self, *a, **kw):
        out = fn(self, *a, **kw)
        try:
            fp16_shadows_invalidate()
        except Exception:
            pass
        try:                       # AMP 的权重 fp16 影子也必须跟着失效
            from tgai_go.train import amp_mlp as _amp
            _amp.invalidate()
        except Exception:
            pass
        return out
    return wrapped


class LAMB(_Optim):
    """LAMB：Adam 的逐层自适应更新 + trust ratio（`||p|| / ||u||`）。

    与 AdamW 的唯一区别是**更新量的缩放方式**：
        p <- p - lr * trust * u,      trust = ||p|| / ||u||
    所以每一层的相对步长都约等于 lr，对**层间尺度差异大**的网络（大 batch 训练）
    比 AdamW 稳。

    目前**仅 CPU**：CUDA 张量会明确报错（修法见消息），不做静默退化。
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m, self.v = [], []
        for p in self.params:
            if getattr(p, "device", "cpu") == "cuda":
                _shape = getattr(p, "shape", None)
                _shape = tuple(int(x) for x in _shape) if _shape else "(未知形状)"
                raise NotImplementedError(
                    "LAMB 目前只支持 CPU 张量（参数 %s 在 cuda 上）。"
                    "修法：① 先用 AdamW（CUDA 已支持）；② 或把参数搬到 CPU；"
                    "③ 要真正在 GPU 上用 LAMB，需要照 AdamW 的 CUDA 分支补一套内核调用"
                    % (_shape,))
            self.m.append(np.zeros_like(p.data))
            self.v.append(np.zeros_like(p.data))

    @_invalidate_fp16_after_step
    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = np.asarray(p.grad.to_numpy() if hasattr(p.grad, "to_numpy") else p.grad,
                           np.float64)
            m, v = self.m[i], self.v[i]
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * (g * g)
            mhat = m / bc1
            vhat = v / bc2
            u = mhat / (np.sqrt(vhat) + self.eps)
            if self.wd:
                u = u + self.wd * np.asarray(p.data, np.float64)
            p_norm = float(np.linalg.norm(np.asarray(p.data, np.float64)))
            u_norm = float(np.linalg.norm(u))
            # trust ratio：任一范数为 0 时按 1 处理（与论文的 "selective" 一致地保守）
            trust = (p_norm / u_norm) if (p_norm > 0.0 and u_norm > 0.0) else 1.0
            p.data -= (self.lr * trust * u).astype(np.float32)
        return None


class AdamW(_Optim):
    """与 torch.optim.AdamW 语义一致（含 decoupled weight decay）。"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = []
        self.v = []
        for p in self.params:
            if getattr(p, "device", "cpu") == "cuda":
                # 值在 _gpu 上，p.data 是 None。动量也放设备上，别 zeros_like(None)。
                rt = _cuda()
                self.m.append(rt.zeros(tuple(int(s) for s in p.shape)))
                self.v.append(rt.zeros(tuple(int(s) for s in p.shape)))
            else:
                self.m.append(np.zeros_like(p.data))
                self.v.append(np.zeros_like(p.data))

    @_invalidate_fp16_after_step
    def step(self):
        _bf16_step_boundary()
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        # 这些标量和参数无关。按 numpy 的标量提升收成 float32，再整步留在 GPU。
        b1 = _f32(self.b1)
        omb1 = _f32(1.0 - self.b1)
        b2 = _f32(self.b2)
        omb2 = _f32(1.0 - self.b2)
        lr = _f32(self.lr)
        wd_lr = _f32(self.lr * self.wd) if self.wd else 0.0
        eps = _f32(self.eps)
        bc1s = _f32(bc1)
        bc2s = _f32(bc2)
        do_wd = bool(self.wd)
        # ★ fused 路径：把所有 CUDA 参数收进一个 kernel（290 -> 1）。
        #   逐元素算术与单张量内核逐条相同，所以数值逐位一致。
        #   只要有一个参数不是 CUDA/梯度类型不对，就整体退回逐参数循环（行为不变）。
        # 只有确实存在 CUDA 参数时才去取运行时（纯 CPU 环境不能碰 _cuda()）
        _has_cuda = any(getattr(p, "device", "cpu") == "cuda" for p in self.params)
        _rt = None
        if _has_cuda:
            try:
                _rt = _cuda()
            except Exception:
                _rt = None
        if _rt is not None and getattr(_rt, "fused_adamw", True):
            ents = []
            fused_ok = True
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                if getattr(p, "device", "cpu") != "cuda":
                    fused_ok = False
                    break
                g = p.grad
                if not isinstance(g, Tensor) or getattr(g, "device", "cpu") != "cuda":
                    fused_ok = False
                    break
                ents.append((p._gpu, g._gpu, self.m[i], self.v[i]))
            if fused_ok and ents:
                _rt.adamw_multi(ents, b1, omb1, b2, omb2, lr, wd_lr, eps,
                                bc1s, bc2s, do_wd)
                return
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if getattr(p, "device", "cpu") == "cuda":
                # 崩在 float * p.data：data 是 None，不是梯度缺失。
                # Linear 的 weight 参加了前向，grad 是 CUDA Tensor。
                # PyTorch 的 AdamW 跳过 grad is None；这里同样跳过，更新本身在 GPU 上做。
                g = p.grad
                if not isinstance(g, Tensor) or getattr(g, "device", "cpu") != "cuda":
                    raise TypeError("CUDA AdamW 需要 CUDA Tensor 梯度，收到 %s" % type(g).__name__ + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
                _cuda().adamw(
                    p._gpu, g._gpu, self.m[i], self.v[i],
                    b1, omb1, b2, omb2, lr, wd_lr, eps, bc1s, bc2s, do_wd,
                )
                continue
            g = p.grad
            if self.wd:
                p.data = p.data - self.lr * self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / bc1
            vh = self.v[i] / bc2
            p.data = p.data - self.lr * mh / (np.sqrt(vh) + self.eps)


def _sumsq32(x):
    """按地址顺序的 float32 平方和，和 GPU sumsq_sequential 同一套舍入。"""
    acc = np.float32(0.0)
    flat = np.ascontiguousarray(x, dtype=np.float32).ravel()
    for v in flat:
        vv = np.float32(v)
        acc = np.float32(acc + np.float32(vv * vv))
    return acc


def _newton_schulz5(g, steps=5, eps=1e-7):
    """Muon 的正交化迭代（a,b,c = 3.4445, -4.7750, 2.0315）。

    注意：该多项式的不动点是 0.868，所以输出奇异值集中在 0.868 附近、
    条件数被压到 ~1.6（实测病态输入 384 -> 1.61）。
    它的作用是**压平谱**，常数倍由 Muon 的学习率吸收。

    ||X|| 用顺序 float32 平方和，和 GPU 上只降下来的那个标量同一套舍入。
    np.linalg.norm 在 n=64 时和这个差约 1 ulp，Newton-Schulz 会把它放大过 1e-7。
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = np.array(g, dtype=np.float32, copy=True)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = np.ascontiguousarray(x.T)
        transposed = True
    nrm = np.sqrt(_sumsq32(x))
    denom = np.float32(nrm + np.float32(eps))
    x = np.float32(x / denom)
    for _ in range(steps):
        s = x @ x.T
        y = b * s + c * (s @ s)
        x = a * x + y @ x
    if transposed:
        x = np.ascontiguousarray(x.T)
    return np.asarray(x, np.float32)


def _newton_schulz5_gpu(rt, x, steps=5, eps=1e-7, fast=False):
    """与 _newton_schulz5 同一公式，值留在显存。

    `fast=False`（默认）：X@X.T 等走设备上的**顺序 FMA** 内核（`sgemm_mode` 1）。
    这与 NumPy 1.26 / OpenBLAS 的 float32 `@` 在这些形状上逐位相同。
    cublasSgemm_v2 单次有约 3e-8 的相对误差，五轮迭代后参数对不上 1e-7，所以默认不用 cuBLAS。

    `fast=True`：改用 cuBLAS（`gemm_nt`/`gemm_nn`）。**放弃与 OpenBLAS 的逐位一致**，
    换来数量级的速度提升 —— 3B 模型有 2550 个张量要逐个正交化，顺序 FMA 实测 62.9 秒/步，
    完全不可用（见 docs 实测）。代价是每步多约 1e-7 量级的差异，属 fp32 正常范围。
    """
    transposed = False
    if int(x.shape[0]) > int(x.shape[1]):
        x = rt.transpose2d(x)
        transposed = True
    # ★ 2026-11-11 实测定位：这一行曾是 Muon 35 s/步的**唯一主因**。
    #   `sumsq_sequential` 是**单线程顺序归约**（当年为了与 OpenBLAS 逐位一致），
    #   实测单个 [1024,4096] 矩阵就要 **101 ms**（含 read_scalar 同步），
    #   而 5 步 NS 的纯 GEMM 只要 9.3 ms —— 91% 的时间花在这里，不是算力问题。
    #   `fast=True` 本来就已放弃逐位一致（cuBLAS 单次 ~3e-8），
    #   所以这里同步改用两趟**并行**归约 `sumsq_parallel`。
    #   顺序版保留给 `fast=False`（需要与 numpy 逐位对齐的场合）。
    sumsq = rt.read_scalar(
        rt.sumsq_parallel(x) if fast else rt.sumsq_sequential(x))
    nrm = np.sqrt(np.float32(sumsq))
    denom = np.float32(nrm + np.float32(eps))
    x = rt.div_scalar(x, float(denom))
    a = _f32(3.4445)
    b = _f32(-4.7750)
    c = _f32(2.0315)
    if fast:
        # cuBLAS 路径：同样的 5 步迭代，只是矩阵乘换成 cuBLAS（运行时只有 gemm_nt）。
        # 利用对称性：s = x x^T 与 ss = s s 都是对称矩阵，所以 s@s == s@s^T；
        # y@x 则用 gemm_nt(y, x^T) —— 多一次 transpose 拷贝，远小于 GEMM 本身。
        for _ in range(int(steps)):
            s = rt.gemm_nt(x, x)                     # x @ x^T（对称）
            ss = rt.gemm_nt(s, s)                    # s @ s^T == s @ s（对称）
            y = rt.add(rt.scale(s, b), rt.scale(ss, c))
            x = rt.add(rt.scale(x, a), rt.gemm_nt(y, rt.transpose2d(x)))
    else:
        for _ in range(int(steps)):
            # X @ X.T：先把 X 转成连续的 X.T，再做顺序 FMA 的普通乘。
            s = rt.sgemm_mode(x, rt.transpose2d(x), 1)
            ss = rt.sgemm_mode(s, s, 1)
            y = rt.add(rt.scale(s, b), rt.scale(ss, c))
            x = rt.add(rt.scale(x, a), rt.sgemm_mode(y, x, 1))
    if transposed:
        x = rt.transpose2d(x)
    return x


class Muon(_Optim):
    """Muon：动量 + Newton-Schulz 正交化；2D 走 Muon，其余回退 AdamW 式更新。

    ## 显存（0.9.28 修正，这是 3B MoE 能不能跑的关键）

    旧实现给**每个参数**都分配 `buf + _adam_m + _adam_v` 三份 fp32 状态
    ⇒ 2.97B 模型要 3×11.1 = **33.3 GiB**，比 AdamW 的 22.2 还差，完全不可用。

    现在按形状只分配真正用得上的：

    | 参数形状 | 走哪条路 | 状态 |
    | --- | --- | --- |
    | 2D（投影矩阵、专家矩阵、embedding） | Muon | 只有 1 份动量 `buf` |
    | 其它（1D 的 RMSNorm 等、标量） | AdamW | m + v |

    `fp16_mom=True` 再把 `buf` 存成 fp16 ⇒ 2.97B 只需要 **5.6 GiB**（vs AdamW fp16 的 11.1 GiB）。
    这正是 Muon 对 MoE 的价值：**一份状态而不是两份**。
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0,
                 fp16_mom=False, ns_fast=False, ns_steps=5):
        _Optim.__init__(self, params, lr)
        #: Newton-Schulz 是否用 cuBLAS（`ns_fast=True`）。
        #: 默认 False = 顺序 FMA 内核，与 OpenBLAS 逐位一致但**很慢**
        #: （3B 有 2550 个张量要逐个正交化，实测 62.9 秒/步）。
        #: 大模型可设 ns_fast=True：实测 62.9 → 35.0 s/步（1.8×），代价是约 1e-7 量级差异。
        #:
        #: ★ 2026-11-11 实测结论：**Muon 在 3B 规模不可用**，与本开关无关。
        #: Newton-Schulz 本身是 FLOP 密集型 —— 每个 [1024,4096] 矩阵 5 步 × 3 GEMM ≈ 96 GFLOP，
        #: 224 个这样的矩阵合计约 21.6 TFLOP/步；V100 fp32 峰值 15.7 TFLOP/s，
        #: 理论下限已约 1.4 秒，实测 35 秒（受带宽限制）。这是**算法固有代价**，不是实现缺陷。
        #: 同卡上 AdamW 只要 0.709 s/步。要训 3B 请用 AdamW。
        self.ns_fast = bool(ns_fast)
        self.ns_steps = int(ns_steps)
        self.mom = momentum
        self.nesterov = nesterov
        self.wd = weight_decay
        self.fp16_mom = bool(fp16_mom)
        self.buf = []          # 仅 2D 参数：动量
        self._adam_m = []      # 仅非 2D 参数
        self._adam_v = []
        self._is2d = []
        rt = None
        for p in self.params:
            shp = tuple(int(s) for s in p.shape)
            two_d = len(shp) == 2
            self._is2d.append(two_d)
            if getattr(p, "device", "cpu") == "cuda":
                if rt is None:
                    rt = _cuda()
                if two_d:
                    self.buf.append(rt.zeros(shp, dtype=np.uint16) if self.fp16_mom
                                    else rt.zeros(shp))
                    self._adam_m.append(None)
                    self._adam_v.append(None)
                else:
                    self.buf.append(None)
                    self._adam_m.append(rt.zeros(shp))
                    self._adam_v.append(rt.zeros(shp))
            else:
                if two_d:
                    self.buf.append(np.zeros(shp, np.float16 if self.fp16_mom else np.float32))
                    self._adam_m.append(None)
                    self._adam_v.append(None)
                else:
                    self.buf.append(None)
                    self._adam_m.append(np.zeros(shp, np.float32))
                    self._adam_v.append(np.zeros(shp, np.float32))
        self.t = 0

    # ---------- 状态占用（给"能不能装下"这类问题一个可靠答案） ----------
    def bytes_per_param(self):
        """2D 参数每元素字节数（fp16 动量 = 2，fp32 = 4）；非 2D 是 Adam 的 8。"""
        return 2.0 if self.fp16_mom else 4.0

    def state_bytes(self):
        tot = 0.0
        for i, p in enumerate(self.params):
            n = int(p.numel())
            if self._is2d[i]:
                tot += n * (2.0 if self.fp16_mom else 4.0)
            else:
                tot += n * 8.0
        return tot

    # ---------- 分层即时更新（与 Fp16MomAdamW 同一套语义） ----------
    _groups = None

    def set_layer_groups(self, groups, require_full=True, auto_rest=True):
        """登记"哪些参数属于哪一层"。语义与 `Fp16MomAdamW.set_layer_groups` 一致。"""
        first = {}
        for i, q in enumerate(self.params):
            first.setdefault(id(q), i)
        self._groups = {}
        for k, params in dict(groups or {}).items():
            idx = []
            for p in params:
                j = first.get(id(p))
                if j is not None and j not in idx:
                    idx.append(j)
            self._groups[int(k)] = idx
        covered = set()
        for idx in self._groups.values():
            covered.update(idx)
        missing = [i for i in sorted(set(first.values())) if i not in covered]
        if missing and auto_rest:
            self._groups.setdefault(-1, [])
            self._groups[-1] = sorted(set(self._groups[-1]) | set(missing))
            _sys_mod.stderr.write(
                "[tgai] Muon 分组未覆盖 %d 个参数，已自动放进 -1 组。\n" % len(missing))
            _sys_mod.stderr.flush()
            missing = []
        if require_full and missing:
            raise ValueError("Muon 分层分组没有覆盖全部参数：%d 个未覆盖" % len(missing) +
                             "。修法：打开 auto_rest 或把 embed/norm/head 单独成组")
        return self

    def all_groups(self):
        return sorted(self._groups.keys())

    def step_layers(self, layer_ids, t=None, release=True):
        """只更新这些层并释放它们的梯度（数学上等价于统一更新，见 opt_fp16mv 的说明）。"""
        if not self._groups:
            raise RuntimeError("还没登记分层参数" + "。修法：先 set_layer_groups(...)")
        tt = int(self.t + 1) if t is None else int(t)
        for lid in layer_ids:
            for i in self._groups.get(int(lid), []):
                p = self.params[i]
                if p.grad is None or not getattr(p, "requires_grad", True):
                    continue
                self._step_one(i, p, tt)
        if release:
            for lid in layer_ids:
                for i in self._groups.get(int(lid), []):
                    self.params[i].grad = None
        return self

    def _step_one(self, i, p, tt):
        """按给定步数 `tt` 更新单个参数（CPU/CUDA 自动分派，供 step 与 step_layers 复用）。"""
        old = self.t
        self.t = tt
        try:
            if getattr(p, "device", "cpu") == "cuda":
                self._step_cuda(i, p)
            else:
                self._step_cpu(i, p)
        finally:
            self.t = old

    def _step_cpu(self, i, p):
        """CPU 版单参数更新（与 `step()` 里的逻辑逐行相同，抽出来供分层更新复用）。"""
        g = p.grad
        # CPU 上 backward 后 .grad 是 numpy；若调用方硬塞了 Tensor，这里统一取成 numpy，
        # 否则 _newton_schulz5 会在 Tensor 上 float() 失败（实测就是这么炸的）。
        if hasattr(g, "to_numpy"):
            g = np.asarray(g.to_numpy(), np.float32)
        if p.data.ndim == 2:
            if self.wd:
                p.data = p.data - self.lr * self.wd * p.data
            self.buf[i] = self.mom * self.buf[i] + g
            u = g + self.mom * self.buf[i] if self.nesterov else self.buf[i]
            o = _newton_schulz5(np.asarray(u, np.float32))
            scale = np.sqrt(_builtin_max(1.0, p.data.shape[0] / p.data.shape[1]))
            p.data = p.data - self.lr * scale * o
        else:
            b1, b2 = 0.9, 0.999
            self._adam_m[i] = b1 * self._adam_m[i] + (1 - b1) * g
            self._adam_v[i] = b2 * self._adam_v[i] + (1 - b2) * (g * g)
            mh = self._adam_m[i] / (1 - b1 ** self.t)
            vh = self._adam_v[i] / (1 - b2 ** self.t)
            p.data = p.data - self.lr * mh / (np.sqrt(vh) + 1e-8)

    def _step_cuda(self, i, p):
        rt = _cuda()
        g = p.grad
        if not isinstance(g, Tensor) or getattr(g, "device", "cpu") != "cuda":
            raise TypeError("CUDA Muon 需要 CUDA Tensor 梯度，收到 %s" % type(g).__name__ + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
        if len(p.shape) != 2:
            b1, b2, eps = 0.9, 0.999, 1e-8
            bc1 = 1.0 - b1 ** self.t
            bc2 = 1.0 - b2 ** self.t
            rt.adamw(
                p._gpu, g._gpu, self._adam_m[i], self._adam_v[i],
                _f32(b1), _f32(1.0 - b1), _f32(b2), _f32(1.0 - b2),
                _f32(self.lr), 0.0, _f32(eps), _f32(bc1), _f32(bc2), False,
            )
            return
        rows = int(p.shape[0])
        cols = int(p.shape[1])
        u = rt.empty(p.shape, dtype=np.float32)
        # fp16 动量：内核要 fp32 buffer，所以先 cast 出来、用完再写回 fp16。
        # 瞬时只多一个张量的 fp32（最大约 100 MB），换来 3B 上省 5.5 GiB。
        buf = self.buf[i]
        if self.fp16_mom:
            buf32 = rt.cast_f16_to_f32(buf)
        else:
            buf32 = buf
        rt.muon_momentum(
            p._gpu, g._gpu, buf32, u,
            _f32(self.mom),
            _f32(self.lr * self.wd) if self.wd else 0.0,
            bool(self.wd), bool(self.nesterov),
        )
        if self.fp16_mom:
            rt.cast_f32_to_f16_into(buf32, buf)
        o = _newton_schulz5_gpu(rt, u, steps=self.ns_steps, fast=self.ns_fast)
        scale = float(np.sqrt(_builtin_max(1.0, rows / cols)))
        rt.sub_scale_(p._gpu, o, _f32(self.lr * scale))

    @_invalidate_fp16_after_step
    def step(self):
        _bf16_step_boundary()
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self._step_one(i, p, self.t)


class _OptimModule:
    AdamW = AdamW
    LAMB = LAMB
    Muon = Muon

    def __getattr__(self, name):
        # 插件登记的优化器也要能从这里取到（如 plugin.optimizer("Lion2", ...)）
        if name.startswith("_"):
            raise AttributeError(name)
        cls = _plugin_mod.resolve_optimizer(name)
        if cls is None:
            raise AttributeError(
                "optim 里没有 %r" % name +
                "。修法：核心自带 AdamW/SGD/Adafactor/LAMB/Lion/Muon；"
                "其它用 plugin.optimizer(name, cls) 登记")
        return cls


class _NNModule:
    Module = Module
    CausalSelfAttention = CausalSelfAttention
    MoE = MoE
    Parameter = Parameter
    Linear = Linear
    Embedding = Embedding
    RMSNorm = RMSNorm
    Sequential = Sequential
    ModuleList = ModuleList
    functional = functional

    def __getattr__(self, name):
        # 插件登记的层也要能从这里取到（如 plugin.layer("MyBlock", cls)）
        if name.startswith("_"):
            raise AttributeError(name)
        cls = _plugin_mod.resolve_layer(name)
        if cls is None:
            raise AttributeError(
                "nn 里没有 %r" % name +
                "。修法：核心层见 docs/OP_COVERAGE.md；"
                "自定义层用 plugin.layer(name, cls) 登记")
        return cls


nn = _NNModule()
optim = _OptimModule()

# ── 插件系统（六条轴，一个命名空间）────────────────────────────────────────
# 必须先于 nn/optim 的 __getattr__ 使用，所以在这里就导入成模块级名字。
from . import plugin as _plugin_mod
from . import ops_plugin as _ops_plugin


def build(name, **kwargs):
    """按插件登记的骨架名构建模型：`T.build("deepseek_v3", n_routed=256, ...)`。

    骨架由 `plugin.arch(name, builder)` 登记 —— 核心不背模型动物园。
    """
    builder = _plugin_mod.resolve_arch(name)
    if builder is None:
        have = _plugin_mod.registered()["archs"]
        raise KeyError(
            "没有登记名为 %r 的骨架" % name +
            "。修法：用 plugin.arch(%r, 构造函数) 登记；已登记的有 %s" % (name, have or "（无）"))
    return builder(**kwargs)


def plugins():
    """列出当前插件登记状态（算子/后端/优化器/层/骨架/API 兼容层）。"""
    return _plugin_mod.registered()


# ── 共享存储参数（0.9.43）────────────────────────────────────────────────
# 放在这里导入：param_group 依赖 plugin（require_inplace）与本模块的 Parameter，
# 此时二者都已定义。
from .param_group import ParameterGroup, SharedStorageError

# ── 补齐 torch 常用优化器与学习率调度器（0.8.0）────────────────────────────
# 放在文件末尾导入：opt_extra / sched 依赖本模块的 _Optim/_cuda 等，此时已定义完毕。
from .opt_extra import (SGD as _SGDImpl, Lion as _LionImpl,        # noqa: E402
                       Adafactor as _AdafactorImpl,
                       Accumulator as _Accumulator, scale_grads as _scale_grads)
from .nn_ops import clip_grad_norm_ as _clip_grad_norm_impl         # noqa: E402
from . import align as _align_impl                                  # noqa: E402
from . import indexing as _indexing_impl                            # noqa: E402
from . import sched as _sched_impl                                 # noqa: E402


class _LRSchedulerModule:
    """对应 torch.optim.lr_scheduler。"""
    ConstantLR = _sched_impl.ConstantLR
    LinearLR = _sched_impl.LinearLR
    CosineLR = _sched_impl.CosineLR
    StepLR = _sched_impl.StepLR
    LambdaLR = _sched_impl.LambdaLR
    make_scheduler = staticmethod(_sched_impl.make_scheduler)


_OptimModule.SGD = _SGDImpl
_OptimModule.Lion = _LionImpl
_OptimModule.lr_scheduler = _LRSchedulerModule()
_OptimModule.Accumulator = _Accumulator
_OptimModule.scale_grads = staticmethod(_scale_grads)
_OptimModule.Adafactor = _AdafactorImpl
optim.SGD = _SGDImpl
optim.Lion = _LionImpl
optim.Adafactor = _AdafactorImpl
optim.Accumulator = _Accumulator
optim.scale_grads = staticmethod(_scale_grads)
optim.lr_scheduler = _OptimModule.lr_scheduler


class _NNUtils:
    """对应 torch.nn.utils。"""
    clip_grad_norm_ = staticmethod(_clip_grad_norm_impl)
    checkpoint = staticmethod(checkpoint)


_NNModule.utils = _NNUtils()
nn.utils = _NNModule.utils
clip_grad_norm_ = _clip_grad_norm_impl
align = _align_impl
logsigmoid = _align_impl.logsigmoid
seq_logprob = _align_impl.seq_logprob
dpo_pair_loss = _align_impl.dpo_pair_loss
dpo_batch_loss = _align_impl.dpo_batch_loss
index_select = _indexing_impl.index_select
gather_flat = _indexing_impl.gather_flat
scatter_flat = _indexing_impl.scatter_flat
from . import conv as _conv_impl                                       # noqa: E402
conv2d = _conv_impl.conv2d
conv1d = _conv_impl.conv1d
Conv2d = _conv_impl.Conv2d
Conv1d = _conv_impl.Conv1d
from . import loader as _loader_impl                                 # noqa: E402
# 注意：data 已经是「数据规范化」模块，这里把加载 API **挂到同一个模块对象上**，
# 绝不能覆盖它（之前用 SimpleNamespace 覆盖过一次，直接打挂了 12 个数据层测试）。
from . import data as _data_mod                                     # noqa: E402
for _n in ("Dataset", "IterableDataset", "Sampler", "SequentialSampler",
           "RandomSampler", "BatchSampler", "default_collate", "DataLoader"):
    setattr(_data_mod, _n, getattr(_loader_impl, _n))
DataLoader = _loader_impl.DataLoader
Dataset = _loader_impl.Dataset
IterableDataset = _loader_impl.IterableDataset
from . import interop as _interop_impl                             # noqa: E402
from_torch = _interop_impl.from_torch
to_torch = _interop_impl.to_torch
state_dict_from_torch = _interop_impl.state_dict_from_torch
state_dict_to_torch = _interop_impl.state_dict_to_torch
OrvSequential = _interop_impl.OrvSequential
InteropError = _interop_impl.InteropError
from . import norms as _norms_impl                                    # noqa: E402
LayerNorm = _norms_impl.LayerNorm
_NNModule.LayerNorm = _norms_impl.LayerNorm
group_advantages = _align_impl.group_advantages
token_logprobs = _align_impl.token_logprobs
grpo_loss = _align_impl.grpo_loss

def permute(input, dims):
    """对应 torch.permute(input, dims)。dims 是轴的排列。"""
    if not isinstance(dims, (tuple, list)):
        raise TypeError("permute 的 dims 需要 tuple 或 list，收到 %s" % type(dims).__name__ + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    return _as_tensor(input).permute(*dims)


def _repeat_interleave_impl(input, repeats, dim=None):
    """整数 repeats：沿 dim 把每个元素连续复制 repeats 次。dim=None 时先展平。"""
    t = _as_tensor(input)
    if isinstance(repeats, (bool, np.bool_)) or not isinstance(repeats, (int, np.integer)):
        raise TypeError("repeat_interleave 的 repeats 目前只支持整数，收到 %s" % type(repeats).__name__ + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    repeats = int(repeats)
    if repeats < 0:
        raise RuntimeError("repeats 必须 >= 0，收到 %d" % repeats + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    if dim is None:
        t = t.reshape((int(t.numel()),))
        dim = 0
    if repeats == 1:
        return t
    if getattr(t, "device", "cpu") == "cuda":
        return t._repeat_interleave_cuda(repeats, dim)
    axis = int(dim)
    if t.ndim == 0:
        raise RuntimeError("标量不能 repeat_interleave，先 reshape 成 1 维" + "。修法：检查上面给出的形状：左边最后一维要等于右边倒数第二维；需要广播时先 reshape")
    if axis < 0:
        axis += t.ndim
    if axis < 0 or axis >= t.ndim:
        raise IndexError("dim %s 越界（ndim=%d）" % (dim, t.ndim) + "。修法：这是驱动/内核级失败；把完整错误、GPU 型号与驱动版本一起反馈（TG_DEBUG=1 可出更多日志）")
    data = np.repeat(t.data, repeats, axis=axis)
    if not np.issubdtype(t.data.dtype, np.floating):
        return Tensor(data, dtype=t.dtype, requires_grad=False)
    out = Tensor(data, requires_grad=t.requires_grad, _parents=(t,))

    def bw(g, grads):
        if not t.requires_grad:
            return
        g = np.asarray(g, dtype=np.float32)
        if repeats == 0:
            _acc(grads, t, np.zeros(t.shape, dtype=np.float32))
            return
        parts = list(g.shape)
        parts[axis] = int(parts[axis]) // repeats
        folded = parts[:axis] + [parts[axis], repeats] + parts[axis + 1:]
        _acc(grads, t, g.reshape(folded).sum(axis=axis + 1))
    out._backward = bw
    return out


def repeat_interleave(input, repeats, dim=None):
    """对应 torch.repeat_interleave(input, repeats, dim)。GQA 用 dim=0 或 dim=1。"""
    return _repeat_interleave_impl(input, repeats, dim)


# 公开名字放在文件末尾，避免盖住上面 Muon / cross_entropy 里的内置 max。
abs = _api_abs
clamp = _api_clamp
max = _api_max
min = _api_min

# ═══════════════════════════════════════════════════════════
# torch 同义名（迁移兼容）：让 PyTorch 脚本"只改导入"即可运行
# ═══════════════════════════════════════════════════════════
tensor = Tensor            # torch.tensor(...) -> Tensor(...)
# torch.set_default_device('cuda') 的同义实现
def set_default_device(device):
    return set_device(device)


# ═══════════════════════════════════════════════════════════
# 补充层（0.9.14）：Dropout / MaxPool / AvgPool / BatchNorm（1d/2d）
# 放在文件末尾导入：layers_extra 依赖本模块已定义的 Tensor/Module/Parameter/indexing
# ═══════════════════════════════════════════════════════════
from . import layers_extra as _layers_extra                          # noqa: E402

for _layer_name in ("Dropout", "MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d",
                    "BatchNorm1d", "BatchNorm2d"):
    setattr(nn, _layer_name, getattr(_layers_extra, _layer_name))
del _layer_name

# 0.9.21：PyT 用户在 `nn` 下找 Conv1d/Conv2d，这两个类原本只在顶层 ⇒ 补进 nn（同一对象，不是副本）
for _layer_name in ("Conv1d", "Conv2d"):
    if hasattr(nn, _layer_name) is False:
        setattr(nn, _layer_name, globals()[_layer_name])
del _layer_name

# ═══════════════════════════════════════════════════════════
# 归约算子（0.9.15）：max / min / var / std / norm 挂到 Tensor 上
# ═══════════════════════════════════════════════════════════
from . import redux_extra as _redux_extra                            # noqa: E402

for _red_name in ("max", "min", "var", "std", "norm"):
    setattr(Tensor, _red_name, getattr(_redux_extra, _red_name))
del _red_name

# ═══════════════════════════════════════════════════════════
# 高层偏好训练器（0.9.16）：DPOTrainer / GRPOTrainer
# ═══════════════════════════════════════════════════════════
from .rl_trainer import DPOTrainer, GRPOTrainer, PPOTrainer                       # noqa: E402

# ═══════════════════════════════════════════════════════════
# 可编程 AMP（0.9.20）：T.amp() / amp_on / amp_off / GradScaler
#
# 注意 `T.amp` 这个名字**以前是 bf16 模块**（`T.amp.to_bf16_bits`、`T.amp.BF16_KERNELS`
# 被测试与工具使用）。为了不破坏旧契约，这里做成"可调用 + 属性代理"的命名空间：
#   with T.amp():            -> 新 API（fp16 GEMM + 权重影子缓存）
#   T.amp.to_bf16_bits(...)  -> 旧 API（bf16 位转换工具）
# ═══════════════════════════════════════════════════════════
from . import amp_api as _amp_api                                     # noqa: E402
from . import amp as _amp_module                                      # noqa: E402


class _AmpNamespace(object):
    """既是 AMP 上下文工厂，又是旧 bf16 模块的代理。"""

    def __call__(self, *args, **kwargs):
        return _amp_api.amp(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(_amp_module, name)

    def __enter__(self):
        self._ctx = _amp_api.amp()
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)


amp = _AmpNamespace()
amp_on = _amp_api.amp_on
amp_off = _amp_api.amp_off
amp_state = _amp_api.amp_state
amp_invalidate = _amp_api.amp_invalidate
GradScaler = _amp_api.GradScaler

# ═══════════════════════════════════════════════════════════
# 开箱即用层（0.9.21）：T.preset.transformer / T.tokens / T.fit
# ═══════════════════════════════════════════════════════════
from . import preset as preset                                        # noqa: E402
from .preset import tokens, fit                                       # noqa: E402
# "从零开始"四步里第 1~2 步：读语料 → 训分词表（新手不该去记子模块路径）
from .preset import train_tokenizer, load_corpus, Tokenizer           # noqa: E402
from .data import read_jsonl, train_eval_split as split_dataset        # noqa: E402
# SFT 数据集与 PPO 子模块：README 里直接写 T.SftDataset / T.ppo.gae，必须真的能点出来
from .sft_data import SftDataset, iter_jsonl_texts                     # noqa: E402
from . import ppo as ppo                                              # noqa: E402

