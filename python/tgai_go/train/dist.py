# -*- coding: utf-8 -*-
"""数据并行（多卡）地基：参数广播 + 梯度全归约。

设计目标：**先正确，再快**。默认走「本机共享内存 + 文件屏障」做交换，
**不依赖 NCCL / MPI / torch.distributed**（我们本来就是自研运行时）；
需要更快时可显式切到 NCCL 后端（见下）。

进程模型：每个 rank 一个进程（各自的 CUDA_VISIBLE_DEVICES），
通过 `TGAI_DP_WORLD` / `TGAI_DP_RANK` 环境变量认领身份，
顺次目录（rendezvous）默认 `/tmp/tgai_dp_<job>`，Windows 下用 `%TEMP%`。

用法：
    dp = DataParallel()                    # 从环境变量读 world/rank
    dp.broadcast(params)                   # rank0 的权重分发给所有人（保证起点一致）
    ... 前向/反向 ...
    dp.all_reduce_grads(params)            # 梯度求平均，使每个 rank 拿到相同的更新方向
    opt.step()

world=1 时所有操作都是**快速空操作**，所以单卡跑不会有任何行为变化。

## 归约后端：`TG_DP_BACKEND=host|nccl|auto`（默认 host）

| 值 | 行为 | 失败时 |
| --- | --- | --- |
| `host`（默认） | 打包 → D2H(pinned) → 共享内存 → H2D → GPU 就地加 | 不会失败（原有路径，行为逐位不变）|
| `nccl` | ctypes 直连 `libnccl.so`，`ncclAllReduce(Sum/fp32)` + 乘 1/world | **明确报错**，不回退 |
| `auto` | 先试 NCCL，不可用则回退 host | 打印 `[dp] WARN` 后回退 |

`nccl` 后端的语义与 host 路径**一致**（都是求和后乘 `1/world`，缺失梯度按 0 补齐），
差别只在数据走 PCIe 还是走「主机中转」。库位置可用 `TG_NCCL_LIB` 指定。
NCCL 的 unique id 走同一个 rendezvous 目录（`<dir>/nccl_id`，rank0 原子写入）。

## 使用契约（2026-11-13 明确，之前只写在实现里）

1. `broadcast(params)` 在**训练开始前**调用一次，让各 rank 从同一组权重出发
   （否则"数据并行"实际在平均**不同模型**的梯度）；
2. `all_reduce_grads(params)` 每个 rank 传入**完全相同、顺序一致**的 params 列表；
   冻结参数不要放进来（缺梯度按 0 补齐，0 梯度仍会走 weight decay）；
3. 数据必须**各卡不同** —— 用 `dp.shard_indices(n)` 取本 rank 的那一份；
4. 检查点只让 `rank == 0` 写盘（本模块不做这件事，调用方判断）。

## 等效 batch 与学习率

每卡 batch = B、world = W 时，默认"梯度平均"让更新方向等价于 **W×B 的大 batch**。
所以学习率要按 W×B 的口径调（常见做法是 lr 保持、B 不变、靠更大 batch 换稳定；
或 lr 按 √W 放宽）。本模块**不会**替你改 lr。

## 梯度通信精度：`TG_DP_GRAD_DTYPE=fp32|bf16|fp16`（默认 fp32）

PCIe-only 机器上 DP 的瓶颈是**字节数**（实测 2×5090：3.16 GiB fp32 all-reduce
≈141 ms，而单卡整步只要 141–251 ms，重叠救不回来）。本模块支持把梯度**按低精度
通信**：归约前 cast 成 bf16/fp16，在低精度下做 all-reduce，结果 cast 回 fp32 写回
`p.grad`。**参数本身永远是 fp32**（不碰主权重、不碰优化器状态）。

* 默认 `fp32` ⇒ 所有路径与历史实现**逐位一致**（老路径的代码一个字没改）；
* 通信字节：848.6M 参数 × 4 B = **3.16 GiB** → bf16 = **1.58 GiB**（正好一半）；
* 均值语义不变（仍然 `sum/world`），只改精度；单元素相对舍入误差上界 bf16 = 2^-8
  （~3.9e-3，尾段典型 2^-9）、fp16 = 2^-11（~4.9e-4），真实 loss/gnorm 偏差见
  `docs/DP_BF16_GRAD.md` 的实测表；
* `world=1` 时仍然是**快速空操作**（单卡不涉及归约精度，梯度逐位不变）；
* `TG_DP_GRAD_EF=1` 打开误差反馈（把 cast 截断的残差留在 fp32 影子里，下次加回）。
  代价是**多一份 fp32 影子梯度**（848M 时 3.16 GiB 显存/内存），所以默认关；
  且 EF 需要「不重叠」路径（残差在打包缓冲里结算），会自动让 `begin_overlap()` 返回 None。
"""
import os
import ctypes
import tempfile
import time

import numpy as np

#: 本进程启动时刻（用于判定"上一轮遗留的文件"）。**必须在模块导入时取**：
#: 位于 `ready` 之前写出的文件只可能来自别的会话，不可能来自本进程。
_PROC_START = time.time()

try:
    from multiprocessing import shared_memory
    _HAS_SHM = True
except Exception:                                    # noqa: BLE001
    _HAS_SHM = False


def _runtime():
    """拿运行时单例（本模块不 import tgai_go.train，避免循环导入）。"""
    import tgai_go.train as T
    return T._cuda()


def _grad_to_numpy(g):
    """把梯度转成 host 上的 float32 ndarray。

    我们的 CUDA 梯度是 Tensor（不是 ndarray），直接 np.asarray 会得到 object 数组。
    用它的 to_numpy()（D2H 拷贝）拿真实数值。
    """
    try:
        if hasattr(g, "to_numpy"):
            return np.ascontiguousarray(np.asarray(g.to_numpy(), dtype=np.float32))
        return np.ascontiguousarray(np.asarray(g, dtype=np.float32))
    except Exception:                                    # noqa: BLE001
        return None


def _set_grad(p, val, orig):
    """把归约后的梯度写回参数。

    CUDA 优化器要求 grad 是**CUDA 张量**（传 ndarray 会报
    "CUDA AdamW 需要 CUDA Tensor 梯度"），所以要按参数所在的设备重建一个。

    ★ 判定依据必须是 **`p.device`**，不是"原来的 grad 长什么样"（2026-11-13 修）：
      某 rank **本来就缺梯度**时 `orig is None`，旧实现会走 `p.grad = val` 这条兜底，
      在 CUDA 上留下一个 **CPU ndarray** 梯度。MoE 里"这一卡没路由到该专家"正好命中，
      于是优化器要么报错、要么静默用错数据。
    """
    if getattr(p, "device", "cpu") == "cuda":
        from . import tensor as _tensor
        p.grad = _tensor(val, device="cuda")
        return
    try:
        if hasattr(orig, "to_numpy"):
            from . import tensor as _tensor
            p.grad = _tensor(val, device="cpu")
            return
    except Exception:                                    # noqa: BLE001
        pass
    p.grad = val


def _zero_grad_of(p):
    """给「本步没有梯度」的参数补一个**全 0 的梯度**（与不重叠路径逐位同一语义）。

    为什么要补而不是跳过：各 rank 的条目数/字节数必须一致（否则共享内存名字不同
    ⇒ 一边创建一边等不到，卡死）；而且不补的话优化器会跳过该参数，
    与"梯度为 0 但仍走 weight decay"的历史行为不一致。
    """
    shape = tuple(int(s) for s in getattr(p, "shape", ()) or ())
    _set_grad(p, np.zeros(shape, np.float32), None)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ══════════════════════════════════════════════════════════════════════════
# 梯度通信精度（TG_DP_GRAD_DTYPE）：fp32 / bf16 / fp16
# ══════════════════════════════════════════════════════════════════════════

#: 允许的低精度通信类型。默认 fp32 ⇒ 历史行为逐位不变。
GRAD_DTYPES = ("fp32", "bf16", "fp16")

#: 通信时每个元素的字节数（bf16/fp16 都是 2）。
_GRAD_ITEMSIZE = {"fp32": 4, "bf16": 2, "fp16": 2}


def grad_dtype_requested():
    """读 `TG_DP_GRAD_DTYPE` 并归一化。非法值**当场报错**，不静默当 fp32。

    为什么不静默：写错一个字母（`TG_DP_GRAD_DTYPE=b16`）就退回 **2 倍**通信字节，
    而日志里看不出任何异常 —— 这类「悄悄降级」是性能实验最难查的错
    （与 `dp_backend_requested` 同一条纪律）。
    """
    raw = os.environ.get("TG_DP_GRAD_DTYPE", "")
    v = str(raw).strip().lower()
    if v in ("", "fp32", "float32", "none", "off", "0"):
        return "fp32"
    if v in ("bf16", "bfloat16"):
        return "bf16"
    if v in ("fp16", "half", "float16"):
        return "fp16"
    raise ValueError(
        "TG_DP_GRAD_DTYPE=%r 不认识（可选 %s）" % (raw, "/".join(GRAD_DTYPES)) +
        "。修法：设成 fp32、bf16 或 fp16；想保持历史行为就删掉这个环境变量")


def grad_ef_requested():
    """读 `TG_DP_GRAD_EF`（误差反馈开关，默认关）。非法值当场报错。"""
    raw = os.environ.get("TG_DP_GRAD_EF", "")
    v = str(raw).strip().lower()
    if v in ("", "0", "false", "off", "no", "none"):
        return False
    if v in ("1", "true", "on", "yes"):
        return True
    raise ValueError(
        "TG_DP_GRAD_EF=%r 不认识" % (raw,) +
        "。修法：设 1 开、0 关；不想用就删掉这个环境变量")


def grad_itemsize(dtype):
    """低精度通信的每元素字节数（fp32=4，bf16/fp16=2）。未知 dtype 报错。"""
    try:
        return _GRAD_ITEMSIZE[str(dtype)]
    except KeyError:
        raise ValueError(
            "未知的梯度通信 dtype=%r（可选 %s）"
            % (dtype, "/".join(GRAD_DTYPES)) +
            "。修法：用 grad_dtype_requested() 归一化后再传，或别传这个参数")


def _f32_bits_flat(a):
    """fp32 主机数组 → uint32 位模式（一维、连续）。"""
    return np.ascontiguousarray(a, dtype=np.float32).reshape(-1).view(np.uint32)


def bf16_bits_from_f32(a):
    """fp32 → bf16 位模式（uint16），round-to-nearest-even。

    与 CUDA 内核 `cast_f32_to_bf16` 的公式**逐位一致**：
    ``(u + 0x7FFF + ((u >> 16) & 1)) >> 16``（u 是 fp32 的 32 位模式）。
    主机侧用 uint64 做加法避免 numpy 的 uint32 溢出告警；CUDA 里无符号回绕语义相同。
    """
    u = _f32_bits_flat(a).astype(np.uint64)
    r = (u + np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)
    return r.astype(np.uint16)


def f32_from_bf16_bits(bits):
    """bf16 位模式（uint16）→ fp32（左移 16 位，**精确**，不丢位）。"""
    b = np.ascontiguousarray(bits, dtype=np.uint16).reshape(-1)
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


def quantize_to_bits(a, dtype):
    """fp32 主机数组 → 2 字节低精度位模式（uint16；bf16/fp16 都是 16 位位模式）。"""
    if dtype == "bf16":
        return bf16_bits_from_f32(a)
    if dtype == "fp16":
        return (np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
                .astype(np.float16).view(np.uint16))
    raise ValueError(
        "quantize_to_bits 只接受 bf16/fp16，收到 %r" % (dtype,) +
        "。修法：fp32 不走位模式；先确认 self.grad_dtype != 'fp32' 再调用")


def dequantize_from_bits(bits, dtype):
    """2 字节低精度位模式 → fp32 主机数组（这一侧是精确的，误差只来自收窄）。"""
    if dtype == "bf16":
        return f32_from_bf16_bits(bits)
    if dtype == "fp16":
        b = np.ascontiguousarray(bits, dtype=np.uint16).reshape(-1)
        return b.view(np.float16).astype(np.float32)
    raise ValueError(
        "dequantize_from_bits 只接受 bf16/fp16，收到 %r" % (dtype,) +
        "。修法：fp32 不走位模式；先确认 self.grad_dtype != 'fp32' 再调用")


# ---- 设备侧：把低精度 cast 进**已分配**缓冲（复用 cuda.py 里已有的内核） ----
# 为什么不调用 `rt.cast_f32_to_bf16_flat()`：那个每次调用都 `empty()` 一块新显存。
# 3.16 GiB 梯度每步分配 1.58 GiB（cast 后）再释放，既抬峰值又拖慢 —— 实测过一次，
# 显存红线（≤0.5 GiB 涨幅）直接破。这里直接发射同名内核，目标缓冲跨步复用。

def _mark_ptrs(rt, *bufs):
    """把缓冲标成「已提交到流」（内存池的复用安全前提）。"""
    for b in bufs:
        if b is not None:
            try:
                rt._mark_submitted(int(b.ptr))
            except Exception:                            # noqa: BLE001
                pass


def _cast_f32_to_16(rt, dtype, src_f32, dst16, n, src_off=0, dst_off=0, stream=None):
    """fp32 缓冲 ``[src_off, src_off+n)`` → 2 字节缓冲 ``[dst_off, dst_off+n)``（异步）。"""
    n = int(n)
    if n <= 0:
        return
    name = "cast_f32_to_bf16" if str(dtype) == "bf16" else "cast_f32_to_f16"
    grid, block = rt._ewise_grid(n)
    _mark_ptrs(rt, src_f32, dst16)
    rt.launch(name, grid, block, [
        (int(src_f32.ptr) + 4 * int(src_off), ctypes.c_uint64),
        (int(dst16.ptr) + 2 * int(dst_off), ctypes.c_uint64),
        (n, ctypes.c_int),
    ], stream=stream)


def _cast_16_to_f32(rt, dtype, src16, dst_f32, n, src_off=0, dst_off=0, stream=None):
    """2 字节缓冲 ``[src_off, src_off+n)`` → fp32 缓冲 ``[dst_off, dst_off+n)``（异步）。"""
    n = int(n)
    if n <= 0:
        return
    name = "cast_bf16_to_f32" if str(dtype) == "bf16" else "cast_f16_to_f32"
    grid, block = rt._ewise_grid(n)
    _mark_ptrs(rt, src16, dst_f32)
    rt.launch(name, grid, block, [
        (int(src16.ptr) + 2 * int(src_off), ctypes.c_uint64),
        (int(dst_f32.ptr) + 4 * int(dst_off), ctypes.c_uint64),
        (n, ctypes.c_int),
    ], stream=stream)


def _zero_16_region(rt, dtype, buf16, elem_off, n_elems, zero1, stream=None):
    """把 2 字节缓冲的 ``[elem_off, elem_off+n_elems)`` 置为 0（bf16/fp16 的 +0 位模式都是 0）。

    为什么不用内核：没有现成的「按 2 字节填 0」内核，而 `scale_inplace` 只认 fp32。
    做法：成对的元素用 `cuMemsetD32Async`（4 字节字，正好两个元素），奇数尾元素
    用 ``cast_f32_to_16(0.0)`` 精确写进去。**绝不**越界写 4 字节（那会把相邻参数
    的第一个元素抹掉 —— 重叠归约里那是一个已经 stage 好的梯度）。
    """
    n = int(n_elems)
    if n <= 0:
        return
    words = n // 2
    if words:
        st = rt._stream if stream is None else stream
        rt.cu.cuMemsetD32Async(ctypes.c_uint64(int(buf16.ptr) + 2 * int(elem_off)),
                               ctypes.c_uint(0), ctypes.c_size_t(words), st)
    if n % 2:
        _cast_f32_to_16(rt, dtype, zero1, buf16, 1,
                        dst_off=int(elem_off) + n - 1, stream=stream)


def _lowp_chunk_elems(total):
    """低精度打包的分块元素数（`TG_DP_PACK_MB` 默认 256 MB；0 ⇒ 一整块）。

    抽成函数是为了单测能用**假运行时**把多分块路径也跑到（见 tests/test_dp_bf16_grad.py）。
    """
    pack_mb = _env_int("TG_DP_PACK_MB", 256)
    total = int(total)
    return (total if pack_mb <= 0
            else max(1, min(total, (int(pack_mb) << 20) // 4)))


#: 允许的归约后端。默认 host（与历史行为逐位一致）。
BACKENDS = ("host", "nccl", "auto")


def dp_backend_requested():
    """读 `TG_DP_BACKEND` 并归一化。非法值**当场报错**，不静默当 host。

    为什么不静默：写错一个字母（`TG_DP_BACKEND=ncl`）就退回慢 1.5 倍的路径，
    而日志里看不出任何异常 —— 这种"悄悄降级"是性能实验最难查的一类错。
    """
    raw = os.environ.get("TG_DP_BACKEND", "")
    v = str(raw).strip().lower()
    if v in ("", "host", "none", "off", "0"):
        return "host"
    if v == "nccl":
        return "nccl"
    if v == "auto":
        return "auto"
    raise ValueError(
        "TG_DP_BACKEND=%r 不认识（可选 %s）" % (raw, "/".join(BACKENDS)) +
        "。修法：设成 host、nccl 或 auto；不想用 NCCL 就直接不设这个环境变量")


def _get_data(p):
    """把参数数值取成 host 上的 float32 ndarray。

    ## 为什么不能直接 `p.data`（2026-11-13 修的静默 bug）

    CUDA 张量的 `self.data` 是 **None**（数据在 `self._gpu`），而
    `np.ascontiguousarray(None, dtype=np.float32)` **不会报错**，它返回 `array([nan])`。
    于是旧版 `broadcast()` 在 GPU 上会把每个参数变成"1 个元素的 nan"：
    rank0 往共享内存写 nan，其余 rank 把一个临时数组写成 nan —— **权重根本没广播，也不报错**。
    多卡就这样在"各卡模型不同"的状态下继续训练。

    这里改成：CUDA 走 `to_numpy()`（D2H），CPU 用 `data`/`np.asarray`。
    """
    if getattr(p, "device", "cpu") == "cuda":
        return np.ascontiguousarray(np.asarray(p.to_numpy(), dtype=np.float32))
    d = getattr(p, "data", None)
    if isinstance(d, np.ndarray):
        return np.ascontiguousarray(np.asarray(d, dtype=np.float32))
    return np.ascontiguousarray(np.asarray(p, dtype=np.float32))


def _set_data(p, val):
    """把 host 上的 float32 数组**写回参数本身**（CUDA 原地 H2D，不新分配缓冲）。

    旧版 `broadcast()` 对 rank!=0 是往 `np.ascontiguousarray(...)` 的临时数组里赋值，
    那个数组与张量无关 ⇒ 即使形状算对，权重也不会被更新。这里区分设备真正写回。
    """
    val = np.ascontiguousarray(np.asarray(val, dtype=np.float32))
    dev = getattr(p, "device", "cpu")
    if dev == "cuda":
        from . import _cuda
        rt = _cuda()
        buf = getattr(p, "_gpu", None)
        if buf is None:
            raise RuntimeError("CUDA 参数缺少 _gpu 缓冲，无法写回" + "。修法：确认该张量是用 tgai_go.train.tensor(...) 或 Tensor._from_gpu 正常创建的")
        if tuple(buf.shape) != tuple(val.shape):
            raise RuntimeError(
                "写回形状不符：参数 %s，数据 %s。修法：形状不同的张量要重建，不能原地写"
                % (tuple(buf.shape), tuple(val.shape)))
        rt.copy_h2d_into(buf, val)
        return
    d = getattr(p, "data", None)
    if not isinstance(d, np.ndarray) or d.shape != val.shape:
        raise RuntimeError(
            "CPU 参数写回失败：data 不可用或形状不符（%r vs %s）。修法：确认传进来的是 ORV Tensor/ndarray"
            % (getattr(d, "shape", None), val.shape))
    d[...] = val


class DataParallel:
    def __init__(self, world=None, rank=None, job=None, timeout=None, backend=None,
                 grad_dtype=None):
        self.world = _env_int("TGAI_DP_WORLD", 1) if world is None else int(world)
        self.rank = _env_int("TGAI_DP_RANK", 0) if rank is None else int(rank)
        self.job = job or os.environ.get("TGAI_DP_JOB", "default")
        # 超时可配：对端崩了以后等 600 s 毫无意义（实验里就是白烧钱）。
        # TGAI_DP_TIMEOUT 让失败快一点现形。
        self.timeout = (float(timeout) if timeout is not None
                        else _env_float("TGAI_DP_TIMEOUT", 600.0))
        # 后端：显式参数 > 环境变量 > 默认 host
        self.backend_requested = (str(backend).strip().lower() if backend is not None
                                  else dp_backend_requested())
        if self.backend_requested not in BACKENDS:
            raise ValueError("backend=%r 不认识（可选 %s）"
                             % (backend, "/".join(BACKENDS)) +
                             "。修法：传 host、nccl 或 auto；默认 host")
        # 梯度通信精度：显式参数 > 环境变量 > 默认 fp32（历史行为逐位不变）
        self.grad_dtype = (str(grad_dtype).strip().lower() if grad_dtype is not None
                           else grad_dtype_requested())
        if self.grad_dtype not in GRAD_DTYPES:
            raise ValueError(
                "grad_dtype=%r 不认识（可选 %s）" % (grad_dtype, "/".join(GRAD_DTYPES)) +
                "。修法：传 fp32/bf16/fp16；默认 fp32")
        # 误差反馈：默认关（多一份 fp32 影子梯度，见模块文档）
        self.grad_ef = bool(grad_ef_requested()) and self.grad_dtype != "fp32"
        if grad_ef_requested() and self.grad_dtype == "fp32":
            raise ValueError(
                "TG_DP_GRAD_EF=1 只对低精度通信有意义（当前 TG_DP_GRAD_DTYPE=fp32）" +
                "。修法：把 TG_DP_GRAD_DTYPE 设成 bf16/fp16，或删掉 TG_DP_GRAD_EF")
        self.backend = "host"                # 实际生效的后端（nccl 初始化失败时可能回退）
        self._nccl = None
        self._nccl_lib = None
        self._nccl_stream = None
        self._nccl_own_stream = False
        self._nccl_events = None
        #: 最近一次 all_reduce 的墙钟耗时（含通信+缩放；表里要单列的就是它）
        self.comm_sec = None
        #: 最近一次 all_reduce 的 NCCL-only 耗时（host 路径为 None）
        self.nccl_sec = None
        self._nccl_calls = 0
        #: 最近一次归约的**通信有效载荷字节**（每 rank 发出去的字节数）。
        #: 验收口径：fp32 时 848.6M 参数 = 3.16 GiB，bf16 = 1.58 GiB（正好一半）。
        #: 低精度路径每条都会写；fp32 路径保持不变（None），由调用方自己算。
        self.comm_rank_bytes = None
        self.comm_rank_bytes_fp32 = None
        #: 误差反馈的影子梯度缓冲（延迟分配，见 _ef_*；fp32 / 未开 EF 时恒为 None）
        self._ef_host_buf = None
        self._ef_dev_buf = None
        #: 低精度归约用的跨步复用缓冲（pack32 / pack16 / scratch32）
        self._lp_buf = None
        self._zero1 = None
        self._host16 = None
        self._ef_ov_warned = False
        #: 低精度 + 反向重叠的组合被实测判为**数值未通过**，只警告一次（见 begin_overlap）
        self._lowp_ov_warned = False
        # ── 反向重叠归约（`begin_overlap` / `finish_overlap`）的状态 ──
        #: 当前 step 的重叠会话（`dp_overlap.OverlapSession`）；没开就是 None
        self._ov = None
        #: 最近一次重叠归约的 stats（暴露时间/重叠时间/桶数/pending 校验结果）
        self.overlap_stats = None
        #: 重叠归约专用的**非阻塞**流（NCCL 用；host 后端自己建一条）
        self._ov_stream = None
        #: ZeRO 用的按用途名管理的辅助共享内存（`_aux_slot`）：key -> (slot, 容量字节)
        self._aux = {}
        base = os.environ.get("TGAI_DP_DIR") or tempfile.gettempdir()
        self.dir = os.path.join(base, "tgai_dp_%s" % self.job)
        self._shm = []
        if self.world > 1 and not _HAS_SHM:
            raise RuntimeError("多卡需要 multiprocessing.shared_memory，当前环境不可用" + "。修法：把 TGAI_DP_WORLD 设为 1 跑单卡，或换支持 shared_memory 的平台")
        if self.world > 1:
            os.makedirs(self.dir, exist_ok=True)
            self._slot = None
            self._bytes = 0
            # ★ 握手放到 `_start()` 里惰性执行（2026-11-13 修）：
            #   构造 `DataParallel(world>1)` 就等 `ready` 会让**任何**不打算立刻通信的用法
            #   卡死（单测里造一个 rank=1 的对象就会永久等待）。
            self._started = False

    # ---------- 基础设施 ----------

    def _start(self):
        """惰性完成 rendezvous：清残留标记 → 放 `ready` → 首次 barrier。

        只有**真的要通信**时才做，所以 `DataParallel(world=2)` 本身不会阻塞。

        为什么要清残留：generation 号每个进程都从 0 重数，若上一轮崩溃留下
        `bar_grads_1.<r>`，本轮同级 barrier 会因为"文件已存在"而**立即通过**
        ⇒ 少一次真实同步，读到别人半写的缓冲，梯度静默变错。

        ★ 2026-11-21 修的第二个残留坑：`ready` 本身也会残留。
          上一轮跑崩后 `/tmp/tgai_dp_<job>/ready` 还在，本轮 rank1 若比 rank0 的
          清理更快，就会相信这个旧文件直接冲向 `bar_init_1`，而 rank0 随后把
          `bar_init_1.1` 当残留删掉 ⇒ **双方各等 600 s**（实测复现两次）。
          修法：`ready` 里写 rank0 的 pid，rank>0 看到 pid 已死就认定是旧文件、
          自己删掉再等新的。rank0 依旧在清理阶段无条件删 `ready`。
        """
        if self.world <= 1 or getattr(self, "_started", False):
            return
        sess = self._session_token()          # 本进程的会话身份（毫秒整数，足够唯一）
        if self.rank == 0:
            # 只清**上一轮**的东西（比本进程更老的），本轮同伴的文件一个都不碰。
            for f in os.listdir(self.dir):
                if not (f.startswith("bar_") or f.startswith("ready")):
                    continue
                p = os.path.join(self.dir, f)
                try:
                    if os.stat(p).st_mtime < _PROC_START - 1.0:
                        os.remove(p)
                except OSError:
                    pass
            with open(os.path.join(self.dir, "ready.%d" % sess), "wb") as fh:
                # 三个字段：写者 pid / 写者当前时间 / **写者的进程启动时刻**。
                fh.write(b"%d %f %f" % (os.getpid(), time.time(), _PROC_START))
        self._session_id = _PROC_START if self.rank == 0 else self._await_peer_ready()
        self._barrier("init")
        # ★ 各 rank 的通信 dtype 必须一致，否则集合通信的字节数/类型不同：
        #   NCCL 会挂或算出垃圾，host 路径会因共享内存名字不同而互相死等。
        #   写错环境变量是最容易发生的一种「各 rank 不一致」，所以在 _start 里强制对齐。
        self._sync_grad_dtype()
        self._started = True
        self._ensure_backend()

    @staticmethod
    def _pid_alive(pid):
        """进程是否还活着（跨平台；**不能用 `os.kill(pid, 0)` 了事**）。

        ★ Windows 上 `os.kill(pid, 0)` 会走 `TerminateProcess`，不能拿来探测存活。
          这里用 `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)`：句柄为空时再看
          `GetLastError` —— 5(拒绝访问)=活着，87(参数错)=pid 不存在。
        """
        pid = int(pid)
        if pid <= 0:
            return False
        if os.name == "nt":
            try:
                import ctypes
                k32 = ctypes.windll.kernel32
                h = k32.OpenProcess(0x1000, False, pid)
                if h:
                    k32.CloseHandle(h)
                    return True
                return int(k32.GetLastError()) == 5
            except Exception:                        # noqa: BLE001
                return True                          # 探测不了就不敢说它死了
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return True
        return True

    @staticmethod
    def _ready_is_stale(path):
        """`ready` 是不是**上一轮**留下的。

        判据：文件里记着写者的**进程启动时刻**（第 3 个字段）。它比本进程启动更早
        ⇒ 必然来自别的会话，与本轮无关 ⇒ **忽略并继续等**，绝**不删除**。

        为什么不用 pid（旧实现）以及为什么不删（2026-11-21，CI ubuntu-3.10 双死锁）：
        旧实现是「读到 pid 已死 ⇒ `os.remove(ready)` 再等」。这是 read-then-delete 竞态：
        rank1 读到旧的（pid 已死）决定删，在它读与删之间 rank0 写入了新的
        ⇒ rank1 删掉了**新文件** ⇒ rank0 卡在 `bar_init_1`、rank1 卡在 `ready`，
        各等 600 s。而且 pid 会被系统复用，"pid 活着"根本不能证明"这是本轮"。
        换成"写者的启动时刻"后，身份判定与 pid 复用无关；再配合"只忽略不删除"，
        任何一方都不会破坏对方的文件。
        """
        try:
            with open(path, "rb") as fh:
                raw = fh.read(96)
            mtime = os.stat(path).st_mtime
        except OSError:
            return False                 # 没有文件 ⇒ 不是"残留"，正常等
        parts = raw.split()
        if len(parts) >= 3:
            try:
                return float(parts[2]) < float(_PROC_START) - 1.0
            except ValueError:
                pass
        # 老格式（只有 pid [时间]）：退回"文件比本进程还老"来判断，
        # 仍是只读不删 —— 宁可多等一轮，也不许删掉别人刚写的标记。
        return float(mtime) < float(_PROC_START) - 1.0

    def _marker(self, name):
        return os.path.join(self.dir, "%s.%d" % (name, self.rank))

    def _wait_for(self, path, what):
        t0 = time.time()
        while not os.path.exists(path):
            if time.time() - t0 > self.timeout:
                raise TimeoutError("数据并行等待超时（%s）: %s" % (what, path) + "。修法：确认所有 rank 都已启动、TGAI_DP_JOB 一致、临时目录可写")
            time.sleep(0.002)

    def _barrier(self, tag):
        """简易 barrier：每个 rank 落一个文件，等所有 rank 到齐。

        ★ 必须带「世代号」且**不删除**标记文件：
          早期版本里 rank0 通过后立刻删掉自己的标记，rank1 可能还没看到就已经被删，
          于是永久等待（实测 2 进程死锁）。带上自增世代之后，每次 barrier 用全新文件名，
          旧文件留着也不会让下一轮误判为「已到齐」。

        ★★ 还必须带**会话号**（2026-11-21）：同一个 `dir` + 同一个 `job` 上如果有
           **两个并发会话**，它们会看到对方的 `bar_*` 标记 ⇒ 假"已到齐" ⇒ 各走各的、
           **静默错序**（比死锁更难查）。把会话号编进文件名，两组标记在名字层面
           就永不交叉，于是并发会话可以安全共用同一个目录。
        """
        self._bar_gen = int(getattr(self, "_bar_gen", 0)) + 1
        # ★ 会话号必须是**共享的** `_session_id`（rank0 的启动时刻，rank>0 从 ready 里读到），
        #   绝不能退回"本进程启动时刻"—— 那会让每个 rank 用各不相同的文件名，
        #   于是互相等不到对方，直接死锁（2026-11-21 我在这里犯过一次）。
        sid = getattr(self, "_session_id", None)
        if sid is None:
            if self.world > 1:
                raise RuntimeError(
                    "barrier 在 rendezvous 之前被调用（_session_id 未设置）" +
                    "。修法：不要在 DataParallel._start() 之前触发集合通信")
            sid = _PROC_START
        tok = int(round(float(sid) * 1000.0))
        t = "%s_%d_%d" % (tag, self._bar_gen, tok)
        p = os.path.join(self.dir, "bar_%s.%d" % (t, self.rank))
        with open(p, "wb") as f:
            f.write(b"1")
        for r in range(self.world):
            self._wait_for(os.path.join(self.dir, "bar_%s.%d" % (t, r)), "barrier " + t)

    @staticmethod
    def _session_token():
        """本进程的会话身份（毫秒整数）。仅用于给 `ready.<token>` 起唯一名字。"""
        return int(round(float(_PROC_START) * 1000.0))

    def _await_peer_ready(self, timeout=None):
        """rank>0 等本轮 rank0 的 `ready.<token>`，返回**rank0 的启动时刻**当会话身份。

        为什么不固定文件名 `ready`（2026-11-21，CI 三条腿红 + 本地复现）：
        `_wait_for` 只判断"文件存在"。上一轮留下的 `ready` 一直在那儿 ⇒ 本轮 rank>0
        会**立刻返回**并拿到**上一轮的身份**，于是两端算出不同的共享内存名，
        各自等到超时。改成"每轮一个文件名 + 只认比我自己新的"之后，
        残留文件在名字层面就不可能被误当成本轮 —— 也仍然不需要删除任何人。

        找不到就明确报错（含修法），绝不退回自己的时刻（那会静默错配名字）。
        """
        limit = float(self.timeout if timeout is None else timeout)
        t0 = time.time()
        while True:
            best = None
            try:
                names = os.listdir(self.dir)
            except OSError:
                names = []
            for f in names:
                if not f.startswith("ready."):
                    continue
                p = os.path.join(self.dir, f)
                try:
                    if os.stat(p).st_mtime < _PROC_START - 1.0:
                        continue                     # 上一轮的，忽略（不删）
                    with open(p, "rb") as fh:
                        parts = fh.read(96).split()
                    if len(parts) >= 3:
                        sid = float(parts[2])
                        if sid >= _PROC_START - 1.0:  # 只认本轮的
                            if best is None or sid > best:
                                best = sid
                except OSError:
                    continue
            if best is not None:
                return best
            if time.time() - t0 > limit:
                raise TimeoutError(
                    "等不到本轮 rank0 的就绪标记（%s/ready.<会话号>）" % self.dir +
                    "。修法：确认所有 rank 都已启动、TGAI_DP_JOB 一致、临时目录可写；"
                    "若上一次运行崩在同一个 job 上，换个 TGAI_DP_JOB 或删掉该目录")
            time.sleep(0.002)

    def _read_session_id(self, path, timeout=5.0):
        """从 `ready` 里取出 rank0 的会话身份（第 3 字段）。带重试：文件可能刚被写出。

        取不到就 **明确报错**，绝不悄悄退回自己的启动时刻 —— 那会让两端算出
        不同的共享内存名，然后各自等到超时（比直接报错难查得多）。
        """
        t0 = time.time()
        while True:
            try:
                with open(path, "rb") as fh:
                    parts = fh.read(96).split()
                if len(parts) >= 3:
                    return float(parts[2])
            except OSError:
                pass
            if time.time() - t0 > float(timeout):
                raise TimeoutError(
                    "读不到会话身份（%s 里没有第 3 字段）" % path +
                    "。修法：确认所有 rank 用的是同一个版本的 tgai-orv（旧版只写 2 个字段）")
            time.sleep(0.01)

    def _slot_name(self, need):
        """共享内存名必须带**会话身份**，不能只用 `job`。

        2026-11-21（CI 三条腿变红）：名字原本是 `tgai_dp_<job>_<need>`。
        同一 `dir` + 同一 `job` 开**第二个会话**时名字与第一个完全相同，
        而第一个会话收尾会 `unlink` 掉它 ⇒ 第二个会话的 rank1 挂不上、
        rank0 卡在 `bar_init_1`，双方超时。这不是"用户用错"，是名字设计漏了会话维度。

        会话身份由 `ready` 里那个**写者启动时刻**给出（rank0 自己就是它，
        rank>0 在 `_start()` 里读到），所以两端不需要额外通信就能算出同一个名字。
        """
        sid = getattr(self, "_session_id", None)
        if sid is None:
            sid = _PROC_START
        return "tgai_dp_%s_%d_%d" % (self.job, int(round(float(sid) * 1000.0)), int(need))

    def _sync_grad_dtype(self):
        """各 rank 交换自己的梯度通信 dtype，不一致就**当场报错**。

        为什么必须对齐：低精度归约改变了集合通信的「元素类型 × 字节数」。
        两个 rank 一个 bf16 一个 fp32 时，NCCL 的调用类型不一致（挂或算错），
        host 路径则因共享内存名字里的字节数不同而互相死等 —— 后者会表现为
        「训练卡住」，极难定位。所以宁可在启动时就炸。
        """
        if self.world <= 1:
            return
        sid = int(round(float(getattr(self, "_session_id", _PROC_START)) * 1000.0))
        path = os.path.join(self.dir, "grad_dtype_%d.%d" % (sid, int(self.rank)))
        with open(path, "wb") as fh:
            fh.write(str(self.grad_dtype).encode("ascii"))
        self._barrier("dtype")
        seen = {}
        for r in range(int(self.world)):
            p = os.path.join(self.dir, "grad_dtype_%d.%d" % (sid, r))
            try:
                with open(p, "rb") as fh:
                    seen[r] = fh.read(64).decode("ascii", "replace").strip() or "?"
            except OSError:
                seen[r] = "?"
        if len(set(seen.values())) > 1:
            raise RuntimeError(
                "各 rank 的 TG_DP_GRAD_DTYPE 不一致：%s"
                % ", ".join("rank%d=%s" % (r, v) for r, v in sorted(seen.items())) +
                "。修法：所有 rank 必须设成同一个值（fp32/bf16/fp16）；"
                "低精度归约会改变集合通信的类型与字节数，不一致会挂死或算错")

    def _ensure_slot(self, nbytes):
        """所有 rank 共用一块 `world * nbytes` 的共享内存，按 rank 分段写。"""
        if self._slot is not None and self._bytes >= nbytes:
            return
        if self._slot is not None:
            self._slot.close()
        need = int(nbytes)
        name = self._slot_name(need)
        if self.rank == 0:
            self._slot = shared_memory.SharedMemory(
                name=name, create=True, size=need * self.world)
            self._slot.buf[:8] = need.to_bytes(8, "little")
            self._shm.append(self._slot)
        else:
            t0 = time.time()
            while True:
                try:
                    self._slot = shared_memory.SharedMemory(name=name, create=False)
                    break
                except FileNotFoundError:
                    if time.time() - t0 > self.timeout:
                        raise TimeoutError("共享内存 %s 未创建" % name + "。修法：确认 rank0 已完成共享内存创建（可能有 rank 提前退出）")
                    time.sleep(0.002)
            self._shm.append(self._slot)
        self._bytes = need

    # ---------- NCCL 后端 ----------

    def _ensure_backend(self):
        """决定实际后端。**显式 nccl 失败 ⇒ 报错；auto 才回退**。

        调用点：`_start()`（第一次真正通信时）。放在这里而不是构造时，是为了让
        「造一个 world=2 的对象但从不通信」的单测不会去碰 GPU/NCCL。
        """
        if self.world <= 1:
            self.backend = "host"
            return
        if self.backend_requested == "host":
            self.backend = "host"
            return
        try:
            self._init_nccl()
            self.backend = "nccl"
        except Exception as e:                           # noqa: BLE001
            if self.backend_requested == "nccl":
                raise RuntimeError(
                    "TG_DP_BACKEND=nccl 但 NCCL 后端初始化失败：%s" % (e,) +
                    "。修法：按上面的原因修（装 NCCL / 设 TG_NCCL_LIB / 对齐 world），"
                    "或显式改用 TG_DP_BACKEND=host 走主机路径") from e
            print("[dp] WARN TG_DP_BACKEND=auto：NCCL 不可用（%s）⇒ 回退 host 路径"
                  % (e,), flush=True)
            self.backend = "host"

    def _make_nccl_stream(self, rt):
        """给 NCCL 单独开一条流，并用 event 与计算流互相同步。

        为什么要单独一条流（而不是直接用 `rt._stream`）：
        `rt._stream` 默认是 **legacy NULL 流**。NCCL 对 NULL 流的处理依赖版本，
        而"独立流 + event 双向栅栏"是 PyTorch/DeepSpeed 同样采用的、无歧义的做法：
        计算流 --event0--> NCCL 流 --event1--> 计算流。

        任何一步拿不到符号就退回 `rt._stream`（并打印一次原因）：这是**流的选择**，
        不是后端失败；NCCL 本身仍然要求 `stream` 参数可用。
        """
        cu = getattr(rt, "cu", None)
        if cu is None:
            return rt._stream, False
        try:
            pvoid = ctypes.POINTER(ctypes.c_void_p)
            create = rt._bind_sym("cuStreamCreate", [pvoid, ctypes.c_uint])
            wait = rt._bind_sym("cuStreamWaitEvent",
                                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint])
            ev_create = rt._bind_sym("cuEventCreate", [pvoid, ctypes.c_uint])
            ev_rec = rt._bind_sym("cuEventRecord",
                                  [ctypes.c_void_p, ctypes.c_void_p])
            if not all((create, wait, ev_create, ev_rec)):
                print("[dp] WARN 驱动缺少 stream/event 入口 ⇒ NCCL 直接用 rt._stream",
                      flush=True)
                return rt._stream, False
            st = ctypes.c_void_p()
            if int(create(ctypes.byref(st), 0)) != 0 or not st.value:
                print("[dp] WARN cuStreamCreate 失败 ⇒ NCCL 直接用 rt._stream",
                      flush=True)
                return rt._stream, False
            e0, e1 = ctypes.c_void_p(), ctypes.c_void_p()
            if (int(ev_create(ctypes.byref(e0), 0)) != 0
                    or int(ev_create(ctypes.byref(e1), 0)) != 0):
                print("[dp] WARN cuEventCreate 失败 ⇒ NCCL 直接用 rt._stream",
                      flush=True)
                return rt._stream, False
            self._nccl_events = (e0, e1)
            return st, True
        except Exception as e:                           # noqa: BLE001
            print("[dp] WARN 建 NCCL 流失败（%r）⇒ 直接用 rt._stream" % (e,),
                  flush=True)
            return rt._stream, False

    def _init_nccl(self):
        """rendezvous → 建 comm。所有 rank 必须都走到这里（`_start()` 已同步过）。"""
        from . import dist_nccl as N

        # ★ 通道数默认 8：同一台机、无 P2P（SYS/PCIe）时实测这组参数把
        #   3.16 GiB fp32 all-reduce 从 134 ms 降到 86 ms（1.55×）。
        #   为什么放在这里：NCCL 在 **ncclCommInitRank** 时读 NCCL_* 环境变量，
        #   而 ORV 的 DP 通信器正是在这里建；用户显式设了就不覆盖（setdefault）。
        #   ★ 这是「同机多卡走 PCIe」的实测最优，不是普适最优：换 NVLink/多机
        #   拓扑时请显式设 NCCL_MIN_NCHANNELS / NCCL_MAX_NCHANNELS 覆盖。
        os.environ.setdefault("NCCL_MIN_NCHANNELS", "8")
        os.environ.setdefault("NCCL_MAX_NCHANNELS", "8")
        rt = _runtime()
        lib = N.find_libnccl()
        if self.rank == 0:
            uid = lib.get_unique_id()
            raw = N.unique_id_to_bytes(uid)
            N.write_unique_id(self.dir, raw)
        else:
            raw = N.wait_unique_id(self.dir, timeout=self.timeout)
        stream, own = self._make_nccl_stream(rt)
        comm = N.NcclComm(lib, self.world, self.rank, N.unique_id_from_bytes(raw),
                          stream=stream)
        self._nccl_lib = lib
        self._nccl = comm
        self._nccl_stream = stream
        self._nccl_own_stream = bool(own)
        print("[dp] nccl backend ready: lib=%s version=%s world=%d rank=%d device=%s "
              "stream=%s events=%s"
              % (lib.path, lib.version_str(), self.world, self.rank, comm.device,
                 "own" if own else "rt._stream", "yes" if own else "no"), flush=True)

    def _nccl_before(self, rt):
        if self._nccl_own_stream and self._nccl_events is not None:
            rt.cu.cuEventRecord(self._nccl_events[0], rt._stream)
            rt.cu.cuStreamWaitEvent(self._nccl_stream, self._nccl_events[0], 0)

    def _nccl_after(self, rt):
        if self._nccl_own_stream and self._nccl_events is not None:
            rt.cu.cuEventRecord(self._nccl_events[1], self._nccl_stream)
            rt.cu.cuStreamWaitEvent(rt._stream, self._nccl_events[1], 0)

    def _all_reduce_nccl(self, entries, total, scale):
        """NCCL 路径：**不做打包**，每张量一次 `ncclAllReduce` 并 group 起来。

        为什么不打包：打包要多付 2×3.16 GiB 的 D2D（~20 ms 各一遍），而 NCCL 对
        10 MB 级的连续块效率已经很高；实测分组后的 321 次调用没有明显劣化。
        """
        rt = _runtime()
        if self._nccl is None:
            raise RuntimeError("NCCL 后端未初始化" + "。修法：先调用一次 broadcast()/_start()")
        ptrs, counts = [], []
        missing = []
        missing_total = 0
        for p, g, n in entries:
            if g is None:
                missing.append((p, n))
                missing_total += int(n)
            else:
                if getattr(g, "_gpu", None) is None:
                    raise RuntimeError(
                        "NCCL 后端要求梯度在 GPU 上，但参数 %s 的梯度没有 _gpu 缓冲"
                        % (tuple(getattr(p, "shape", ())),) +
                        "。修法：TG_DP_BACKEND=nccl 只能用于 CUDA 训练；CPU 后端请用 host")
            ptrs.append(0 if g is None else int(g._gpu.ptr))
            counts.append(int(n))
        zero_buf = None
        if missing_total > 0:
            # 缺梯度（MoE 没路由到该专家等）必须**以 0 参与集合通信**：
            # 各 rank 的 params 列表相同、count 相同，谁也不能少发一次。
            zero_buf = rt.empty((int(missing_total),), dtype=np.float32)
            rt.copy_h2d_into(zero_buf, np.zeros(int(missing_total), np.float32))
            off = 0
            for i, (p, g, n) in enumerate(entries):
                if g is None:
                    ptrs[i] = int(zero_buf.ptr) + off * 4
                    off += int(n)

        calls = [(ptrs[i], counts[i]) for i in range(len(entries)) if counts[i] > 0]
        _ph = {} if os.environ.get("TG_DP_TIMING") else None
        t_all = time.perf_counter()
        t0 = time.perf_counter()
        self._nccl_before(rt)
        self._nccl.all_reduce_group(calls)
        self._nccl_after(rt)
        rt.sync()
        t1 = time.perf_counter()
        self.nccl_sec = t1 - t0
        self._nccl_calls = len(calls)

        sc = float(scale) if scale is not None else 1.0 / float(self.world)
        for p, g, n in entries:
            if g is None:
                _set_grad(p, np.zeros(tuple(getattr(p, "shape", (n,))), np.float32), None)
            elif sc != 1.0:
                rt.scale_inplace(g._gpu, sc)
        self.comm_sec = time.perf_counter() - t_all
        if _ph is not None:
            print("[dp-phases] backend=nccl nccl=%.4fs scale+fill=%.4fs calls=%d "
                  "zero_bytes=%d"
                  % (self.nccl_sec, self.comm_sec - self.nccl_sec, len(calls),
                     missing_total * 4), flush=True)

    # ---------- 对外接口 ----------

    def shard_indices(self, n_items, *, contiguous=True):
        """把 `0..n_items-1` 按 rank 切成互不重叠的一份（数据并行必须各卡喂不同数据）。

        * `contiguous=True`：连续切分（`[0..k)`,`[k..2k)`,…），最后一块吃掉余数；
        * `contiguous=False`：步长切分（`rank, rank+world, …`），适合样本分布不均时更均匀。

        没有这一步，"数据并行"会退化成**各卡拷同样的数据**：梯度数学上仍成立，
        但等于白花算力，且等效 batch 小于你以为的值。
        """
        n = int(n_items)
        if n <= 0:
            return []
        w = max(1, int(self.world))
        r = int(self.rank)
        if w == 1:
            return list(range(n))
        if contiguous:
            per = n // w
            start = r * per
            end = n if r == w - 1 else start + per
            return list(range(start, end))
        return list(range(r, n, w))

    def broadcast(self, params):
        """把 rank0 的权重分发给所有 rank（保证各卡起点完全一致）。"""
        if self.world <= 1:
            return
        if not params:
            return
        self._start()
        arrs = [_get_data(p) for p in params]
        total = sum(int(a.size) for a in arrs)
        if total <= 0:
            raise RuntimeError("广播的参数总长度为 0" + "。修法：确认传入的是权重张量而不是空容器")
        if not np.isfinite(np.concatenate([a.reshape(-1)[:1] for a in arrs])).all():
            raise RuntimeError(
                "广播源里有非有限值（nan/inf）。修法：检查参数是否未初始化；"
                "若用了 CUDA 张量，别读 `.data`（它是 None），要用 to_numpy()")
        self._ensure_slot(max(total * 4, 8))
        off = 0
        if self.rank == 0:
            for a in arrs:
                self._slot.buf[off:off + a.nbytes] = a.tobytes()
                off += a.nbytes
        self._barrier("bcast")
        if self.rank != 0:
            off = 0
            for p, a in zip(params, arrs):
                val = np.frombuffer(bytes(self._slot.buf[off:off + a.nbytes]),
                                    dtype=np.float32).reshape(a.shape)
                _set_data(p, val)          # ← 必须写回张量本身（旧版写的是临时数组）
                off += a.nbytes
        self._barrier("bcast2")

    def _all_reduce_host(self, entries, total, scale):
        """纯主机路径：CPU 后端（无 CUDA）时使用，语义与 GPU 路径完全一致。"""
        off = 0
        base = self.rank * total * 4
        own = np.ndarray((total,), dtype=np.float32, buffer=self._slot.buf, offset=base)
        for p, g, n in entries:
            dst = own[off:off + n]
            if g is None:
                dst.fill(0.0)
            else:
                arr = _grad_to_numpy(g)
                np.copyto(dst, np.ascontiguousarray(arr, dtype=np.float32).reshape(-1))
            off += n
        self._gen = int(getattr(self, "_gen", 0)) + 1
        self._barrier("grads")
        allv = np.ndarray((self.world, total), dtype=np.float32, buffer=self._slot.buf)
        acc = np.empty(total, dtype=np.float32)
        np.sum(allv, axis=0, out=acc)
        acc *= (float(scale) if scale is not None else 1.0 / float(self.world))
        off = 0
        for p, g, n in entries:
            val = acc[off:off + n]
            val = val.reshape(tuple(getattr(p, "shape", (n,))))
            _set_grad(p, val, g)
            off += n

    def _buffers(self, pack_elems, host_elems=None):
        """返回 (打包用设备缓冲, pinned 主机缓冲)，按需扩容并**跨步复用**。

        反复 new 会每步重付一次分配/锁定开销（3.16 GiB 的 pinned 分配很贵），
        所以缓存在实例上，只在变大时重建。

        `pack_elems` 是**分块**大小（2026-11-21 起默认 256 MB）；主机缓冲仍按
        总元素数 `host_elems` 分配 —— 那是 pinned 主机内存，不占显存。
        """
        rt = _runtime()
        need = int(host_elems if host_elems is not None else pack_elems) * 4
        b = getattr(self, "_pack", None)
        if b is None or int(b.n) < int(pack_elems):
            b = rt.empty((int(pack_elems),), dtype=np.float32)
            self._pack = b
        h = getattr(self, "_host", None)
        if h is None or int(h.nbytes) < need:
            h = rt.host_pinned(need, slot="allreduce")
            self._host = h
        return b, h

    def all_reduce_grads(self, params, scale=None):
        """梯度全归约，默认取**平均**（`scale=None`）。

        ## 两个必须遵守的约定

        1. **每个 rank 传入完全相同的 `params` 列表（顺序也要一致）**。
           归约按位置对号入座，顺序不同就是把 A 的梯度加到 B 上。
        2. **冻结的参数不要放进来**。缺失梯度会按 0 补齐（这样各 rank 的字节数一致，
           否则共享内存名字不同 ⇒ 一边创建一边等不到，直接卡死），而 0 梯度在 AdamW
           里仍会走 weight decay，冻结参数会被慢慢改掉。

        `scale`：给了就 **乘**（不是除）。要让等效 batch 变大而保持学习率语义，
        通常保持默认的"平均"，然后把 lr 按 world 放大（或保持 lr 不变、把每卡 batch 调小）。

        缺失梯度按形状补 0 而不是跳过 —— 旧版跳过会导致各 rank 的条目数不同，见上。
        """
        if self.world <= 1:
            return
        if not params:
            return
        self._start()
        entries = []
        for p in params:
            g = getattr(p, "grad", None)
            if g is not None:
                # ★ 不能假设 `g._gpu` 存在：CPU 后端下梯度是主机张量（tests/test_dist_cpu.py 守着）
                gshape = tuple(int(s) for s in getattr(g, "shape", ()) or ())
                n = int(np.prod(gshape)) if gshape else int(getattr(g, "numel", 0) or 0)
                pshape = tuple(getattr(p, "shape", (n,)))
                if int(np.prod(pshape)) != n:
                    # 形状对不上时不能静默：后面按偏移切片会整段错位
                    raise RuntimeError(
                        "梯度长度 %d 与参数形状 %s 不一致。修法：确认没有手动改过 p.grad"
                        % (n, pshape))
            else:
                n = int(p.numel())
            entries.append((p, g, n))
        total = sum(e[2] for e in entries)
        if total <= 0:
            raise RuntimeError("梯度总长度为 0" + "。修法：确认反向已经跑过、params 非空")
        if self.grad_dtype != "fp32":
            # ★ 低精度通信：256 MiB 的 fp32 打包缓冲在 bf16 下只需 128 MiB，
            #   峰值涨幅是 384 MiB（< 0.5 GiB 红线），详见 docs/DP_BF16_GRAD.md。
            self._all_reduce_lowp(entries, total, scale)
            return
        if self.backend == "nccl":
            self._all_reduce_nccl(entries, total, scale)
            return
        self._ensure_slot(max(total * 4, 8))

        # ★ 2026-11-18 性能整改（第二版）：把 321 次逐张量同步拷贝换成
        #   「GPU 内打包 → 一次 d2h(pinned) → 一次 h2d」，并把主机侧缓冲改成**页锁定**。
        #   实测：可分页 2.18/1.48 GiB/s → pinned 12.05/10.54 GiB/s（3.16 GiB 往返
        #   3.55 s → 0.56 s）。原来的瓶颈是每次 `cuMemcpyDtoH` 都要驱动逐页锁定，
        #   加上 321 次调用的固定开销（每次约 15 ms）。
        # ★ CPU 后端（没有 CUDA）时必须走纯主机路径：GPU 原语不可用。
        #   这一条由 tests/test_dist_cpu.py 守着，不能被性能优化破坏。
        try:
            rt = _runtime()
        except Exception:
            rt = None
        if rt is None:
            return self._all_reduce_host(entries, total, scale)
        # ★ 2026-11-21：打包缓冲改成**分块**（默认 256 MB，`TG_DP_PACK_MB` 可调，
        #   0 = 退回"一整块"的旧行为）。
        #   为什么：旧实现要一整块 3.16 GiB 的 pack，seq2048/848M 时把 5090 的
        #   31.4 GiB 顶到只剩 168 MiB（pool_peak_live 25.25 GiB，而 NCCL 路径
        #   只要 22.00 GiB），实测每步**多花约 90 ms**（0.497 s vs 0.409 s），
        #   而两者的通信时间几乎相同（143 vs 145 ms）——多出来的是显存贴顶后
        #   分配器的代价，不是搬运。分块只多占 `chunk`，字节数与语义不变。
        pack_mb = _env_int("TG_DP_PACK_MB", 256)
        chunk = (int(total) if pack_mb <= 0
                 else max(1, min(int(total), (pack_mb << 20) // 4)))
        pack, host = self._buffers(chunk, total)
        # ★ 把共享内存注册成页锁定 ⇒ d2h 可以直接写进去，省掉
        #   「pinned → shm」那一次 3.16 GiB 主机拷贝（实测约 0.3 s）。
        #   注册失败就退回老路（pinned + copyto），不影响正确性。
        shm_stage = rt.host_register(self._slot.buf)
        own_off = self.rank * total * 4
        _ph = {} if os.environ.get("TG_DP_TIMING") else None
        _t = [time.perf_counter()]

        def _mark(name):
            if _ph is not None:
                now = time.perf_counter()
                _ph[name] = now - _t[0]
                _t[0] = now

        # ① GPU 内打包 + ② 分块 d2h：优先**直接写进已注册的共享内存**
        #   ★ d2h 与后面 h2d 必须落在**同一段地址**里（own_off / r*total*4），
        #     否则回传的是另一块缓冲的旧数据（曾因此让梯度全 0，是
        #     「各 rank gnorm 逐位相同」这条判据当场抓到的）。
        lo = 0
        while lo < int(total):
            hi = min(int(total), lo + chunk)
            self._pack_range(rt, entries, lo, hi, pack)
            if shm_stage is not None:
                dst = ctypes.c_void_p(shm_stage.ptr.value + own_off + lo * 4)
            else:
                dst = ctypes.c_void_p(host.ptr.value + lo * 4)
            rt.d2h_into(pack, dst, nbytes=(hi - lo) * 4)
            lo = hi
        if shm_stage is not None:
            own_view = np.ndarray((total,), dtype=np.float32, buffer=self._slot.buf,
                                  offset=own_off)
        else:
            own_view = host.array(np.float32, total)
        _mark("1_pack_d2h")

        # ③ 缺梯度的位置在**主机侧**清零（不占额外显存；这类位置很少）
        off = 0
        for p, g, n in entries:
            if g is None:
                own_view[off:off + n] = 0.0
            off += n

        # ④ 写进共享内存自己那一段（若已直接 d2h 进去则跳过）
        if shm_stage is None:
            base = self.rank * total
            shm_own = np.ndarray((total,), dtype=np.float32, buffer=self._slot.buf,
                                 offset=base * 4)
            np.copyto(shm_own, own_view)
        _mark("3_shm_copy")
        self._barrier("grads")
        _mark("4_barrier")

        # ⑤ **GPU 侧求和**：把对端那段分块 h2d 上来，直接在各自的梯度缓冲上就地加。
        #   ★ 为什么不在主机上求和：3.16 GiB 的 numpy 加法实测 0.46 s，而 GPU 就地加
        #     约 0.02 s。主机只负责搬运（d2h 0.28 + h2d 0.32）——无 P2P 时这是物理成本。
        #   ★ 复用 `pack` 作为对端数据的落点（d2h 之后它的内容已经在共享内存里）。
        sc = float(scale) if scale is not None else 1.0 / float(self.world)
        for r in range(self.world):
            if r == self.rank:
                continue
            lo = 0
            while lo < int(total):
                hi = min(int(total), lo + chunk)
                if shm_stage is not None:
                    src = ctypes.c_void_p(shm_stage.ptr.value
                                          + r * total * 4 + lo * 4)
                else:                   # 未注册成功：从 shm 的正常映射拷
                    seg = np.ndarray((hi - lo,), dtype=np.float32,
                                     buffer=self._slot.buf,
                                     offset=(r * total + lo) * 4)
                    src = ctypes.c_void_p(seg.ctypes.data)
                rt.h2d_into(pack, src, nbytes=(hi - lo) * 4)
                self._add_range(rt, entries, lo, hi, pack)
                lo = hi
        _mark("5_h2d_add")

        # ⑥ 就地乘均值；没有梯度的参数补 0（与原语义一致）
        for p, g, n in entries:
            if g is not None:
                rt.scale_inplace(g._gpu, sc)
            else:
                _set_grad(p, np.zeros(tuple(getattr(p, "shape", (n,))), np.float32), None)
        _mark("6_scale")
        if _ph is not None:
            print("[dp-phases] " + "  ".join("%s=%.3fs" % (k, v) for k, v in
                                             sorted(_ph.items()))
                  + "  chunk_mb=%d" % (chunk * 4 >> 20), flush=True)

    @staticmethod
    def _pack_range(rt, entries, lo, hi, pack):
        """把 entries 在元素区间 ``[lo, hi)`` 的部分 D2D 拷进 pack（pack[0] ↔ lo）。

        支持**跨块**的张量：按区间求交后按段拷贝，偏移都是元素单位。
        """
        off = 0
        for p, g, n in entries:
            a = off
            off += n
            b = off
            if g is None or b <= lo or a >= hi:
                continue
            s = max(a, lo)
            e = min(b, hi)
            rt.copy_d2d(pack, g._gpu, (e - s) * 4,
                        dst_off=(s - lo) * 4, src_off=(s - a) * 4)

    @staticmethod
    def _add_range(rt, entries, lo, hi, pack):
        """把 pack 里区间 ``[lo, hi)`` 的数据就地加到对应梯度上（支持跨块张量）。"""
        off = 0
        for p, g, n in entries:
            a = off
            off += n
            b = off
            if g is None or b <= lo or a >= hi:
                continue
            s = max(a, lo)
            e = min(b, hi)
            seg = (e - s)
            src = pack if (s - lo) == 0 and seg == int(pack.n) else \
                rt.sub_view(pack, (seg,), s - lo)
            if (s - a) == 0 and seg == n:
                rt.add_(g._gpu, src)
            else:
                rt.add_(rt.sub_view(g._gpu, (seg,), s - a), src)

    # ---------- ZeRO 用的三个集合通信（2026-11-21） ----------
    #
    # 这三个与 `all_reduce_grads` 的区别：它们**不认参数列表**，只搬「一段已经打好的
    # 连续 fp32 缓冲」，因此与模型结构无关，可以在 CPU 上用假运行时跑 2 进程对拍
    # （`tests/zero_worker.py` 就是靠这一点，不需要两张卡）。
    #
    # 共同契约：
    #   * `count` 必须**每个 rank 完全相同**（NCCL 的 all_gather/reduce_scatter 要求
    #     等长；host 路径用同一个数拼共享内存名字，不同就是一边创建一边死等）；
    #   * `send` / `recv` 都是**设备缓冲**（有 `.ptr`）。`rt` 可注入，供 CPU 单测传假运行时。

    def _nccl_rt(self, rt):
        """取运行时：显式注入的优先（CPU 单测传假运行时），否则用全局单例。"""
        return _runtime() if rt is None else rt

    def _aux_slot(self, key, per_rank_bytes):
        """按用途名分配一块共享内存（总大小 = ``per_rank_bytes * world``），跨步复用。

        与 `_ensure_slot` 的区别：那个是 DP 梯度专用的单例（名字里带字节数），
        ZeRO 需要**几块互不干扰的**缓冲，复用同一个会因为大小交替变化而反复重建。
        这里每块按 ``key`` 独立管理，只在变大时重建。
        """
        w = int(self.world)
        need = int(per_rank_bytes)
        cache = getattr(self, "_aux", None)
        if cache is None:
            cache = self._aux = {}
        ent = cache.get(key)
        if ent is not None and ent[1] >= need:
            return ent[0]
        if ent is not None:
            try:
                ent[0].close()
            except Exception:                            # noqa: BLE001
                pass
        name = "tgai_dp_%s_%s_%d" % (self.job, key, need)
        if self.rank == 0:
            slot = shared_memory.SharedMemory(name=name, create=True, size=need * w)
        else:
            t0 = time.time()
            while True:
                try:
                    slot = shared_memory.SharedMemory(name=name, create=False)
                    break
                except FileNotFoundError:
                    if time.time() - t0 > self.timeout:
                        raise TimeoutError(
                            "共享内存 %s 未创建（ZeRO 的 %s 缓冲）" % (name, key) +
                            "。修法：确认所有 rank 都启动了、TGAI_DP_JOB 一致；"
                            "若有 rank 提前退出，其余 rank 会在这里超时")
                    time.sleep(0.002)
        self._shm.append(slot)
        cache[key] = (slot, need)
        return slot

    def all_gather_flat(self, send, recv, count, rt=None):
        """``recv[r*count : (r+1)*count] = 第 r 个 rank 的 send``（fp32）。

        ZeRO-1/2 更新完参数后，用它把每 rank 只更新过的那 1/w 收回来。
        """
        count = int(count)
        if count <= 0:
            return
        self._start()
        rtk = self._nccl_rt(rt)
        w = int(self.world)
        if self.backend == "nccl":
            self._nccl_before(rtk)
            self._nccl.all_gather_flat(int(send.ptr), int(recv.ptr), count)
            self._nccl_after(rtk)
            rtk.sync()
            return
        span = w * count
        slot = self._aux_slot("allgather", count * 4)     # 总大小 = w*count*4
        hp = rtk.host_pinned(span * 4, slot="zero_allgather")
        arr = hp.array(np.float32, span)
        shm = np.ndarray((span,), np.float32, buffer=slot.buf)
        rtk.d2h_into(send, ctypes.c_void_p(hp.ptr.value), nbytes=count * 4)
        lo = self.rank * count
        shm[lo:lo + count] = arr[:count]                  # ★ 只写自己那一段
        self._barrier("zg")
        np.copyto(arr, shm)
        rtk.h2d_into(recv, ctypes.c_void_p(hp.ptr.value), nbytes=span * 4)
        # ★ 第二个 barrier 不能省：快的 rank 若先进入下一块的写入，
        #   会把慢 rank 正在读的那一段覆盖掉（静默算错，不是卡死）。
        self._barrier("zg2")

    def reduce_scatter_flat(self, send, recv, count, scale=None, rt=None):
        """``recv[i] = sum_r send[r*count + i] * scale``（fp32，默认乘 ``1/world``）。

        ZeRO-2 的梯度归约：``send`` 是每 rank 打好的 ``world*count`` 元素，
        第 ``r`` 段最终归第 ``r`` 个 rank 所有 ⇒ 每 rank 只**收** ``count`` 个，
        所以通信量是同一批数据 all_reduce 的**一半**。
        """
        count = int(count)
        if count <= 0:
            return
        self._start()
        rtk = self._nccl_rt(rt)
        w = int(self.world)
        sc = float(scale) if scale is not None else 1.0 / float(w)
        if self.backend == "nccl":
            self._nccl_before(rtk)
            self._nccl.reduce_scatter_flat(int(send.ptr), int(recv.ptr), count)
            self._nccl_after(rtk)
            rtk.sync()
            rtk.scale_inplace(recv, sc)
            return
        # host：每个 rank 把**自己那 world 段**全部写进共享内存，
        # 一次 barrier 之后各自按 (world, world*count) 视图沿 rank 轴求和。
        span = w * count
        slot = self._aux_slot("reduce_scatter", span * 4)   # 总大小 = w*span*4
        hp = rtk.host_pinned(span * 4, slot="zero_rs")
        arr = hp.array(np.float32, span)
        shm = np.ndarray((w, span), np.float32, buffer=slot.buf)
        rtk.d2h_into(send, ctypes.c_void_p(hp.ptr.value), nbytes=span * 4)
        np.copyto(shm[self.rank], arr)                    # ★ 只写自己那一行
        self._barrier("zr")
        acc = np.empty(count, np.float32)
        np.sum(shm[:, self.rank * count:(self.rank + 1) * count], axis=0, out=acc)
        acc *= sc
        np.copyto(arr[:count], acc)
        rtk.h2d_into(recv, ctypes.c_void_p(hp.ptr.value), nbytes=count * 4)
        self._barrier("zr2")

    def all_reduce_array(self, arr, rt=None):
        """把一个小 float32 主机数组在所有 rank 上**求和**，返回同形状数组。

        只用于「控制/统计量」（ZeRO-2 的逐张量平方和，MB 级），**不要**拿它搬梯度。
        求和顺序与 `_all_reduce_host` 完全一致（`np.sum(axis=0)` 沿 rank 轴），
        因此同一组输入在所有 rank 上给出**逐位相同**的结果 ——
        ZeRO-2 的 gnorm 能与不分片版本逐位一致，靠的就是这一条。
        """
        shape = np.shape(arr)
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32)).reshape(-1)
        n = int(a.size)
        if n <= 0 or self.world <= 1:
            return a.reshape(shape)
        self._start()
        slot = self._aux_slot("array", n * 4)             # 总大小 = w*n*4
        shm = np.ndarray((self.world, n), np.float32, buffer=slot.buf)
        shm[self.rank] = a
        self._barrier("za")
        out = np.empty(n, np.float32)
        np.sum(shm, axis=0, out=out)
        self._barrier("za2")
        return out.reshape(shape)

    # ---------- 低精度梯度通信（TG_DP_GRAD_DTYPE=bf16/fp16） ----------
    #
    # 三条路径共用同一套语义：「cast → 低精度 all-reduce → cast 回 fp32 → ×1/world」：
    #
    #   pack32 = 本 rank 梯度分块打包（缺梯度按 0）
    #   pack16 = cast(pack32)            ← 只有这 1/2 的字节走集合通信
    #   all_reduce(pack16, dtype)        ← NCCL 用 bf16/fp16 类型；host 走共享内存交换
    #   pack32 = dequant(pack16) * 1/world
    #   写回各 p.grad（**覆盖**，不是累加：p.grad 里原来那份本 rank 梯度已经打包过）
    #
    # 参数本身**永远 fp32**；`world=1` 在 `all_reduce_grads` 开头就返回，
    # 所以单卡的梯度逐位不变（验收硬要求）。

    def _set_comm_bytes(self, total):
        """记录本步通信有效载荷字节（验收口径：3.16 GiB → 1.58 GiB 的算术值）。"""
        self.comm_rank_bytes = int(total) * _GRAD_ITEMSIZE[self.grad_dtype]
        self.comm_rank_bytes_fp32 = int(total) * 4
        return self.comm_rank_bytes

    def _ef_host(self, total):
        """CPU 路径的误差反馈影子（主机内存，total×4 字节；未开 EF 返回 None）。"""
        if not self.grad_ef:
            return None
        buf = self._ef_host_buf
        if buf is None or int(buf.size) != int(total):
            self._ef_host_buf = np.zeros(int(total), dtype=np.float32)
            buf = self._ef_host_buf
        return buf

    def _ef_dev(self, rt, total):
        """CUDA 路径的误差反馈影子（显存，total×4 字节；未开 EF 返回 None）。"""
        if not self.grad_ef:
            return None
        buf = self._ef_dev_buf
        if buf is None or int(buf.n) != int(total):
            self._ef_dev_buf = rt.zeros((int(total),), dtype=np.float32)
            buf = self._ef_dev_buf
        return buf

    def _nccl_dtype(self):
        """低精度通信的 ncclDataType_t。fp32 返回 None（调用方走老的无类型路径）。"""
        from . import dist_nccl as _N
        if self.grad_dtype == "bf16":
            return int(_N.NCCL_BFLOAT16)
        if self.grad_dtype == "fp16":
            return int(_N.NCCL_FLOAT16)
        return None

    def _all_reduce_lowp(self, entries, total, scale):
        """低精度归约入口：按「有没有 CUDA 运行时」分派。"""
        self._set_comm_bytes(total)
        sc = float(scale) if scale is not None else 1.0 / float(self.world)
        rt = None
        try:
            rt = _runtime()
        except Exception:                                # noqa: BLE001
            rt = None
        if rt is None:
            self._all_reduce_lowp_cpu(entries, total, sc)
        else:
            self._all_reduce_lowp_gpu(rt, entries, total, sc)
        if os.environ.get("TG_DP_TIMING"):
            print("[dp-lp] dtype=%s comm_rank_bytes=%d (fp32=%d, ratio=%.3f) backend=%s"
                  % (self.grad_dtype, int(self.comm_rank_bytes),
                     int(self.comm_rank_bytes_fp32),
                     float(self.comm_rank_bytes)
                     / max(1, int(self.comm_rank_bytes_fp32)), self.backend),
                  flush=True)

    def _all_reduce_lowp_cpu(self, entries, total, sc):
        """无 CUDA 时的 numpy 直通路径（语义与 GPU 路径一致，用于 CPU 对拍/单测）。"""
        total = int(total)
        esz = _GRAD_ITEMSIZE[self.grad_dtype]
        self._ensure_slot(max(total * esz, 8))
        resid = self._ef_host(total)
        own_off = int(self.rank) * total * esz
        own_bits = np.ndarray((total,), dtype=np.uint16, buffer=self._slot.buf,
                              offset=own_off)
        off = 0
        for p, g, n in entries:
            n = int(n)
            if g is None:
                own_bits[off:off + n] = 0
                if resid is not None:
                    resid[off:off + n] = 0.0
            else:
                arr = _grad_to_numpy(g)
                if arr is None or int(arr.size) != n:
                    raise RuntimeError(
                        "低精度归约读不出梯度 #%d 的数值（长度 %s，期望 %d）"
                        % (off, None if arr is None else int(arr.size), n) +
                        "。修法：确认该 rank 的 .grad 是有效张量（CUDA 张量用 to_numpy）；"
                        "或先设 TG_DP_GRAD_DTYPE=fp32 排除低精度路径")
                v = arr.reshape(-1).astype(np.float32, copy=True)
                if resid is not None:
                    v += resid[off:off + n]
                bits = quantize_to_bits(v, self.grad_dtype)
                own_bits[off:off + n] = bits
                if resid is not None:
                    # 残差 = g_eff − dequant(quant(g_eff))
                    resid[off:off + n] = v - dequantize_from_bits(bits, self.grad_dtype)
            off += n
        self._barrier("grads")
        allb = np.ndarray((int(self.world), total), dtype=np.uint16, buffer=self._slot.buf)
        acc = np.zeros(total, np.float32)
        for r in range(int(self.world)):
            acc += dequantize_from_bits(allb[r], self.grad_dtype)
        acc *= np.float32(sc)
        off = 0
        for p, g, n in entries:
            n = int(n)
            val = acc[off:off + n].reshape(tuple(getattr(p, "shape", (n,))))
            _set_grad(p, val, g)
            off += n

    def _all_reduce_lowp_gpu(self, rt, entries, total, sc):
        """CUDA 路径：分块「打包 → cast → 集合通信 → dequant → ×1/world → 写回」。"""
        total = int(total)
        esz = _GRAD_ITEMSIZE[self.grad_dtype]
        chunk = _lowp_chunk_elems(total)
        b = self._lp_buf
        if b is None or int(b[0].n) < chunk:
            b = (rt.empty((chunk,), dtype=np.float32),      # pack32：本 rank 打包
                 rt.empty((chunk,), dtype=np.uint16),       # pack16：通信载荷
                 rt.empty((chunk,), dtype=np.float32))      # scratch32：对端 dequant / EF
            self._lp_buf = b
        pack32, pack16, scratch32 = b
        if self._zero1 is None:
            self._zero1 = rt.zeros((1,), dtype=np.float32)
        resid = self._ef_dev(rt, total)
        use_nccl = (self.backend == "nccl")
        if use_nccl and self._nccl is None:
            raise RuntimeError("NCCL 后端未初始化" + "。修法：先调用一次 broadcast()/_start()")
        total16 = total * esz
        shm_stage = None
        own_view = None
        own_off = 0
        if not use_nccl:
            self._ensure_slot(max(total16, 8))
            own_off = int(self.rank) * total16
            shm_stage = rt.host_register(self._slot.buf)
            if shm_stage is None:
                raise RuntimeError(
                    "低精度归约要求共享内存能被 cuMemHostRegister 注册成页锁定内存" +
                    "。修法：设 TG_DP_BACKEND=nccl 走设备侧归约（推荐），"
                    "或 TG_DP_GRAD_DTYPE=fp32 退回逐位不变的老路径")
            own_view = np.ndarray((total,), dtype=np.uint16, buffer=self._slot.buf,
                                  offset=own_off)
        nccl_dtype = self._nccl_dtype() if use_nccl else None
        if use_nccl and self.grad_dtype == "bf16":
            ver = int(getattr(self._nccl_lib, "version", 0) or 0)
            if ver and ver < 21000:
                raise RuntimeError(
                    "当前 libnccl %s 不支持 bf16 集合通信（需要 >= 2.10）"
                    % (self._nccl_lib.version_str(),) +
                    "。修法：升级 NCCL（>=2.10）；或改用 TG_DP_GRAD_DTYPE=fp16/fp32")
        for lo in range(0, total, chunk):
            hi = min(total, lo + chunk)
            n = hi - lo
            self._zero_pack_range(rt, lo, hi, pack32)
            self._pack_range(rt, entries, lo, hi, pack32)
            if resid is not None:
                self._add_resid_range(rt, entries, lo, hi, pack32, resid)
            _cast_f32_to_16(rt, self.grad_dtype, pack32, pack16, n)
            if resid is not None:
                # 残差 = g_eff − dequant(quant(g_eff))；缺梯度区间 0−0=0（残差重置）
                _cast_16_to_f32(rt, self.grad_dtype, pack16, scratch32, n)
                rt.scale_inplace(rt.sub_view(scratch32, (n,), 0), -1.0)
                rt.add_(rt.sub_view(scratch32, (n,), 0), rt.sub_view(pack32, (n,), 0))
                rt.copy_d2d(resid, scratch32, nbytes=n * 4, dst_off=lo * 4)
            if use_nccl:
                self._nccl_before(rt)
                self._nccl.all_reduce_group([(int(pack16.ptr), n)], dtype=nccl_dtype)
                self._nccl_after(rt)
                rt.sync()
                # 归约结果（低精度 sum）→ fp32 → ×1/world → 覆盖写回各 grad
                _cast_16_to_f32(rt, self.grad_dtype, pack16, pack32, n)
            else:
                # host：把自己的低精度载荷 d2h 进共享内存，缺梯度区间在主机侧清零
                rt.d2h_into(pack16, ctypes.c_void_p(
                    int(shm_stage.ptr.value) + own_off + lo * esz), nbytes=n * esz)
                self._zero_host_missing(entries, lo, hi, own_view)
                self._barrier("grads")
                _cast_16_to_f32(rt, self.grad_dtype, pack16, pack32, n)
                for r in range(int(self.world)):
                    if r == int(self.rank):
                        continue
                    rt.h2d_into(pack16, ctypes.c_void_p(
                        int(shm_stage.ptr.value) + r * total16 + lo * esz),
                        nbytes=n * esz)
                    _cast_16_to_f32(rt, self.grad_dtype, pack16, scratch32, n)
                    rt.add_(rt.sub_view(pack32, (n,), 0), rt.sub_view(scratch32, (n,), 0))
            rt.scale_inplace(rt.sub_view(pack32, (n,), 0), sc)
            self._copy_range_to_grads(rt, entries, lo, hi, pack32)
            if not use_nccl:
                # 下一个分块要覆盖共享内存：必须等对端读完本块（否则静默读到半新半旧）
                self._barrier("grads2")
        # 缺梯度：与 fp32 路径一致，补**全 0** 的梯度（不是跳过）
        for p, g, n in entries:
            if g is None:
                _set_grad(p, np.zeros(tuple(getattr(p, "shape", (n,))), np.float32), None)

    @staticmethod
    def _zero_pack_range(rt, lo, hi, pack32):
        """把 pack32 的分块区间清零（缺梯度位置不能留上一块的旧值）。"""
        n = int(hi) - int(lo)
        if n > 0:
            rt.scale_inplace(rt.sub_view(pack32, (n,), 0), 0.0)

    @staticmethod
    def _add_resid_range(rt, entries, lo, hi, pack32, resid):
        """EF：把残差加到**有梯度**的参数区间上（缺梯度不参与，不能把残差当梯度发出去）。"""
        off = 0
        for p, g, n in entries:
            a = off
            off += int(n)
            b = off
            if g is None or b <= lo or a >= hi:
                continue
            s = max(a, lo)
            e = min(b, hi)
            m = e - s
            rt.add_(rt.sub_view(pack32, (m,), s - lo), rt.sub_view(resid, (m,), s))

    @staticmethod
    def _copy_range_to_grads(rt, entries, lo, hi, pack):
        """把 pack 的分块区间**覆盖**写回各梯度（不是 add：本 rank 梯度已打包）。

        ★ `dst_off=(s-a)*4` 不能写成 0：分块边界可能落在**一个参数中间**（256 MB 的块
          与 321 个参数几乎必然错位），只写 param 头部会把后半段留在"本 rank 原值"，
          结果是"一半归约过、一半没归约"，而且不报错。这条由假运行时单测
          `test_gpu_pipeline_multi_chunk_matches_reference` 钉住。
        """
        off = 0
        for p, g, n in entries:
            a = off
            off += int(n)
            b = off
            if g is None or b <= lo or a >= hi:
                continue
            s = max(a, lo)
            e = min(b, hi)
            rt.copy_d2d(g._gpu, pack, nbytes=(e - s) * 4, dst_off=(s - a) * 4,
                        src_off=(s - lo) * 4)

    @staticmethod
    def _zero_host_missing(entries, lo, hi, own_view):
        """host 路径：本 rank 缺失的梯度在共享内存里必须是 0（各 rank 都发同样多的字节）。"""
        off = 0
        for p, g, n in entries:
            a = off
            off += int(n)
            b = off
            if g is not None or b <= lo or a >= hi:
                continue
            s = max(a, lo)
            e = min(b, hi)
            own_view[s:e] = 0

    def stats(self):
        """给日志/RESULT 用的后端信息（**实测值，不是配置值**）。"""
        out = {"backend_requested": self.backend_requested,
               "backend": self.backend,
               "world": int(self.world), "rank": int(self.rank),
               "comm_sec": self.comm_sec, "nccl_sec": self.nccl_sec,
               "nccl_calls": int(self._nccl_calls),
               "grad_dtype": self.grad_dtype,
               "grad_ef": bool(self.grad_ef),
               "comm_rank_bytes": self.comm_rank_bytes,
               "comm_rank_bytes_fp32": self.comm_rank_bytes_fp32}
        if self.overlap_stats is not None:
            out["overlap"] = self.overlap_stats
        if self._nccl_lib is not None:
            out["nccl_lib"] = self._nccl_lib.path
            out["nccl_version"] = self._nccl_lib.version_str()
            out["nccl_stream"] = "own" if self._nccl_own_stream else "rt._stream"
        return out

    def _close_nccl(self):
        # ★ world=1 / host 后端 / NCCL 从未初始化时**必须完全不碰运行时**：
        #   否则一个「造出来只为取 world 的单卡对象」在 close() 里也会去 _cuda()，
        #   在无卡机器上是没必要的副作用（单测 test_world_one_is_noop 守着）。
        if (self._nccl is None and self._nccl_events is None
                and not self._nccl_own_stream and self._nccl_stream is None
                and self._ov_stream is None):
            return
        if self._nccl is not None:
            try:
                self._nccl.close()
            except Exception:                            # noqa: BLE001
                pass
            self._nccl = None
        rt = None
        try:
            rt = _runtime()
        except Exception:                                # noqa: BLE001
            pass
        if rt is not None:
            cu = getattr(rt, "cu", None)
            if cu is not None:
                if self._nccl_events is not None:
                    destroy_ev = getattr(cu, "cuEventDestroy_v2", None) \
                        or getattr(cu, "cuEventDestroy", None)
                    for ev in self._nccl_events:
                        try:
                            if destroy_ev is not None:
                                destroy_ev(ev)
                        except Exception:                # noqa: BLE001
                            pass
                if self._nccl_own_stream and self._nccl_stream is not None:
                    destroy_st = getattr(cu, "cuStreamDestroy_v2", None) \
                        or getattr(cu, "cuStreamDestroy", None)
                    try:
                        if destroy_st is not None:
                            destroy_st(self._nccl_stream)
                    except Exception:                    # noqa: BLE001
                        pass
                # 重叠归约自己的非阻塞流（可能只被重叠路径创建过）
                if self._ov_stream is not None:
                    destroy_st2 = getattr(cu, "cuStreamDestroy_v2", None) \
                        or getattr(cu, "cuStreamDestroy", None)
                    try:
                        if destroy_st2 is not None:
                            destroy_st2(self._ov_stream)
                    except Exception:                    # noqa: BLE001
                        pass
        self._nccl_events = None
        self._nccl_stream = None
        self._nccl_own_stream = False
        self._ov_stream = None

    # ---------- 反向重叠归约（可选；world<=1 / TG_DP_OVERLAP=0 时是快速空操作） ----------

    def _make_overlap_stream(self, rt):
        """给重叠归约开一条**非阻塞**流（与计算流靠 event 互相同步）。

        必须 `CU_STREAM_NON_BLOCKING`：ORV 的计算流是 legacy NULL 流，默认（blocking）
        流会与它**隐式互相同步**，重叠直接归零 —— 在真机上的表现就是"开了重叠但
        时间一点没变"，而且不报任何错。这是本功能最容易静默失效的一处。
        """
        from . import dp_overlap

        if self._ov_stream is not None:
            return self._ov_stream
        cu = getattr(rt, "cu", None)
        if cu is None:
            raise dp_overlap.OverlapUnavailable("运行时没有 CUDA 符号表" +
                                           "。修法：重叠归约只支持 CUDA；CPU 后端请设 TG_DP_OVERLAP=0")
        try:
            pvoid = ctypes.POINTER(ctypes.c_void_p)
            create = rt._bind_sym("cuStreamCreate", [pvoid, ctypes.c_uint])
            if create is None:
                raise dp_overlap.OverlapUnavailable("驱动缺少 cuStreamCreate" +
                                           "。修法：升级显卡驱动；或设 TG_DP_OVERLAP=0 用不重叠路径")
            st = ctypes.c_void_p()
            rc = int(create(ctypes.byref(st), 1))          # 1 = NON_BLOCKING
            if rc != 0 or not st.value:
                raise dp_overlap.OverlapUnavailable(
                "cuStreamCreate(非阻塞) rc=%d" % rc +
                "。修法：按 rc 查 CUDA 错误；急着训练就设 TG_DP_OVERLAP=0")
        except dp_overlap.OverlapUnavailable:
            raise
        except Exception as exc:                           # noqa: BLE001
            raise dp_overlap.OverlapUnavailable(
                "建重叠流失败 %r" % (exc,) +
                "。修法：按上面的原始异常排查；或设 TG_DP_OVERLAP=0 退回不重叠路径")
        self._ov_stream = st
        return st

    def begin_overlap(self, params, order=None, scale=None):
        """开始「反向即归约」，返回会话（`world<=1`、开关关闭或不可用时返回 **None**）。

        调用方契约（`v3_pretrain` 已按此实现）::

            ov = dp.begin_overlap(params)      # 必须在 loss.backward() **之前**
            loss.backward()
            st = dp.finish_overlap()           # None ⇒ 请退回 dp.all_reduce_grads(params)

        `order`：参数的就绪顺序（下标排列）。不给就用默认倒序；知道结构的一方
        （`v3_pretrain`）可以给「head 最前、emb 最后」，见 `dp_overlap.build_plan`。
        `scale`：默认 1/world（取平均，与 `all_reduce_grads` 一致）。

        ★ 会话**不一定**真的重叠：图里有梯度检查点、已有 `.grad`（梯度累积）、
          传输初始化失败……都会让它在反向开始时退回不重叠路径。届时
          `finish_overlap()` 返回 None，调用方照常跑 `all_reduce_grads`，
          `dp.overlap_stats["reason"]` 里写着为什么。
        """
        from . import dp_overlap

        if self.world <= 1:
            return None                        # ★ 单卡零行为变化
        if not dp_overlap.overlap_enabled():
            return None
        if self.grad_ef:
            # EF 的残差要在**打包缓冲**里结算（见 _add_resid_range），重叠路径没有那个
            # 时刻：stage 一进桶就发集合通信了。所以 EF 开着时整步走不重叠路径。
            if not self._ef_ov_warned:
                self._ef_ov_warned = True
                print("[dp] WARN TG_DP_GRAD_EF=1 需要不重叠路径（残差在打包缓冲里结算）"
                      "⇒ 本步及后续都走 all_reduce_grads", flush=True)
            return None
        if self.grad_dtype != "fp32":
            # ★ 2026-11-21 GPU 实测（2×4090）：低精度 + 反向重叠这条组合**数值不对**。
            #   隔离实验（seq512/3 步/同一份数据）：
            #     overlap=0 bf16  gnorm = 10.741 / 9.809 / 9.228   （= 逐位对拍的正确答案）
            #     overlap=1 bf16  gnorm = 49.387 / 46.842 / 45.398  （×4.6，两个 rank 仍然相同）
            #   非重叠低精度路径在 GPU 上**逐位**对拍通过（maxdiff=0.0，
            #   nccl bf16 / host bf16 / nccl fp32 三组全过），所以这里退回它，
            #   不要用一个"loss 看起来正常"的错误路径（Adam 近似尺度不变，gnorm 错了
            #   loss 也照样降 —— 这正是最危险的一类 bug）。
            #   修好之前**不允许**静默启用；要复现实验可显式 TG_DP_GRAD_DTYPE=fp32。
            if not getattr(self, "_lowp_ov_warned", False):
                self._lowp_ov_warned = True
                print("[dp] WARN TG_DP_GRAD_DTYPE=%s 与反向重叠组合的数值验证未通过"
                      "（GPU 实测 gnorm 偏大 4.6×）⇒ 低精度一律走不重叠路径"
                      "（该路径已在 GPU 上逐位对拍通过）" % (self.grad_dtype,),
                      flush=True)
            return None
        if self._ov is not None:
            raise RuntimeError(
                "上一个重叠会话还没 finish_overlap()/abort_overlap()" +
                "。修法：每步成对调用；异常路径要 abort_overlap()")
        if not params:
            return None
        self._start()
        try:
            rt = _runtime()
        except Exception:                                  # noqa: BLE001
            return None
        if getattr(rt, "_capturing", False):
            # CUDA Graph 捕获期间不允许集合通信（dist_nccl 的已有约定）
            return None
        # ① 按 id 去重：tied 权重会被 parameters() 列两次；重复会让同一份梯度被归约两次
        seen, uniq, deduped = set(), [], 0
        for p in params:
            if id(p) in seen:
                deduped += 1
                continue
            seen.add(id(p))
            uniq.append(p)
        # ② 静态布局（各 rank 必须逐字节相同 ⇒ 只依赖 params 的顺序与大小）
        elems = [int(p.numel()) for p in uniq]
        try:
            plan = dp_overlap.build_plan(elems, order=order)
        except Exception as exc:                           # noqa: BLE001
            print("[dp] WARN 重叠归约的桶布局失败（%r）⇒ 退回不重叠路径" % (exc,),
                  flush=True)
            return None
        sc = (1.0 / float(self.world)) if scale is None else float(scale)
        # ③ 传输
        try:
            if self.backend == "nccl":
                st = self._make_overlap_stream(rt)
                transport = dp_overlap.NcclBucketTransport(
                    self._nccl, st, rt, sc, slots=dp_overlap.resolve_slots(),
                    grad_dtype=self.grad_dtype)
            else:
                # ★ 2026-11-21 实测后**禁用** host 重叠传输：它的后台线程一个桶都不处理。
                #   证据（2×4090, --dp 2, mode=host, 13 桶, 321 参数）：
                #     [dp-ov] comm=0.00ms exposed=0.03ms pending_zeroed=True  ← 看起来健康
                #     诊断 per_bucket_ms=None                                ← 线程没跑过
                #     两 rank gnorm 相差 1e-2（相对），量级与"没归约的本地梯度"一致
                #   即：梯度根本没被归约，数据并行静默退化成各卡各训。
                #   宁可报不可用、退回不重叠路径（正确），也不许静默少做工作。
                raise dp_overlap.OverlapUnavailable(
                    "host 后端的重叠传输（HostBucketTransport）实测不工作："
                    "后台线程一个桶都没处理（comm=0.00ms、per_bucket_ms=None），"
                    "梯度不会被归约" +
                    "。修法：设 TG_DP_BACKEND=nccl 再用重叠（NCCL 路径实测有真实通信时间），"
                    "或设 TG_DP_OVERLAP=0 走不重叠路径（正确）")
        except dp_overlap.OverlapUnavailable as exc:
            print("[dp] WARN 重叠归约不可用（%s）⇒ 退回不重叠路径" % (exc,), flush=True)
            return None
        except Exception as exc:                           # noqa: BLE001
            print("[dp] WARN 重叠归约初始化异常（%r）⇒ 退回不重叠路径" % (exc,),
                  flush=True)
            return None
        sess = dp_overlap.OverlapSession(
            uniq, plan, transport, world=self.world, scale=sc,
            zero_grad=_zero_grad_of, deduped=deduped, tag=str(self.job))
        self.overlap_stats = {"active": False, "reason": "in_flight",
                              "mode": transport.kind,
                              "buckets": plan.nbuckets,
                              "max_bucket_mb": plan.summary()["max_bucket_mb"]}
        self._ov = sess
        import tgai_go.train as T
        T._set_bwd_overlap(sess)
        return sess

    def finish_overlap(self):
        """结束重叠归约。返回 stats；**返回 None 表示请调用方跑老路径**。"""
        sess = self._ov
        if sess is None:
            return None
        self._ov = None
        import tgai_go.train as T
        T._set_bwd_overlap(None)
        try:
            st = sess.finish()
        finally:
            sess.abort()                    # 幂等：关流/事件/桶缓冲
        self.overlap_stats = st
        if not st.get("active"):
            # 会话在反向开始时退回了不重叠路径（检查点/梯度累积/传输失败……）
            return None
        self.comm_sec = st.get("exposed_sec")      # 表里的 comm 列 = **暴露**时间
        self.nccl_sec = st.get("comm_sec")         # 真实通信忙时
        return st

    def abort_overlap(self):
        """异常路径：卸下钩子、关传输（不归约）。"""
        sess, self._ov = self._ov, None
        if sess is None:
            return
        import tgai_go.train as T
        T._set_bwd_overlap(None)
        sess.abort()

    def close(self):
        """释放共享内存与 NCCL 通信器。

        ★ **不要删自己落下的 barrier 标记文件**（2026-11-13 修）。
          旧实现在这里 `os.remove("bar_*.%d")`，于是**跑得快的 rank 一收尾**就把标记删了，
          而慢的 rank 还在 `_wait_for` 等那个文件 ⇒ 永久卡死（实测 3 次里挂 1 次）。
          残留标记由下次启动时 rank0 在 `ready` 之前统一清理，所以留着是安全的。
        """
        self._close_nccl()
        for s in self._shm:
            try:
                s.close()
            except Exception:                        # noqa: BLE001
                pass
            try:
                if self.rank == 0:
                    s.unlink()
            except Exception:                        # noqa: BLE001
                pass
        self._shm = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
