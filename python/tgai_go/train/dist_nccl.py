# -*- coding: utf-8 -*-
"""NCCL 后端（**纯 ctypes 直连 libnccl.so**，不引入 torch / C 扩展）。

本模块只做三件事，全部与「怎么用」无关：

1. **找库**：``find_libnccl()`` 按固定顺序找 `libnccl.so`，并把来源写清楚
   （环境变量 → ``ctypes.util.find_library`` → torch/nvidia pip 包自带的副本 →
   ``LD_LIBRARY_PATH`` / 常见系统目录）。找不到就抛 :class:`NcclError`，
   **不会**悄悄退回主机路径 —— 回退决策在 :mod:`tgai_go.train.dist` 里，且只在
   显式配置 ``TG_DP_BACKEND=auto`` 时发生，并打印 WARNING。
2. **握手**：``unique_id_to_bytes`` / ``unique_id_from_bytes`` +
   ``write_unique_id`` / ``wait_unique_id``。rank0 生成 id 写文件，其余 rank 等文件；
   写入用「临时文件 + ``os.replace``」，读者不可能看到半截 id。
3. **集合通信**：:class:`NcclComm` 封装 ``ncclCommInitRank`` / ``ncclAllReduce`` /
   ``ncclGroupStart|End`` / ``ncclCommDestroy``，并在初始化后**校验**
   ``ncclCommCount`` / ``ncclCommUserRank`` 与调用方声明一致 —— rank 数不符属于
   「配错了也不报错」的经典事故（会卡死或算错），这里必须当场抛。

## ZeRO 需要的两个额外集合通信（2026-11-21 加）

| 符号 | 用途 | 语义 |
| --- | --- | --- |
| ``ncclAllGather`` | ZeRO-1/2 更新后把参数分片收回 | ``sendcount`` 指**每 rank 发出的元素数**；``recv`` 里按 rank 顺序排列 |
| ``ncclReduceScatter`` | ZeRO-2 的梯度归约 | ``recvcount`` 指**每 rank 收到的元素数**；``send`` 按 rank 排成 ``world`` 段 |

两者都是 OPTIONAL 绑定：真 NCCL（>=2.0）都有，但只要库是残缺的，就应当在**第一次用到**
时报 ``NcclUnavailable``（带修法），而不是 ``AttributeError``。
**注意**：这两个原语都要求**每 rank 的 count 相同**。ORV 的 ZeRO 分片按整张量切，
各 rank 的字节数不一定相等 ⇒ 调用方必须自己 padding 到 ``cap = max(shard_elems)``。

## 为什么用 ctypes 而不是 torch

ORV 全链路是 ctypes（CUDA driver + cuBLAS + NVRTC），多卡只是「每个 rank 一个进程」。
引入 torch 只为了拿它的 NCCL 绑定会把依赖面扩大一个数量级，而且 torch 的
``ProcessGroupNCCL`` 自带线程/队列模型，与本模块「同步、按桶归约」的语义对不上。

## 已知边界（不要当能力）

* 只支持 **NCCL >= 2.0**（``ncclCommInitRank`` 的起点）；低于此版本直接报错。
* 只做 ``Sum``/``float32``（ORV 梯度永远是 fp32 连续缓冲）。
* 不处理 ``ncclUniqueId`` 的跨机分发 —— 本模块的 rendezvous 用**同机共享目录**的文件，
  与 ORV 现有 ``dist.py`` 的屏障同一套假设。
* 不做 CUDA Graph 捕获：捕获期间不允许调用集合通信（调用方负责）。
"""
import ctypes
import ctypes.util
import glob
import os
import sys
import time

__all__ = [
    "NcclError",
    "NcclUnavailable",
    "NcclVersionMismatch",
    "NcclRankMismatch",
    "LibNccl",
    "NcclComm",
    "find_libnccl",
    "unique_id_to_bytes",
    "unique_id_from_bytes",
    "write_unique_id",
    "wait_unique_id",
    "UNIQUE_ID_BYTES",
    "MIN_VERSION_CODE",
    "version_string",
]

#: `ncclUniqueId` 在所有 NCCL 2.x 里都是 128 字节的不透明结构。
UNIQUE_ID_BYTES = 128

#: 最低支持的 NCCL 版本（``major*10000 + minor*100 + patch``）。2.0.0 起有 ncclCommInitRank。
#: ★ 编码是 ``major*10000 + minor*100 + patch``：2.0.0 对应 **20000**，不是 2000
#:   （2000 会被解成 0.20.0，等于把所有版本都放过）。
MIN_VERSION_CODE = 20000

#: ncclDataType_t
NCCL_FLOAT32 = 7
#: ncclRedOp_t
NCCL_SUM = 0

_LIB_NAMES = (
    "libnccl.so.2",
    "libnccl.so",
    "libnccl.so.1",
    "libnccl.so.2.0.0",
    "nccl.dll",
    "libnccl.dylib",
)


class NcclError(RuntimeError):
    """NCCL 不可用/调用失败。所有消息都必须带「修法：」。"""


class NcclUnavailable(NcclError):
    """找不到动态库或缺少必需符号。"""


class NcclVersionMismatch(NcclError):
    """库版本低于 :data:`MIN_VERSION_CODE`。"""


class NcclRankMismatch(NcclError):
    """``ncclCommCount`` / ``ncclCommUserRank`` 与调用方声明不一致。"""


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * UNIQUE_ID_BYTES)]


def unique_id_to_bytes(uid):
    """``ctypes`` 结构 → 128 字节（写文件/跨进程传递用）。"""
    return bytes(bytearray(ctypes.string_at(ctypes.byref(uid), UNIQUE_ID_BYTES)))


def unique_id_from_bytes(raw):
    """128 字节 → ``ctypes`` 结构。长度不符**立即报错**（不截断、不补零）。"""
    raw = bytes(raw)
    if len(raw) != UNIQUE_ID_BYTES:
        raise NcclError(
            "NCCL unique id 长度不符：期望 %d 字节，实际 %d 字节"
            % (UNIQUE_ID_BYTES, len(raw)) +
            "。修法：rendezvous 文件被截断或写成了文本；删掉 TGAI_DP_DIR 下该 job 的"
            " nccl_id 文件后重跑（rank0 会重新生成）")
    uid = _NcclUniqueId()
    ctypes.memmove(ctypes.byref(uid), raw, UNIQUE_ID_BYTES)
    return uid


def _atomic_write(path, data):
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_unique_id(dirpath, raw):
    """rank0 用：原子写入 ``<dir>/nccl_id``。"""
    if len(raw) != UNIQUE_ID_BYTES:
        raise NcclError(
            "写出的 NCCL unique id 长度 %d != %d" % (len(raw), UNIQUE_ID_BYTES) +
            "。修法：unique id 必须来自 ncclGetUniqueId（不要自己拼字节）")
    os.makedirs(dirpath, exist_ok=True)
    _atomic_write(os.path.join(dirpath, "nccl_id"), raw)


def wait_unique_id(dirpath, timeout=600.0, poll=0.002):
    """其余 rank 用：等 ``<dir>/nccl_id`` 出现并返回 128 字节。

    超时消息要能定位问题 —— 多卡现场 90% 的「卡住」都是 rank 根本没起来，
    或者两边 ``TGAI_DP_JOB`` 不一致（于是看的是两个目录）。
    """
    path = os.path.join(dirpath, "nccl_id")
    t0 = time.time()
    while True:
        if os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    raw = fh.read()
            except OSError:
                raw = b""
            if len(raw) == UNIQUE_ID_BYTES:
                return raw
            # 文件刚被创建但还没写完（理论上有 os.replace 不会发生）——继续等
        if time.time() - t0 > float(timeout):
            raise NcclError(
                "等待 NCCL unique id 超时（%.1fs）：%s" % (float(timeout), path) +
                "。修法：确认所有 rank 都已启动、TGAI_DP_JOB 一致、rank0 没有先崩"
                "（rank0 负责生成 nccl_id）")
        time.sleep(float(poll))


def _candidate_dirs(extra_dirs=None):
    """按「最可能命中」的顺序给出目录列表；纯函数，方便测试。"""
    out = []

    def _add(p):
        if p and os.path.isdir(p):
            ap = os.path.abspath(p)
            if ap not in out:
                out.append(ap)

    for d in (extra_dirs or ()):
        _add(d)
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        _add(d)
    # torch 的 wheel 自带 libnccl.so.2 —— 只查路径，**不 import torch**
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec is not None and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                _add(os.path.join(loc, "lib"))
    except Exception:                                    # noqa: BLE001
        pass
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        if spec is not None and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                for sub in ("nccl/lib", "nccl", "cuda_nccl/lib"):
                    _add(os.path.join(loc, *sub.split("/")))
    except Exception:                                    # noqa: BLE001
        pass
    prefix = sys.prefix
    for sub in ("lib", "lib64", "lib/x86_64-linux-gnu"):
        _add(os.path.join(prefix, *sub.split("/")))
    for pat in ("/usr/local/cuda*/lib64", "/usr/local/cuda*/targets/*/lib"):
        for d in sorted(glob.glob(pat)):
            _add(d)
    for d in ("/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib64",
              "/usr/local/lib", "/usr/lib/aarch64-linux-gnu", "/opt/conda/lib"):
        _add(d)
    return out


class LibNccl(object):
    """`libnccl.so` 的 ctypes 绑定 + 版本校验。"""

    #: 必需符号 → 用途（缺一个就 NcclUnavailable，不做半残降级）
    REQUIRED = (
        "ncclGetUniqueId",
        "ncclCommInitRank",
        "ncclAllReduce",
        "ncclCommDestroy",
        "ncclGetVersion",
        "ncclGetErrorString",
        "ncclGroupStart",
        "ncclGroupEnd",
        "ncclCommCount",
        "ncclCommUserRank",
    )

    def __init__(self, path, handle=None):
        self.path = str(path)
        self._dll = handle if handle is not None else self._load(self.path)
        self._bind()
        self._version = self._read_version()
        if self._version < MIN_VERSION_CODE:
            raise NcclVersionMismatch(
                "NCCL 版本过老：%s（%d）< 2.0.0；文件 %s"
                % (version_string(self._version), self._version, self.path) +
                "。修法：ORV 的 DP 后端要求 NCCL >= 2.0（ncclCommInitRank 的起点）；"
                "升级 NCCL 或改用 TG_DP_BACKEND=host")

    # ---------- 加载 ----------

    @staticmethod
    def _load(path):
        mode = getattr(ctypes, "RTLD_GLOBAL", 0) | getattr(ctypes, "RTLD_NOW", 0)
        try:
            return ctypes.CDLL(path, mode=mode)
        except OSError as e:
            try:
                return ctypes.CDLL(path)
            except OSError:
                raise NcclUnavailable(
                    "无法加载 NCCL 动态库：%s（%s）" % (path, e) +
                    "。修法：设置 TG_NCCL_LIB 指向可加载的 libnccl.so.2，"
                    "或改用 TG_DP_BACKEND=host")

    def _sym(self, name):
        try:
            return getattr(self._dll, name)
        except AttributeError:
            raise NcclUnavailable(
                "NCCL 库缺少符号 %s：%s" % (name, self.path) +
                "。修法：库不是完整 NCCL（或版本过老）；换一个 libnccl.so.2，"
                "或改用 TG_DP_BACKEND=host")

    def _bind(self):
        c = ctypes
        pvoid = c.POINTER(c.c_void_p)
        self._get_unique_id = self._sym("ncclGetUniqueId")
        self._get_unique_id.argtypes = [c.POINTER(_NcclUniqueId)]
        self._get_unique_id.restype = c.c_int

        self._comm_init_rank = self._sym("ncclCommInitRank")
        self._comm_init_rank.argtypes = [pvoid, c.c_int, _NcclUniqueId, c.c_int]
        self._comm_init_rank.restype = c.c_int

        self._all_reduce = self._sym("ncclAllReduce")
        self._all_reduce.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t, c.c_int,
                                     c.c_int, c.c_void_p, c.c_void_p]
        self._all_reduce.restype = c.c_int

        self._comm_destroy = self._sym("ncclCommDestroy")
        self._comm_destroy.argtypes = [c.c_void_p]
        self._comm_destroy.restype = c.c_int

        abort = getattr(self._dll, "ncclCommAbort", None)
        if abort is not None:
            abort.argtypes = [c.c_void_p]
            abort.restype = c.c_int
        self._comm_abort = abort

        self._get_version = self._sym("ncclGetVersion")
        self._get_version.argtypes = [c.POINTER(c.c_int)]
        self._get_version.restype = c.c_int

        self._err_string = self._sym("ncclGetErrorString")
        self._err_string.argtypes = [c.c_int]
        self._err_string.restype = c.c_char_p

        self._group_start = self._sym("ncclGroupStart")
        self._group_start.argtypes = []
        self._group_start.restype = c.c_int

        self._group_end = self._sym("ncclGroupEnd")
        self._group_end.argtypes = []
        self._group_end.restype = c.c_int

        self._comm_count = self._sym("ncclCommCount")
        self._comm_count.argtypes = [c.c_void_p, c.POINTER(c.c_int)]
        self._comm_count.restype = c.c_int

        self._comm_user_rank = self._sym("ncclCommUserRank")
        self._comm_user_rank.argtypes = [c.c_void_p, c.POINTER(c.c_int)]
        self._comm_user_rank.restype = c.c_int

        self._comm_cu_device = None
        cu_dev = getattr(self._dll, "ncclCommCuDevice", None)
        if cu_dev is not None:
            cu_dev.argtypes = [c.c_void_p, c.POINTER(c.c_int)]
            cu_dev.restype = c.c_int
            self._comm_cu_device = cu_dev

        # ---- ZeRO 用的两个原语（OPTIONAL：缺了就在调用时报带修法的错）----
        # ncclAllGather(sendbuff, recvbuff, sendcount, dtype, comm, stream)
        self._all_gather = self._opt_sym(
            "ncclAllGather",
            [c.c_void_p, c.c_void_p, c.c_size_t, c.c_int, c.c_void_p, c.c_void_p])
        # ncclReduceScatter(sendbuff, recvbuff, recvcount, dtype, op, comm, stream)
        self._reduce_scatter = self._opt_sym(
            "ncclReduceScatter",
            [c.c_void_p, c.c_void_p, c.c_size_t, c.c_int, c.c_int,
             c.c_void_p, c.c_void_p])

    def _opt_sym(self, name, argtypes):
        """绑定一个**可选**符号：库里有就绑，没有就返回 None（调用时再报错）。

        为什么不放进 :data:`REQUIRED`：`ncclAllGather` / `ncclReduceScatter` 只有
        ZeRO 用得上，而 ZeRO 是可选功能；把它们列成必需会让**本来能跑 DP 的环境**
        因为一个残缺库直接起不来。
        """
        try:
            fn = getattr(self._dll, name)
        except AttributeError:
            return None
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
        return fn

    def _read_version(self):
        code = ctypes.c_int(0)
        rc = int(self._get_version(ctypes.byref(code)))
        if rc != 0:
            raise NcclUnavailable("ncclGetVersion 失败 rc=%d（%s）" % (rc, self.path) +
                                  "。修法：库与驱动不匹配；改用 TG_DP_BACKEND=host")
        return int(code.value)

    # ---------- 对外 ----------

    @property
    def version(self):
        return int(self._version)

    def version_str(self):
        return version_string(self._version)

    def error_string(self, rc):
        try:
            s = self._err_string(int(rc))
            if s:
                return s.decode("utf-8", "replace")
        except Exception:                                # noqa: BLE001
            pass
        return "rc=%d" % int(rc)

    def check(self, rc, what):
        rc = int(rc)
        if rc != 0:
            raise NcclError("%s 失败：%s" % (what, self.error_string(rc)) +
                            "。修法：先跑 `python -m tgai_go.train.dist_nccl --selftest` "
                            "定位是握手还是集合通信；仍失败则用 TG_DP_BACKEND=host")

    def get_unique_id(self):
        uid = _NcclUniqueId()
        self.check(self._get_unique_id(ctypes.byref(uid)), "ncclGetUniqueId")
        return uid

    def init_rank(self, world, rank, uid):
        comm = ctypes.c_void_p()
        self.check(self._comm_init_rank(ctypes.byref(comm), int(world), uid, int(rank)),
                   "ncclCommInitRank")
        if not comm.value:
            raise NcclError("ncclCommInitRank 返回空 comm" +
                            "。修法：这是 NCCL/驱动级失败；把完整错误、GPU 型号与驱动版本一起反馈")
        return comm

    def comm_count(self, comm):
        v = ctypes.c_int(0)
        self.check(self._comm_count(comm, ctypes.byref(v)), "ncclCommCount")
        return int(v.value)

    def comm_user_rank(self, comm):
        v = ctypes.c_int(-1)
        self.check(self._comm_user_rank(comm, ctypes.byref(v)), "ncclCommUserRank")
        return int(v.value)

    def comm_cu_device(self, comm):
        if self._comm_cu_device is None:
            return None
        v = ctypes.c_int(-1)
        self.check(self._comm_cu_device(comm, ctypes.byref(v)), "ncclCommCuDevice")
        return int(v.value)

    def destroy(self, comm):
        return int(self._comm_destroy(comm))

    def abort(self, comm):
        if self._comm_abort is None:
            return None
        return int(self._comm_abort(comm))

    def group_start(self):
        self.check(self._group_start(), "ncclGroupStart")

    def group_end(self):
        self.check(self._group_end(), "ncclGroupEnd")

    def all_reduce(self, comm, send_ptr, recv_ptr, count, stream=None):
        """单次 ``Sum(float32)`` 就地归约。``stream`` 为 ``c_void_p`` / int / None。"""
        if isinstance(stream, ctypes.c_void_p):
            st = stream
        elif stream is None:
            st = ctypes.c_void_p(0)
        else:
            st = ctypes.c_void_p(int(stream))
        self.check(self._all_reduce(ctypes.c_void_p(int(send_ptr)),
                                    ctypes.c_void_p(int(recv_ptr)),
                                    ctypes.c_size_t(int(count)),
                                    ctypes.c_int(NCCL_FLOAT32),
                                    ctypes.c_int(NCCL_SUM),
                                    comm,
                                    st),
                   "ncclAllReduce")

    # ---------- ZeRO 用的两个原语 ----------

    def all_gather(self, comm, send_ptr, recv_ptr, send_count, stream=None):
        """``ncclAllGather``：每 rank 发 ``send_count`` 个 fp32，收 ``world*send_count``。

        ``recv`` 的布局是 **rank 顺序**（rank0 的 send_count 个在最前）。
        **所有 rank 的 `send_count` 必须相同** —— 传不同的值会挂或者算错。
        """
        if self._all_gather is None:
            raise NcclUnavailable(
                "NCCL 库没有 ncclAllGather 符号：%s" % self.path +
                "。修法：换一个完整的 libnccl.so.2（>=2.0 都有这个符号），"
                "或不用 ZeRO（--zero 0）")
        st = self._stream_arg(stream)
        self.check(self._all_gather(ctypes.c_void_p(int(send_ptr)),
                                    ctypes.c_void_p(int(recv_ptr)),
                                    ctypes.c_size_t(int(send_count)),
                                    ctypes.c_int(NCCL_FLOAT32),
                                    comm, st),
                   "ncclAllGather")

    def reduce_scatter(self, comm, send_ptr, recv_ptr, recv_count, stream=None):
        """``ncclReduceScatter``：``send`` 按 rank 排成 ``world`` 段，每段 ``recv_count``。

        结果 ``recv[i] = sum_r send[r*recv_count + i]``（fp32，SUM）。
        **所有 rank 的 `recv_count` 必须相同**。
        """
        if self._reduce_scatter is None:
            raise NcclUnavailable(
                "NCCL 库没有 ncclReduceScatter 符号：%s" % self.path +
                "。修法：换一个完整的 libnccl.so.2（>=2.0 都有这个符号），"
                "或把 ZeRO 降一档（--zero 1 只用 all_reduce + all_gather）")
        st = self._stream_arg(stream)
        self.check(self._reduce_scatter(ctypes.c_void_p(int(send_ptr)),
                                        ctypes.c_void_p(int(recv_ptr)),
                                        ctypes.c_size_t(int(recv_count)),
                                        ctypes.c_int(NCCL_FLOAT32),
                                        ctypes.c_int(NCCL_SUM),
                                        comm, st),
                   "ncclReduceScatter")

    @staticmethod
    def _stream_arg(stream):
        if isinstance(stream, ctypes.c_void_p):
            return stream
        if stream is None:
            return ctypes.c_void_p(0)
        return ctypes.c_void_p(int(stream))


def version_string(code):
    """``22703`` → ``"2.27.3"``。"""
    code = int(code)
    return "%d.%d.%d" % (code // 10000, (code // 100) % 100, code % 100)


def find_libnccl(extra_dirs=None):
    """返回 ``LibNccl`` 实例；找不到/加载失败/版本过老都抛 :class:`NcclError`。

    ``TG_NCCL_LIB`` 优先级最高，且**语义是硬要求**：给了却加载不了就报错，
    不会悄悄去别处找一个来用（否则「我明明指定了却跑了另一个库」无从排查）。
    """
    explicit = os.environ.get("TG_NCCL_LIB", "").strip()
    if explicit:
        cands = []
        if os.path.isdir(explicit):
            for nm in _LIB_NAMES:
                cands.append(os.path.join(explicit, nm))
            cands.extend(sorted(glob.glob(os.path.join(explicit, "libnccl.so*"))))
        else:
            cands.append(explicit)
        for p in cands:
            if os.path.isfile(p):
                return LibNccl(p)
        raise NcclUnavailable(
            "TG_NCCL_LIB=%s 不是可加载的 libnccl（不是文件，或目录里没有 libnccl.so*）"
            % explicit +
            "。修法：把它指向 libnccl.so.2 的完整路径或是包含它的目录；"
            "或清掉该变量让程序自动搜索")

    tried = []
    found = ctypes.util.find_library("nccl")
    if found:
        tried.append(found)
        try:
            return LibNccl(found)
        except NcclUnavailable:
            pass
    for d in _candidate_dirs(extra_dirs):
        for nm in _LIB_NAMES[:-2]:                       # 不试 Windows/mac 名字
            p = os.path.join(d, nm)
            if not os.path.isfile(p):
                continue
            tried.append(p)
            try:
                return LibNccl(p)
            except NcclUnavailable:
                continue
        for p in sorted(glob.glob(os.path.join(d, "libnccl.so.2*"))):
            if p in tried:
                continue
            tried.append(p)
            try:
                return LibNccl(p)
            except NcclUnavailable:
                continue
    raise NcclUnavailable(
        "找不到可用的 libnccl.so（试过 %d 个位置%s）"
        % (len(tried), ("：" + ", ".join(tried[:6])) if tried else "") +
        "。修法：装 NCCL（或 pip install nvidia-nccl-cu12），设 TG_NCCL_LIB 指向 "
        "libnccl.so.2；不想装就用 TG_DP_BACKEND=host（当前默认）")


class NcclComm(object):
    """一个 rank 的 NCCL 通信器。

    ``lib`` 可注入（测试用假库）；``stream`` 为空时按 `None` 交给 NCCL（默认流）。
    初始化后校验「comm 里看到的 world/rank」与调用方一致 —— 不一致直接抛。
    """

    def __init__(self, lib, world, rank, unique_id, stream=None, expect_device=None):
        self.lib = lib
        self.world = int(world)
        self.rank = int(rank)
        self.stream = stream
        self._comm = lib.init_rank(self.world, self.rank, unique_id)
        self.closed = False
        try:
            got_count = lib.comm_count(self._comm)
            got_rank = lib.comm_user_rank(self._comm)
            if got_count != self.world:
                raise NcclRankMismatch(
                    "NCCL 通信器里的 rank 数是 %d，但调用方声明 world=%d"
                    % (got_count, self.world) +
                    "。修法：两个进程的 TGAI_DP_WORLD 必须相同；"
                    "并确认没有第三个进程加入了同一个 job")
            if got_rank != self.rank:
                raise NcclRankMismatch(
                    "NCCL 通信器里的 rank 是 %d，但调用方声明 rank=%d"
                    % (got_rank, self.rank) +
                    "。修法：TGAI_DP_RANK 必须与启动顺序/unique id 一致")
            self.device = lib.comm_cu_device(self._comm)
            if expect_device is not None and self.device is not None \
                    and int(self.device) != int(expect_device):
                raise NcclRankMismatch(
                    "NCCL 通信器绑在 CUDA 设备 %d，但当前进程用的是设备 %s"
                    % (int(self.device), expect_device) +
                    "。修法：每个 rank 必须只看见一张卡（CUDA_VISIBLE_DEVICES=0/1）")
        except Exception:
            self.close(abort=True)
            raise

    def all_reduce_group(self, entries, stream=None):
        """``entries``: [(ptr:int, count:int), ...]，一次 group 内全部就地 Sum。

        所有 rank 必须以**完全相同的顺序和 count** 调用（少一个/多一个都会挂）。

        ``stream``：可选的覆盖流（``None`` ⇒ 用建 communicator 时给的流）。
        数据并行**重叠归约**用它把每个桶的 all-reduce 放到一条独立的**非阻塞**流上，
        与反向的计算流并发（见 ``train/dp_overlap.py``）。
        """
        if self.closed:
            raise NcclError("NcclComm 已关闭" + "。修法：不要跨 close() 复用通信器")
        st = self.stream if stream is None else stream
        lib = self.lib
        lib.group_start()
        try:
            for ptr, count in entries:
                if int(count) <= 0:
                    continue
                lib.all_reduce(self._comm, ptr, ptr, int(count), stream=st)
        finally:
            lib.group_end()

    def all_gather_flat(self, send_ptr, recv_ptr, send_count):
        """一次 ``ncclAllGather``（fp32）。``send_count`` 必须各 rank 相同。"""
        if self.closed:
            raise NcclError("NcclComm 已关闭" + "。修法：不要跨 close() 复用通信器")
        self.lib.all_gather(self._comm, send_ptr, recv_ptr, int(send_count),
                            stream=self.stream)

    def reduce_scatter_flat(self, send_ptr, recv_ptr, recv_count):
        """一次 ``ncclReduceScatter``（fp32 Sum）。``recv_count`` 必须各 rank 相同。"""
        if self.closed:
            raise NcclError("NcclComm 已关闭" + "。修法：不要跨 close() 复用通信器")
        self.lib.reduce_scatter(self._comm, send_ptr, recv_ptr, int(recv_count),
                                stream=self.stream)

    def close(self, abort=False):
        if self.closed or self._comm is None:
            return
        try:
            if abort:
                self.lib.abort(self._comm)
            else:
                self.lib.destroy(self._comm)
        finally:
            self.closed = True
            self._comm = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close(abort=exc[0] is not None)
        return False


# ═════════════════════════════════════════════════════════════════════════
# 自检（远端第一次上机时用；不参与训练）
# ═════════════════════════════════════════════════════════════════════════
def _selftest():
    """``python -m tgai_go.train.dist_nccl --selftest``

    两个进程各跑一次（``TGAI_DP_WORLD=2 TGAI_DP_RANK=0/1``），
    在设备上对 [1,2,3,...] 做 Sum，期望每个 rank 得到 ``world * 原值``。
    """
    import numpy as np

    import tgai_go.train as T
    from tgai_go.train import dist as D

    world = int(os.environ.get("TGAI_DP_WORLD", "1"))
    rank = int(os.environ.get("TGAI_DP_RANK", "0"))
    if world <= 1:
        print("[selftest] world<=1：NCCL 用不到（DP 空操作）")
        return 0
    T.set_default_device("cuda")
    rt = T._cuda()
    lib = find_libnccl()
    dp = D.DataParallel(world=world, rank=rank, job=os.environ.get("TGAI_DP_JOB", "selftest"))
    dp._start()
    uid_path_dir = dp.dir
    if rank == 0:
        uid = lib.get_unique_id()
        write_unique_id(uid_path_dir, unique_id_to_bytes(uid))
        raw = unique_id_to_bytes(uid)
    else:
        raw = wait_unique_id(uid_path_dir, timeout=dp.timeout)
    uid = unique_id_from_bytes(raw)
    comm = NcclComm(lib, world, rank, uid, stream=rt._stream)
    n = 1024
    buf = rt.h2d(np.full((n,), float(rank + 1), dtype=np.float32))
    comm.all_reduce_group([(buf.ptr, n)])
    rt.sync()
    got = rt.d2h(buf)
    want = float(world * (world + 1) / 2.0)
    ok = bool(np.allclose(got, want))
    print("[selftest] lib=%s ver=%s world=%d rank=%d device=%s got=%.4f want=%.4f %s"
          % (lib.path, lib.version_str(), world, rank, comm.device,
             float(got[0]), want, "OK" if ok else "MISMATCH"))
    comm.close()
    dp.close()
    return 0 if ok else 3


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
