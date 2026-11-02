# -*- coding: utf-8 -*-
"""GPU P2P 梯度全归约（跨进程：CUDA IPC 句柄 + `cuMemcpyPeerAsync`）。

为什么需要：`dist.py` 现有实现把梯度拷回主机、经共享内存交换、再拷回 GPU。
0.5B 模型每步约 1.9 GB 梯度，主机往返 0.3~0.5 s ⇒ **两卡会比一卡还慢**，拿不到 1.7×。

做法（每卡一个进程，槽位编号 `r * n + i`）：
  1. 每个 rank 在**自己**显存里分配 world*n 个槽位，把 IPC 句柄写到 rendezvous 目录
  2. rank r 把自己的第 i 个梯度写进**每个对端**的槽位 (r, i)
  3. 每个 rank 本地按 rank 顺序累加自己那 world 份 → 求和顺序固定（可复现）
  4. 平均后替换 `p.grad`

开关：`TGAI_DP_WORLD` / `TGAI_DP_RANK` / `TGAI_DP_P2P=1`。
world==1 或未开 P2P 时，所有方法都是空操作（不改变单卡行为）。
"""
import ctypes
import os
import struct
import tempfile
import time

IPC_HANDLE_BYTES = 64
_IPC_LAZY_ENABLE_PEER = 1


class P2PError(RuntimeError):
    """P2P 不可用或通信失败（消息里一定带修法）。"""


def p2p_requested():
    return str(os.environ.get("TGAI_DP_P2P", "")).strip().lower() in ("1", "true", "on", "yes")


def rendezvous_dir(job=None):
    job = job or os.environ.get("TGAI_DP_JOB", "default")
    d = os.path.join(tempfile.gettempdir(), "tgai_dp_%s" % job)
    os.makedirs(d, exist_ok=True)
    return d


class _Cu:
    def __init__(self):
        lib = None
        for n in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
            try:
                lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if lib is None:
            raise P2PError("找不到 CUDA 驱动库（libcuda.so.1 / nvcuda.dll）。"
                           "修法：确认驱动可用，或不开 TGAI_DP_P2P 走主机共享内存")
        self.lib = lib
        b = lib
        b.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        b.cuCtxGetDevice.restype = ctypes.c_int
        b.cuIpcGetMemHandle.argtypes = [ctypes.POINTER(ctypes.c_byte), ctypes.c_void_p]
        b.cuIpcGetMemHandle.restype = ctypes.c_int
        b.cuIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                         ctypes.POINTER(ctypes.c_byte), ctypes.c_uint]
        b.cuIpcOpenMemHandle.restype = ctypes.c_int
        b.cuMemcpyPeerAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_size_t, ctypes.c_void_p]
        b.cuMemcpyPeerAsync.restype = ctypes.c_int
        b.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        b.cuStreamCreate.restype = ctypes.c_int
        b.cuCtxSynchronize.argtypes = []
        b.cuCtxSynchronize.restype = ctypes.c_int

    def check(self, rc, what):
        if rc != 0:
            raise P2PError("%s 失败 rc=%d。修法：确认两卡在同一台机器、"
                           "`nvidia-smi topo -m` 显示 PIX/PXB/NV# 连接" % (what, rc))


class P2PAllReduce:
    def __init__(self):
        self.world = max(1, int(os.environ.get("TGAI_DP_WORLD", "1") or 1))
        self.rank = int(os.environ.get("TGAI_DP_RANK", "0") or 0)
        if not (0 <= self.rank < self.world):
            raise P2PError("TGAI_DP_RANK=%d 越界（world=%d）。修法：0 <= rank < world"
                           % (self.rank, self.world))
        self.enabled = self.world > 1 and p2p_requested()
        self.cu = None
        self.slots = []          # world*n 个本地缓冲
        self.sizes = []
        self.dir = None
        self.stream = None
        self._opened = {}
        if self.enabled:
            self.cu = _Cu()
            dev = ctypes.c_int()
            self.cu.check(self.cu.lib.cuCtxGetDevice(ctypes.byref(dev)), "cuCtxGetDevice")
            self.dev = int(dev.value)
            st = ctypes.c_void_p()
            self.cu.check(self.cu.lib.cuStreamCreate(ctypes.byref(st), ctypes.c_uint(0)),
                          "cuStreamCreate")
            self.stream = st
            self.dir = rendezvous_dir()

    # ---------- 建立阶段 ----------
    def attach(self, params):
        """按参数形状建槽位、交换 IPC 句柄。训练前调一次。"""
        if not self.enabled:
            return
        import tgai_go.train as T
        rt = T._cuda()
        sizes = []
        for p in params:
            n = int(getattr(p, "numel", lambda: 0)())
            if n:
                sizes.append(n)
        if not sizes:
            raise P2PError("没有可归约的参数。修法：传入至少一个非空参数")
        self.sizes = sizes
        n = len(sizes)
        self.slots = [rt.empty((s,), dtype=T.float32 if hasattr(T, "float32") else None)
                      for s in sizes for _ in range(self.world)]
        handles = []
        for b in self.slots:
            h = (ctypes.c_byte * IPC_HANDLE_BYTES)()
            self.cu.check(self.cu.lib.cuIpcGetMemHandle(h, ctypes.c_void_p(int(b.ptr))),
                          "cuIpcGetMemHandle")
            handles.append(bytes(h))
        my = os.path.join(self.dir, "rank%d.handles" % self.rank)
        tmp = my + ".tmp"
        with open(tmp, "wb") as f:
            f.write(struct.pack("<II", self.world, n))
            for s in sizes:
                f.write(struct.pack("<Q", int(s)))
            for h in handles:
                f.write(h)
        os.replace(tmp, my)
        deadline = time.time() + 180
        while True:
            ready = all(os.path.isfile(os.path.join(self.dir, "rank%d.handles" % r))
                        for r in range(self.world))
            if ready:
                break
            if time.time() > deadline:
                raise P2PError(
                    "等对端 IPC 句柄超时（world=%d，目录 %s）。修法：确认所有 rank 都已启动、"
                    "TGAI_DP_JOB 一致、临时目录可写" % (self.world, self.dir))
            time.sleep(0.05)

    def _peer_ptrs(self, rank):
        if rank in self._opened:
            return self._opened[rank]
        with open(os.path.join(self.dir, "rank%d.handles" % rank), "rb") as f:
            blob = f.read()
        world, n = struct.unpack_from("<II", blob, 0)
        off = 8 + 8 * n
        ptrs = []
        for k in range(world * n):
            h = (ctypes.c_byte * IPC_HANDLE_BYTES).from_buffer_copy(
                blob[off + IPC_HANDLE_BYTES * k: off + IPC_HANDLE_BYTES * (k + 1)])
            p = ctypes.c_void_p()
            self.cu.check(self.cu.lib.cuIpcOpenMemHandle(
                ctypes.byref(p), h, ctypes.c_uint(_IPC_LAZY_ENABLE_PEER)),
                "cuIpcOpenMemHandle(rank %d)" % rank)
            ptrs.append(int(p.value))
        self._opened[rank] = ptrs
        return ptrs

    # ---------- 归约 ----------
    def all_reduce_grads(self, params):
        if not self.enabled:
            return
        import tgai_go.train as T
        rt = T._cuda()
        grads = [(i, p) for i, p in enumerate(params)
                 if getattr(p, "grad", None) is not None]
        if not grads:
            return
        if len(grads) != len(self.sizes):
            raise P2PError("梯度个数 %d != attach 时的 %d。修法：attach() 与 all_reduce_grads() "
                           "传同一批参数" % (len(grads), len(self.sizes)))
        n = len(self.sizes)
        # 1) 自己的第 i 份写进每个对端（含自己）的槽位 (self.rank, i)
        for r in range(self.world):
            ptrs = ([int(b.ptr) for b in self.slots] if r == self.rank
                    else self._peer_ptrs(r))
            for i, (_idx, p) in enumerate(grads):
                dst = ptrs[self.rank * n + i]
                if r == self.rank:
                    continue
                self.cu.check(self.cu.lib.cuMemcpyPeerAsync(
                    ctypes.c_void_p(dst), ctypes.c_void_p(0),
                    ctypes.c_void_p(int(p.grad._gpu.ptr)), ctypes.c_void_p(0),
                    ctypes.c_size_t(self.sizes[i] * 4), self.stream),
                    "cuMemcpyPeerAsync(%d->%d)" % (self.rank, r))
        self.cu.check(self.cu.lib.cuCtxSynchronize(), "cuCtxSynchronize(copy)")
        # 自己那份也放进本地槽位（rank==self.rank 时上面跳过了拷贝）
        for i, (_idx, p) in enumerate(grads):
            self.cu.check(self.cu.lib.cuMemcpyPeerAsync(
                ctypes.c_void_p(int(self.slots[self.rank * n + i].ptr)), ctypes.c_void_p(0),
                ctypes.c_void_p(int(p.grad._gpu.ptr)), ctypes.c_void_p(0),
                ctypes.c_size_t(self.sizes[i] * 4), self.stream),
                "cuMemcpyPeerAsync(本地)")
        self.cu.check(self.cu.lib.cuCtxSynchronize(), "cuCtxSynchronize(local)")
        # 2) 本地按 rank 顺序求和并平均，替换 p.grad
        for i, (_idx, p) in enumerate(grads):
            acc = rt.clone(self.slots[0 * n + i])
            for r in range(1, self.world):
                acc = rt.add(acc, self.slots[r * n + i])
            acc = rt.scale(acc, 1.0 / float(self.world))
            p.grad = T.Tensor._from_gpu(acc, requires_grad=False)
        rt.sync()
