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
"""
import os
import ctypes
import tempfile
import time

import numpy as np

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
    def __init__(self, world=None, rank=None, job=None, timeout=None, backend=None):
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
        ready = os.path.join(self.dir, "ready")
        if self.rank == 0:
            for f in os.listdir(self.dir):
                if f.startswith("bar_") or f.startswith("ready"):
                    try:
                        os.remove(os.path.join(self.dir, f))
                    except OSError:
                        pass
            with open(ready, "wb") as fh:
                fh.write(b"%d %f" % (os.getpid(), time.time()))
        elif self._ready_is_stale(ready):
            print("[dp] WARN 发现上一轮残留的 ready（owner pid 已退出），清掉后重新等"
                  "：%s" % ready, flush=True)
            try:
                os.remove(ready)
            except OSError:
                pass
        self._wait_for(ready, "rendezvous 就绪")
        self._barrier("init")
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
        """`ready` 是否是上一轮遗留：里面记着 rank0 的 pid，pid 不在了就是旧的。

        格式是 ``"<pid> <时间戳>"``（两个字段）。只有一个字段的是**旧版本**写的，
        无法证明它属于当前这轮 ⇒ 一律当残留。
        """
        try:
            with open(path, "rb") as fh:
                raw = fh.read(64)
        except OSError:
            return False                     # 没有文件 ⇒ 不是"残留"，正常等
        parts = raw.split()
        if len(parts) < 2:
            return True
        try:
            pid = int(parts[0])
        except (ValueError, IndexError):
            return True
        if pid == os.getpid():
            return False
        return not DataParallel._pid_alive(pid)

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
        """
        self._bar_gen = int(getattr(self, "_bar_gen", 0)) + 1
        t = "%s_%d" % (tag, self._bar_gen)
        p = os.path.join(self.dir, "bar_%s.%d" % (t, self.rank))
        with open(p, "wb") as f:
            f.write(b"1")
        for r in range(self.world):
            self._wait_for(os.path.join(self.dir, "bar_%s.%d" % (t, r)), "barrier " + t)

    def _ensure_slot(self, nbytes):
        """所有 rank 共用一块 `world * nbytes` 的共享内存，按 rank 分段写。"""
        if self._slot is not None and self._bytes >= nbytes:
            return
        if self._slot is not None:
            self._slot.close()
        need = int(nbytes)
        if self.rank == 0:
            self._slot = shared_memory.SharedMemory(
                name="tgai_dp_%s_%d" % (self.job, need), create=True, size=need * self.world)
            self._slot.buf[:8] = need.to_bytes(8, "little")
            self._shm.append(self._slot)
        else:
            name = "tgai_dp_%s_%d" % (self.job, need)
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

    def stats(self):
        """给日志/RESULT 用的后端信息（**实测值，不是配置值**）。"""
        out = {"backend_requested": self.backend_requested,
               "backend": self.backend,
               "world": int(self.world), "rank": int(self.rank),
               "comm_sec": self.comm_sec, "nccl_sec": self.nccl_sec,
               "nccl_calls": int(self._nccl_calls)}
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
                and not self._nccl_own_stream and self._nccl_stream is None):
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
        self._nccl_events = None
        self._nccl_stream = None
        self._nccl_own_stream = False

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
