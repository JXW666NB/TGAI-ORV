# -*- coding: utf-8 -*-
"""数据并行（多卡）地基：参数广播 + 梯度全归约。

设计目标：**先正确，再快**。第一版走「本机共享内存」做交换，
不依赖 NCCL / MPI / torch.distributed —— 我们本来就是自研运行时。

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
import tempfile
import time

import numpy as np

try:
    from multiprocessing import shared_memory
    _HAS_SHM = True
except Exception:                                    # noqa: BLE001
    _HAS_SHM = False


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
    def __init__(self, world=None, rank=None, job=None, timeout=600.0):
        self.world = _env_int("TGAI_DP_WORLD", 1) if world is None else int(world)
        self.rank = _env_int("TGAI_DP_RANK", 0) if rank is None else int(rank)
        self.job = job or os.environ.get("TGAI_DP_JOB", "default")
        self.timeout = float(timeout)
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
                fh.write(b"1")
        self._wait_for(ready, "rendezvous 就绪")
        self._barrier("init")
        self._started = True

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
            arr = _grad_to_numpy(g) if g is not None else None
            if arr is not None and arr.shape != tuple(getattr(p, "shape", arr.shape)):
                # 形状对不上时不能静默：后面的按位置切片会整段错位
                raise RuntimeError(
                    "梯度形状 %s 与参数形状 %s 不一致。修法：确认没有手动改过 p.grad"
                    % (arr.shape, tuple(getattr(p, "shape", ()))))
            n = int(arr.size) if arr is not None else int(p.numel())
            entries.append((p, arr, g, n))
        total = sum(e[3] for e in entries)
        if total <= 0:
            raise RuntimeError("梯度总长度为 0" + "。修法：确认反向已经跑过、params 非空")
        self._ensure_slot(max(total * 4, 8))

        off = 0
        base = self.rank * total * 4
        for p, arr, _orig, n in entries:       # 各 rank 写自己那一段；缺梯度写 0
            if arr is None:
                blob = np.zeros(n, dtype=np.float32).tobytes()
            else:
                blob = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
            self._slot.buf[base + off:base + off + len(blob)] = blob
            off += len(blob)
        self._barrier("grads")

        buf = np.ndarray((self.world, total), dtype=np.float32,
                         buffer=self._slot.buf, offset=0)
        acc = buf.sum(axis=0)
        if scale is None:
            acc /= float(self.world)
        else:
            acc *= float(scale)
        off = 0
        for p, arr, orig, n in entries:
            val = acc[off:off + n].astype(np.float32, copy=True)
            if arr is not None:
                val = val.astype(arr.dtype, copy=False).reshape(arr.shape)
            else:
                val = val.reshape(tuple(getattr(p, "shape", (n,))))
            _set_grad(p, val, orig)
            off += n

    def close(self):
        """释放共享内存。

        ★ **不要删自己落下的 barrier 标记文件**（2026-11-13 修）。
          旧实现在这里 `os.remove("bar_*.%d")`，于是**跑得快的 rank 一收尾**就把标记删了，
          而慢的 rank 还在 `_wait_for` 等那个文件 ⇒ 永久卡死（实测 3 次里挂 1 次）。
          残留标记由下次启动时 rank0 在 `ready` 之前统一清理，所以留着是安全的。
        """
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
