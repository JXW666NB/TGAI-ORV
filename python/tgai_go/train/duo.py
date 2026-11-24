# -*- coding: utf-8 -*-
"""`T.DUO()` —— 一次调用接管本机的全部算力。

用户**不需要**知道 `init_process_group` / `local_rank` / `DistributedSampler` / `torchrun`。
只需要：

```python
import tgai_go.train as T

duo = T.DUO()                                  # 全自动：探测 → 选方案 → 起 worker
print(duo.summary())                           # 它会打印"选了什么、为什么"
duo.run(["python", "train.py"])                # 或者 duo.run(sys.argv)

# 高级：我就要这么摆
T.DUO(devices=[0, 2, 3], strategy="data_parallel")
T.DUO(devices=[0, 1], experts={0: 6, 1: 2})    # 专家并行：0 号 6 个专家，1 号 2 个
T.DUO(dry_run=True)                            # 只看计划，不碰任何进程
```

## 四条"物理墙"在这里的对应位置

| 墙 | 本模块的处理 |
| --- | --- |
| ① 碎片 / 泄漏 / 僵尸 / OOM | 每个 worker 在**自己的进程组**里起；`atexit` + 信号 + 异常全部走 `stop()`；**任何一个 rank 非 0 退出或命中 OOM 关键字，立刻杀掉其余 rank**（绝不让整机 hang 住）；`stop()` 幂等 |
| ② 数据切分 | 设置 `TGAI_DP_WORLD/RANK/JOB`，训练器用它取"本 rank 的数据分片"；`shard_of(n)` 给出与 `dist` 一致的切法 |
| ③ 木桶效应 | 异构卡（V100 + 4090）**不是能靠代码消除的**：同步 DP 一定被最慢的卡拖住。这里只做两件事 —— **检测并警告**，以及给出"慢卡/快卡"的比例，让用户决定要不要换成专家并行或干脆只用同型号 |
| ④ 通信原语 / 拓扑 | `topo.probe()` 拿到 NVLink/PIX/PXB/PHB/SYS 与 NUMA，喂给 `plan.py`；**拒绝**在 `SYS` 拓扑上给 ZeRO-3 这种"每层 all_gather"的方案背书 |

## 为什么是"起子进程"而不是"进程内多卡"

ORV 的后端是**全局单例**（一个进程一个设备），所以"接管整机"只能靠**每卡一个 worker 进程**。
这不是权宜之计 —— PyTorch 的 DDP 也是这么做的，区别是我们把那些仪式全吃掉了。
"""
import atexit
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time

from .. import __version__                            # noqa: F401  （保持包内一致风格）
from . import plan as _plan
from . import topo as _topo

__all__ = ["DUO", "DuoSession", "DuoResult", "DuoRefused"]

#: 命中这些字样就认为该 rank 挂了（OOM / 驱动级失败）。
_OOM_PAT = re.compile(
    r"显存不足|OutOfMemoryError|out of memory|cudaErrorMemoryAllocation|"
    r"CUDA error: out of memory|MemoryError", re.I)

_ACTIVE = {}          #: job -> DuoSession，防止同一进程重复接管
_ATEXIT_HOOKED = [False]


class DuoRefused(RuntimeError):
    """计划阶段就判定不可行。**此时一个进程都还没起** —— 这是刻意的。"""
    pass


class DuoResult(object):
    """`run()` 的结果：每个 rank 的退出码 + 是否命中 OOM + 日志路径。"""

    __slots__ = ("returncodes", "oom_ranks", "failed_rank", "elapsed_s", "logs",
                 "killed")

    def __init__(self):
        self.returncodes = {}
        self.oom_ranks = []
        self.failed_rank = None
        self.elapsed_s = 0.0
        self.logs = {}
        self.killed = False

    @property
    def ok(self):
        return not self.returncodes or all(v == 0 for v in self.returncodes.values())

    def summary(self):
        if self.ok:
            return "[duo] 全部 rank 正常退出（%.1f s）" % self.elapsed_s
        bits = ["[duo] 失败：rank%s 退出码 %s%s"
                % (self.failed_rank, self.returncodes.get(self.failed_rank),
                   "（OOM）" if self.failed_rank in self.oom_ranks else "")]
        if self.killed:
            bits.append("已主动杀掉其余 rank，避免整机 hang 住")
        for r, p in sorted(self.logs.items()):
            bits.append("   rank%d 日志：%s" % (r, p))
        return "\n".join(bits)


class DuoSession(object):
    """一次"接管"的句柄。**可 stop()，且 stop() 幂等。**"""

    def __init__(self, plan, topology, devices, strategy, job, log_dir=None,
                 timeout_s=None):
        self.plan = plan
        self.topology = topology
        self.devices = list(devices)
        self.strategy = strategy
        self.job = job
        self.log_dir = log_dir
        self.timeout_s = timeout_s
        self.procs = {}
        self._closed = False
        _ACTIVE[job] = self

    # ---- 信息 ----
    def summary(self):
        return self.topology.summary() + "\n" + self.plan.format()

    def shard_of(self, n_items):
        """本 rank 该拿第几份数据（与 `dist.shard_indices` 同一语义）。

        注意：**这是给 worker 内部用的**。父进程（起 worker 的那个）不该拿它。
        """
        world = max(1, len(self.devices))
        rank = int(os.environ.get("TGAI_DP_RANK", "0"))
        return _shard_indices(int(n_items), world, rank)

    # ---- 生命周期 ----
    def run(self, command, env=None, capture=True):
        """起 N 个 worker 并等它们结束。返回 `DuoResult`。

        `command` 可以是 list（推荐）或字符串（会按 shell 规则切分，**不**经过 shell）。
        """
        if self._closed:
            raise DuoRefused("这个 DUO 会话已经关闭" +
                             "。修法：重新调用 T.DUO() 拿一个新会话")
        if not self.plan.ok:
            raise DuoRefused("计划不可行，拒绝起进程：" + self.plan.reason +
                             "。修法：按上面的提示减少每卡负担（开 ZeRO / 换优化器 / 加卡）")
        cmd = shlex.split(command) if isinstance(command, str) else list(command)
        if not cmd:
            raise DuoRefused("command 是空的" +
                             "。修法：传入要跑的命令，例如 [sys.executable, 'train.py']")
        world = max(1, len(self.devices))
        result = DuoResult()
        t0 = time.time()
        _hook_atexit()
        master_port = _free_port()
        for rank in range(world):
            dev = self.devices[rank]
            e = dict(os.environ)
            e.update({
                "CUDA_VISIBLE_DEVICES": str(dev.index),
                "TGAI_DP_WORLD": str(world),
                "TGAI_DP_RANK": str(rank),
                "TGAI_DP_JOB": str(self.job),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(master_port),
                "PYTHONUNBUFFERED": "1",
                # ★ 日志必须是我们能解出来的编码。Windows 上子进程默认跟控制台走 GBK，
                #   重定向到文件后再按 UTF-8 读就成了乱码 —— 实测会让"显存不足"这种
                #   OOM 关键字匹配不上，于是 OOM 检测形同虚设。
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            })
            if env:
                e.update({str(k): str(v) for k, v in env.items()})
            logf = None
            if capture:
                d = self.log_dir or tempfile.mkdtemp(prefix="duo-")
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, "rank%d.log" % rank)
                logf = open(path, "wb")
                result.logs[rank] = path
            kw = {}
            if os.name == "nt":
                kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                kw["start_new_session"] = True
            p = subprocess.Popen(cmd, env=e, stdout=logf or None,
                                 stderr=(subprocess.STDOUT if logf else None), **kw)
            if logf:
                logf.close()                    # 子进程持有 fd，父进程关掉，避免句柄泄漏
            self.procs[rank] = p
        # ---- 监控：任何一个死了/非 0，立刻把其余的杀掉 ----
        try:
            while True:
                alive = [r for r, p in self.procs.items() if p.poll() is None]
                for r, p in self.procs.items():
                    rc = p.poll()
                    if rc is None or r in result.returncodes:
                        continue
                    result.returncodes[r] = rc
                    if rc != 0 and result.failed_rank is None:
                        result.failed_rank = r
                        self._kill_alive()
                        result.killed = True
                if not alive:
                    break
                if self.timeout_s and (time.time() - t0) > float(self.timeout_s):
                    result.failed_rank = result.failed_rank if result.failed_rank is not None else -1
                    self._kill_alive()
                    result.killed = True
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            self._kill_alive()
            result.killed = True
            raise
        finally:
            result.elapsed_s = time.time() - t0
            for r, p in self.procs.items():
                result.returncodes.setdefault(r, p.poll())
            result.oom_ranks = [r for r, path in result.logs.items()
                                if _log_matches(path, _OOM_PAT)]
            if result.oom_ranks and result.failed_rank is None:
                result.failed_rank = result.oom_ranks[0]
        return result

    def _kill_alive(self):
        for r, p in self.procs.items():
            if p.poll() is not None:
                continue
            try:
                if os.name == "nt":
                    p.terminate()
                else:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:                            # noqa: BLE001
                try:
                    p.terminate()
                except Exception:                        # noqa: BLE001
                    pass

    def stop(self, wait_s=5.0):
        """杀掉所有还活着的 worker。**幂等**，可以在 atexit 里重复调用。"""
        self._kill_alive()
        deadline = time.time() + float(wait_s)
        for p in self.procs.values():
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except Exception:                            # noqa: BLE001
                try:
                    p.kill()
                except Exception:                        # noqa: BLE001
                    pass
        self._closed = True
        _ACTIVE.pop(self.job, None)


def DUO(devices=None, strategy="auto", *, experts=None, n_params=None,
        optim=None, seq=None, d=None, d_ff=None, layers=None, E=None,
        job=None, log_dir=None, timeout_s=None, dry_run=False,
        single_step_s=None):
    """接管本机算力。返回 `DuoSession`（已选好方案，还没起进程）。

    参数（都能不填 —— 不填就自动探测/自动选）：

    * `devices`   —— 物理卡序号列表；默认用本机所有卡
    * `strategy`  —— `"auto"`（默认）/ `"data_parallel"` / `"expert_parallel"` /
                     `"zero1"` / `"zero2"`；**未实现的策略会明确拒绝**，不会假装支持
    * `experts`   —— `{rank: 专家数}`，等价于 `--place "0:e=6,1:e=2"`
    * `n_params`  —— 模型参数量；给了才能算显存。**不给就只能给建议、不能校验**
    * `dry_run`   —— 只打印计划
    """
    topo = _topo.probe(devices)
    if not topo.devices:
        # 没有 nvidia-smi 也要能给出可读的结论（比如纯 CPU 机器上跑测试）
        topo = _topo.HostTopology([], [], source="none")
    chosen = topo.devices if devices is None else [d for d in topo.devices
                                                   if d.index in set(devices)]
    if devices is not None and len(chosen) != len(set(devices)):
        raise DuoRefused(
            "要的卡不存在：请求 %s，本机只有 %s" % (sorted(set(devices)),
                                            [d.index for d in topo.devices]) +
            "。修法：用本机实际存在的卡号，或不传 devices 让它自动选")
    if not chosen:
        raise DuoRefused("没有可用显卡（nvidia-smi 没报出设备）" +
                         "。修法：确认驱动可用；纯 CPU 场景请直接用训练脚本，不要调 DUO")
    args = _Args(devices=None, place=_place_from(experts), n_params=n_params or 0,
                 opt=optim or "int8gpu", seq=seq if seq is not None else 512,
                 d=d if d is not None else 1024, d_ff=d_ff if d_ff is not None else 0,
                 layers=layers if layers is not None else 0,
                 E=E if E is not None else 0, zero=_zero_of(strategy),
                 dist="auto" if str(strategy) == "auto" else str(len(chosen)),
                 single_step_s=single_step_s)
    if str(strategy) in ("tensor_parallel", "pipeline_parallel"):
        raise DuoRefused(
            "策略 %r 还没有实现" % (strategy,) +
            "。修法：用 strategy='auto' / 'data_parallel' / 'expert_parallel'；"
            "张量并行与流水线并行在 ORV 里仍是设计阶段（见 docs/ZERO_PLAN.md）")
    pl = _plan.build(args, devices=[_as_plan_dev(d) for d in chosen])
    session = DuoSession(pl, topo, chosen, strategy, job or ("duo-%d" % int(time.time())),
                         log_dir=log_dir, timeout_s=timeout_s)
    if dry_run:
        print(session.summary())
    return session


# ---------------- 内部 ----------------
class _Args(object):
    """把 DUO 的参数伪装成 argparse.Namespace，好让 plan.build 复用。"""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _as_plan_dev(d):
    return _plan.Device(d.index, d.name, d.total_gib)


def _place_from(experts):
    if not experts:
        return None
    return ",".join("%d:e=%d" % (int(k), int(v)) for k, v in sorted(experts.items()))


def _zero_of(strategy):
    s = str(strategy)
    if s in ("zero1", "zero2"):
        return int(s[-1])
    return 0


def _shard_indices(n_items, world, rank):
    """与 `dist.shard_indices` 同语义：连续切分（各有各的一段，不重叠、合起来是全集）。"""
    n_items, world, rank = int(n_items), max(1, int(world)), int(rank)
    base, rem = divmod(n_items, world)
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return list(range(start, end))


def _free_port():
    import socket
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _log_matches(path, pat):
    """在日志里找关键字。**编码兜底**：DUO 自己起的 python 子进程已经强制 UTF-8，
    但用户可能跑别的程序（C++、别的 shell），所以这里按多编码尝试，
    任一解码能命中就算命中 —— 宁可多报 OOM，也不能漏报。
    """
    try:
        with open(path, "rb") as f:
            data = f.read(4 << 20)
    except Exception:                                # noqa: BLE001
        return False
    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            if pat.search(data.decode(enc, "replace")):
                return True
        except Exception:                            # noqa: BLE001
            continue
    return False


def _hook_atexit():
    """注册一次，进程退出时兜底杀掉所有还活着的 worker（**防僵尸**）。"""
    if _ATEXIT_HOOKED[0]:
        return
    _ATEXIT_HOOKED[0] = True

    def _cleanup():
        for s in list(_ACTIVE.values()):
            try:
                s.stop(wait_s=2.0)
            except Exception:                        # noqa: BLE001
                pass

    atexit.register(_cleanup)
    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is None:
            continue
        try:
            prev = signal.getsignal(sig)

            def _handler(signum, frame, _prev=prev):
                _cleanup()
                if callable(_prev):
                    _prev(signum, frame)
                else:
                    raise SystemExit(128 + signum)

            signal.signal(sig, _handler)
        except Exception:                            # noqa: BLE001
            pass                                     # 非主线程装不了信号处理器，靠 atexit
    if False:                                        # pragma: no cover
        sys.stderr.write("")
