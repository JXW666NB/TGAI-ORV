# -*- coding: utf-8 -*-
"""DP 归约后端（host / NCCL）的**无卡**测试。

这里钉的是「开关与失败行为」，不是带宽 —— 带宽必须上双卡测。要钉住的三件事：

1. **默认仍是 host**：不设 `TG_DP_BACKEND` 时绝不加载 NCCL（历史行为逐位不变）；
2. **显式 `nccl` 失败必须报错**：库缺失/版本过老/rank 数不符都不能悄悄退回 host
   （否则「我明明开了 NCCL」与「实际跑的是主机路径」无从区分）；
3. **只有 `auto` 才允许回退**，并且要留下 WARN 日志。

NCCL 通信器本身用**假库**测调用协议（group + 就地 all_reduce + count 校验），
因为真库在 CI/本机不一定有，而且真通信需要两张卡。
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train import dist as D          # noqa: E402
from tgai_go.train import dist_nccl as N     # noqa: E402


# ═════════════════════════════════════════════════════════════════════════
# 1) 后端选择
# ═════════════════════════════════════════════════════════════════════════
def test_default_backend_is_host(monkeypatch):
    monkeypatch.delenv("TG_DP_BACKEND", raising=False)
    assert D.dp_backend_requested() == "host"


@pytest.mark.parametrize("raw,want", [
    ("host", "host"), ("HOST", "host"), ("", "host"), ("none", "host"),
    ("off", "host"), ("0", "host"), ("nccl", "nccl"), ("NCCL", "nccl"),
    ("auto", "auto"), (" auto ", "auto"),
])
def test_backend_aliases(monkeypatch, raw, want):
    monkeypatch.setenv("TG_DP_BACKEND", raw)
    assert D.dp_backend_requested() == want


def test_unknown_backend_is_loud(monkeypatch):
    """写错一个字母不能静默退回 host —— 那是性能实验最难查的一类错。"""
    monkeypatch.setenv("TG_DP_BACKEND", "ncl")
    with pytest.raises(ValueError) as ei:
        D.dp_backend_requested()
    msg = str(ei.value)
    assert "ncl" in msg and "修法" in msg and "host" in msg


def test_explicit_backend_argument_wins_over_env(monkeypatch):
    monkeypatch.setenv("TG_DP_BACKEND", "host")
    dp = D.DataParallel(world=1, rank=0, backend="nccl")
    assert dp.backend_requested == "nccl"
    dp.close()


def test_world_one_is_noop_even_with_nccl_requested(monkeypatch):
    """world=1 必须是**快速空操作**：连库都不该去找。"""
    monkeypatch.setenv("TG_NCCL_LIB", "/nonexistent/libnccl.so.2")

    class _Boom(object):
        def __getattr__(self, name):
            raise AssertionError("world=1 不能碰 CUDA/NCCL（访问了 %s）" % name)

    import tgai_go.train as T
    monkeypatch.setattr(T, "_cuda", lambda: _Boom())
    dp = D.DataParallel(world=1, rank=0, backend="nccl")
    dp.all_reduce_grads([])                      # 空操作
    assert dp.backend == "host"
    assert dp.stats()["backend"] == "host"
    dp.close()


# ═════════════════════════════════════════════════════════════════════════
# 2) 显式 nccl 失败 / auto 回退
# ═════════════════════════════════════════════════════════════════════════
class _FakeRt(object):
    """只够 `_make_nccl_stream` 走「没有 cu」的分支。"""

    cu = None
    _stream = None

    def __init__(self):
        import ctypes
        self._stream = ctypes.c_void_p(0)


def _make_dp(monkeypatch, backend, tmp_path):
    import tgai_go.train as T
    monkeypatch.setattr(T, "_cuda", lambda: _FakeRt())
    monkeypatch.setenv("TGAI_DP_DIR", str(tmp_path))
    dp = D.DataParallel(world=2, rank=0, job="unit_backend", backend=backend,
                        timeout=5.0)
    return dp


def test_explicit_nccl_missing_lib_raises(monkeypatch, tmp_path):
    dp = _make_dp(monkeypatch, "nccl", tmp_path)

    def _boom(*a, **k):
        raise N.NcclUnavailable("找不到可用的 libnccl.so。修法：装 NCCL")

    monkeypatch.setattr(N, "find_libnccl", _boom)
    with pytest.raises(RuntimeError) as ei:
        dp._ensure_backend()
    msg = str(ei.value)
    assert "TG_DP_BACKEND=nccl" in msg, "错误必须点明是用户显式选了 nccl"
    assert "修法" in msg
    dp.close()


def test_auto_falls_back_to_host_with_warning(monkeypatch, tmp_path, capsys):
    dp = _make_dp(monkeypatch, "auto", tmp_path)

    def _boom(*a, **k):
        raise N.NcclUnavailable("找不到可用的 libnccl.so。修法：装 NCCL")

    monkeypatch.setattr(N, "find_libnccl", _boom)
    dp._ensure_backend()
    assert dp.backend == "host", "auto 必须回退 host"
    out = capsys.readouterr().out
    assert "WARN" in out and "回退 host" in out, "回退必须留下日志，不能静默"
    dp.close()


def test_host_backend_never_touches_nccl(monkeypatch, tmp_path):
    dp = _make_dp(monkeypatch, "host", tmp_path)
    called = []
    monkeypatch.setattr(N, "find_libnccl", lambda *a, **k: called.append(1))
    dp._ensure_backend()
    assert dp.backend == "host"
    assert not called, "host 后端不应该去找 libnccl"
    dp.close()


# ═════════════════════════════════════════════════════════════════════════
# 2b) rendezvous 残留 ready（2026-11-21 复现的 600 s 假死）
# ═════════════════════════════════════════════════════════════════════════
def test_ready_missing_is_not_stale(tmp_path):
    assert D.DataParallel._ready_is_stale(str(tmp_path / "ready")) is False


def test_ready_with_dead_owner_pid_is_stale(tmp_path, monkeypatch):
    """上一轮 rank0 已退出 ⇒ 旧 ready 必须被判为残留。

    这正是实测踩到的：两次实验复用同一个 TGAI_DP_JOB，rank1 相信旧 ready，
    直接冲向 bar_init_1，而 rank0 把它当残留删掉 ⇒ 双方各等 600 s。
    """
    monkeypatch.setattr(D.DataParallel, "_pid_alive",
                        staticmethod(lambda pid: False))
    path = tmp_path / "ready"
    path.write_bytes(b"4242 1.0")
    assert D.DataParallel._ready_is_stale(str(path)) is True


def test_ready_with_live_owner_pid_is_not_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(D.DataParallel, "_pid_alive",
                        staticmethod(lambda pid: True))
    path = tmp_path / "ready"
    path.write_bytes(b"4242 1.0")
    assert D.DataParallel._ready_is_stale(str(path)) is False


def test_own_pid_is_never_stale(tmp_path):
    path = tmp_path / "ready"
    path.write_bytes(b"%d 1.0" % os.getpid())
    assert D.DataParallel._ready_is_stale(str(path)) is False


def test_ready_with_garbage_content_is_stale(tmp_path):
    path = tmp_path / "ready"
    path.write_bytes(b"1")                  # 旧格式：没有 pid 字段
    assert D.DataParallel._ready_is_stale(str(path)) is True


# ═════════════════════════════════════════════════════════════════════════
# 2c) 分块打包的偏移数学（2026-11-21：pack 从 3.16 GiB 改成 256 MB 一块）
# ═════════════════════════════════════════════════════════════════════════
class _NBuf(object):
    """假设备缓冲：numpy 载体 + `.n`（元素数）。"""

    def __init__(self, arr):
        self.np = arr
        self.n = int(arr.size)
        self.shape = tuple(arr.shape)


class _FakeDma(object):
    """只实现 dist._pack_range / _add_range 用到的三个原语。"""

    @staticmethod
    def copy_d2d(dst, src, nbytes, dst_off=0, src_off=0):
        n = int(nbytes) // 4
        a = int(src_off) // 4
        b = int(dst_off) // 4
        dst.np[b:b + n] = src.np[a:a + n]

    @staticmethod
    def h2d_into(dst, src, nbytes=None, dst_off=0, src_off=0):
        pass

    @staticmethod
    def sub_view(buf, shape, elem_offset=0):
        n = 1
        for s in shape:
            n *= int(s)
        return _NBuf(np.ascontiguousarray(buf.np[int(elem_offset):int(elem_offset) + n]))

    @staticmethod
    def add_(dst, src):
        assert dst.np.size == src.np.size, "add_ 长度必须相同"
        dst.np += src.np


class _FakeGrad(object):
    """假梯度张量：dist 里所有访问都走 `g._gpu`。"""

    def __init__(self, buf):
        self._gpu = buf


def _entries_for_chunks(sizes):
    """造 (p, g, n) 三元组，梯度值 = 元素全局下标（便于校验）。"""
    total = sum(sizes)
    flat = np.arange(total, dtype=np.float32)
    out = []
    off = 0
    for n in sizes:
        out.append((object(), _FakeGrad(_NBuf(np.ascontiguousarray(flat[off:off + n]))), n))
        off += n
    return out, total


@pytest.mark.parametrize("chunk,sizes", [
    (4, [7, 3, 5]),                  # 每块都跨张量
    (10, [10, 10]),                  # 正好对齐
    (3, [1, 1, 28]),                 # 单元素 + 超长张量
    (1000, [7, 3, 5]),               # 块比总长还大（=旧的整块行为）
    (5, [5]),
])
def test_chunked_pack_add_covers_every_element_exactly_once(chunk, sizes):
    """分块打包/加回必须**逐元素覆盖一次**：漏一块会静默少加一部分梯度。"""
    entries, total = _entries_for_chunks(sizes)
    pack = _NBuf(np.zeros(chunk, dtype=np.float32))
    rt = _FakeDma
    # ① 分块打包到一块"平铺缓冲"
    flat = np.zeros(total, dtype=np.float32)
    lo = 0
    while lo < total:
        hi = min(total, lo + chunk)
        rt.copy_d2d  # noqa: B018 - 明确这是同一套原语
        _pack = np.zeros(chunk, dtype=np.float32)
        D.DataParallel._pack_range(rt, entries, lo, hi, _NBuf(_pack))
        flat[lo:hi] = _pack[:hi - lo]
        lo = hi
    assert np.array_equal(flat, np.arange(total, dtype=np.float32)), (
        "打包结果必须逐元素等于原始梯度（偏移算错就会在这里露出来）")

    # ② 分块加回：目标初始全 0，加完应等于原始梯度（每块只加一次）
    dst_entries = [(p, _FakeGrad(_NBuf(np.zeros(n, dtype=np.float32))), n)
                   for p, g, n in entries]
    lo = 0
    while lo < total:
        hi = min(total, lo + chunk)
        _chunk = np.ascontiguousarray(np.arange(lo, hi, dtype=np.float32))
        D.DataParallel._add_range(rt, dst_entries, lo, hi, _NBuf(_chunk))
        lo = hi
    got = np.concatenate([g._gpu.np for _p, g, _n in dst_entries])
    assert np.array_equal(got, np.arange(total, dtype=np.float32)), (
        "加回后每个元素必须恰好被加一次")
    assert pack.n == chunk


# ═════════════════════════════════════════════════════════════════════════
# 3) 库发现 / 版本 / unique id
# ═════════════════════════════════════════════════════════════════════════
def test_version_string():
    assert N.version_string(22703) == "2.27.3"
    assert N.version_string(20000) == "2.0.0"
    assert N.version_string(2000) == "0.20.0"       # ← 编码陷阱：2.0.0 是 20000
    assert N.version_string(28105) == "2.81.5"


def test_explicit_lib_path_that_does_not_exist_is_loud(monkeypatch, tmp_path):
    monkeypatch.setenv("TG_NCCL_LIB", str(tmp_path / "nope.so"))
    with pytest.raises(N.NcclUnavailable) as ei:
        N.find_libnccl()
    msg = str(ei.value)
    assert "TG_NCCL_LIB" in msg and "修法" in msg


def test_unique_id_roundtrip():
    raw = bytes(range(128))
    uid = N.unique_id_from_bytes(raw)
    assert N.unique_id_to_bytes(uid) == raw


def test_unique_id_rejects_wrong_length():
    with pytest.raises(N.NcclError) as ei:
        N.unique_id_from_bytes(b"short")
    assert "长度不符" in str(ei.value) and "修法" in str(ei.value)


def test_unique_id_file_roundtrip(tmp_path):
    raw = bytes([7] * 128)
    N.write_unique_id(str(tmp_path), raw)
    assert N.wait_unique_id(str(tmp_path), timeout=1.0) == raw


def test_unique_id_wait_times_out_with_actionable_message(tmp_path):
    with pytest.raises(N.NcclError) as ei:
        N.wait_unique_id(str(tmp_path), timeout=0.05)
    msg = str(ei.value)
    assert "超时" in msg and "修法" in msg and "TGAI_DP_JOB" in msg


class _FakeSym(object):
    def __init__(self, fn):
        self._fn = fn
        self.argtypes = None
        self.restype = None

    def __call__(self, *a):
        return self._fn(*a)


class _FakeDll(object):
    """最小 CDLL 替身：按名字返回符号，`ncclGetVersion` 写回指定版本。"""

    def __init__(self, version=22703):
        self.version = version
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def _fn(*a, **k):
            self.calls.append(name)
            if name == "ncclGetVersion":
                a[0]._obj.value = self.version       # ctypes.byref 解引用
                return 0
            return 0

        return _FakeSym(_fn)


def test_old_nccl_version_is_rejected():
    """2.0.0 的编码是 20000；19999 = 1.99.99 必须被拒。"""
    with pytest.raises(N.NcclVersionMismatch) as ei:
        N.LibNccl("fake-libnccl.so", handle=_FakeDll(version=19999))
    msg = str(ei.value)
    assert "1.99.99" in msg and "修法" in msg


def test_lib_accepts_modern_version():
    lib = N.LibNccl("fake-libnccl.so", handle=_FakeDll(version=22703))
    assert lib.version_str() == "2.27.3"
    assert lib.version == 22703


# ═════════════════════════════════════════════════════════════════════════
# 4) NcclComm 的调用协议（假库）
# ═════════════════════════════════════════════════════════════════════════
class _FakeLib(object):
    def __init__(self, count=2, user_rank=0, device=0):
        self.count = count
        self.user_rank = user_rank
        self.device = device
        self.events = []
        self.destroyed = 0
        self.aborted = 0

    def init_rank(self, world, rank, uid):
        self.events.append(("init", world, rank))
        return 0xDEAD

    def comm_count(self, comm):
        return self.count

    def comm_user_rank(self, comm):
        return self.user_rank

    def comm_cu_device(self, comm):
        return self.device

    def group_start(self):
        self.events.append(("group_start",))

    def group_end(self):
        self.events.append(("group_end",))

    def all_reduce(self, comm, send, recv, count, stream=None):
        self.events.append(("all_reduce", send, recv, count))

    def destroy(self, comm):
        self.destroyed += 1

    def abort(self, comm):
        self.aborted += 1


def _uid():
    return N.unique_id_from_bytes(bytes(128))


def test_comm_rejects_world_mismatch():
    lib = _FakeLib(count=3, user_rank=0)
    with pytest.raises(N.NcclRankMismatch) as ei:
        N.NcclComm(lib, 2, 0, _uid())
    assert "rank 数" in str(ei.value) and "修法" in str(ei.value)
    assert lib.aborted == 1, "初始化失败必须 abort，不能把半死的 comm 留着"


def test_comm_rejects_rank_mismatch():
    lib = _FakeLib(count=2, user_rank=1)
    with pytest.raises(N.NcclRankMismatch) as ei:
        N.NcclComm(lib, 2, 0, _uid())
    assert "rank 是" in str(ei.value)


def test_comm_rejects_device_mismatch():
    lib = _FakeLib(count=2, user_rank=0, device=1)
    with pytest.raises(N.NcclRankMismatch) as ei:
        N.NcclComm(lib, 2, 0, _uid(), expect_device=0)
    assert "CUDA 设备" in str(ei.value)


def test_comm_groups_and_all_reduces_in_place():
    lib = _FakeLib()
    comm = N.NcclComm(lib, 2, 0, _uid())
    comm.all_reduce_group([(0x1000, 16), (0x2000, 8), (0x3000, 0)])
    kinds = [e[0] for e in lib.events]
    assert kinds == ["init", "group_start", "all_reduce", "all_reduce", "group_end"], (
        "必须在一次 group 里发完所有 all_reduce；count=0 的要跳过")
    assert ("all_reduce", 0x1000, 0x1000, 16) in lib.events, "必须是就地归约"
    comm.close()
    assert lib.destroyed == 1


def test_comm_close_is_idempotent():
    lib = _FakeLib()
    comm = N.NcclComm(lib, 2, 0, _uid())
    comm.close()
    comm.close()
    assert lib.destroyed == 1, "close() 不能重复 destroy"
    with pytest.raises(N.NcclError):
        comm.all_reduce_group([(0x1000, 4)])


def test_comm_context_manager_aborts_on_exception():
    lib = _FakeLib()
    with pytest.raises(ZeroDivisionError):
        with N.NcclComm(lib, 2, 0, _uid()):
            1 / 0
    assert lib.aborted == 1, "异常退出必须 abort（否则其它 rank 会挂在集合通信里）"
