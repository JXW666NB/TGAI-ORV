# -*- coding: utf-8 -*-
"""`T.DUO()` 的测试：**全部不需要 GPU**。

这是刻意的 —— `DUO` 最容易出事的地方（起进程、杀进程、僵尸、OOM 识别、拒绝不可行计划）
**都不需要显卡就能测**。真机验证留给后面，但"会不会把整机 hang 死"这种问题
必须现在就有证据。
"""
import os
import sys
import time

import pytest

from tgai_go.train import duo as D
from tgai_go.train import plan as P
from tgai_go.train import topo as TO


# ---------------- 拓扑解析 ----------------
_TOPO_TXT = """        GPU0    GPU1    GPU2    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      PIX     PHB     0-15            0               N/A
GPU1    PIX      X      PHB     0-15            0               N/A
GPU2    PHB     PHB     X      16-31            1               N/A

Legend:
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes
"""

_NV_TXT = """        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NV4     0-15            0
GPU1    NV4      X      0-15            0
"""


def test_parse_topo_reads_links_and_numa():
    links, numas = TO.parse_topo(_TOPO_TXT)
    assert links[0][0] == "X"
    assert links[0][1] == "PIX"
    assert links[2][0] == "PHB"
    assert numas == 2


def test_link_rank_orders_nvlink_before_pcie_before_sys():
    assert TO.link_rank("NV4") < TO.link_rank("PIX")
    assert TO.link_rank("PIX") < TO.link_rank("PXB") < TO.link_rank("PHB")
    assert TO.link_rank("PHB") < TO.link_rank("SYS")
    assert TO.link_rank("") == 9          # 未知一律当最差，别乐观


def test_closest_pairs_sorted_and_nvlink_detected():
    links, _ = TO.parse_topo(_NV_TXT)
    t = TO.HostTopology([TO.DeviceInfo(0, "5090", 31.4), TO.DeviceInfo(1, "5090", 31.4)],
                        links)
    assert t.has_nvlink() is True
    pairs = t.closest_pairs()
    assert pairs[0][:3] == (0, 0, 1)      # 距离级别 0 = NVLink


def test_sys_topology_reports_no_nvlink():
    t = TO.HostTopology([TO.DeviceInfo(0), TO.DeviceInfo(1)], [["X", "SYS"], ["SYS", "X"]])
    assert t.has_nvlink() is False
    assert t.has_p2p() is False
    assert "无 NVLink" in t.summary()


def test_heterogeneous_is_flagged():
    t = TO.HostTopology([TO.DeviceInfo(0, "V100"), TO.DeviceInfo(1, "4090")],
                        [["X", "SYS"], ["SYS", "X"]])
    assert t.is_heterogeneous() is True
    assert "木桶" in t.summary()


def test_parse_gpu_query():
    devs = TO.parse_gpu_query("0, NVIDIA GeForce RTX 5090, 32607, 0000:01:00.0\n"
                              "1, NVIDIA GeForce RTX 5090, 32607, 0000:02:00.0\n")
    assert [d.index for d in devs] == [0, 1]
    assert devs[0].total_gib == pytest.approx(32607 / 1024.0)
    assert devs[1].pci_bus == "0000:02:00.0"


# ---------------- DUO 的拒绝（一个进程都不许起） ----------------
def _fake_topo(n=2, gib=31.36, link="SYS"):
    devs = [TO.DeviceInfo(i, "RTX 5090", gib) for i in range(n)]
    links = [["X" if i == j else link for j in range(n)] for i in range(n)]
    return TO.HostTopology(devs, links if n > 1 else [])


def _session(n=2, n_params=3_000_000_000, gib=31.36, single_step_s=None):
    """绕过 nvidia-smi，直接造一个会话（测试用）。"""
    devs = [TO.DeviceInfo(i, "RTX 5090", gib) for i in range(n)]
    args = D._Args(devices=None, place=None, n_params=n_params, opt="int8gpu",
                   seq=512, d=1024, d_ff=2816, layers=20, E=8, zero=0,
                   dist="2" if n > 1 else "1", single_step_s=single_step_s)
    pl = P.build(args, devices=[P.Device(d.index, d.name, d.total_gib) for d in devs])
    return D.DuoSession(pl, _fake_topo(n, gib), devs, "auto", "test-%d" % time.time_ns())


def test_run_refuses_before_spawning_anything():
    """3B 在 2×31.36 上 DP 装不下 ⇒ 必须**在起进程之前**就拒绝。"""
    s = _session(n_params=6_000_000_000)
    assert s.plan.ok is False
    with pytest.raises(D.DuoRefused) as e:
        s.run([sys.executable, "-c", "print(1)"])
    assert "拒绝起进程" in str(e.value)
    assert s.procs == {}                  # 一个进程都没起
    s.stop()


def test_unimplemented_strategy_is_refused_explicitly(monkeypatch):
    monkeypatch.setattr(D._topo, "probe", lambda *a, **k: _fake_topo())
    with pytest.raises(D.DuoRefused) as e:
        D.DUO(strategy="tensor_parallel")
    assert "没有实现" in str(e.value) and "修法" in str(e.value)


def test_missing_device_is_refused(monkeypatch):
    monkeypatch.setattr(D._topo, "probe", lambda *a, **k: _fake_topo(2))
    with pytest.raises(D.DuoRefused) as e:
        D.DUO(devices=[0, 5], n_params=848_645_257)
    assert "不存在" in str(e.value)


def test_no_gpu_at_all_is_refused_with_hint(monkeypatch):
    monkeypatch.setattr(D._topo, "probe",
                        lambda *a, **k: TO.HostTopology([], [], source="none"))
    with pytest.raises(D.DuoRefused) as e:
        D.DUO()
    assert "没有可用显卡" in str(e.value) and "修法" in str(e.value)


# ---------------- 真起进程：成功、失败、OOM、超时 ----------------
def test_run_success_two_ranks(tmp_path):
    s = _session(n=2, n_params=848_645_257)      # 848M DP 装得下
    s.log_dir = str(tmp_path)
    r = s.run([sys.executable, "-c",
               "import os;print(os.environ['TGAI_DP_RANK'],os.environ['TGAI_DP_WORLD'],"
               "os.environ['CUDA_VISIBLE_DEVICES'])"])
    assert r.ok, r.summary()
    assert r.returncodes == {0: 0, 1: 0}
    # 每个 rank 拿到不同的卡与不同的 RANK
    for rank in (0, 1):
        txt = open(r.logs[rank], encoding="utf-8", errors="replace").read()
        assert str(rank) in txt and "2" in txt
    s.stop()


def test_one_rank_fails_others_are_killed(tmp_path):
    """rank1 立刻失败、rank0 睡 30 秒 ⇒ 必须**杀掉 rank0**，不许等它。"""
    s = _session(n=2, n_params=848_645_257)
    s.log_dir = str(tmp_path)
    code = ("import os,sys,time;"
            "r=int(os.environ['TGAI_DP_RANK']);"
            "sys.exit(3) if r==1 else time.sleep(30)")
    t0 = time.time()
    r = s.run([sys.executable, "-c", code], )
    dt = time.time() - t0
    assert r.ok is False
    assert r.failed_rank == 1
    assert r.killed is True
    assert dt < 15, "没有及时杀掉另一个 rank（等了 %.1fs）" % dt
    s.stop()


def test_oom_text_is_detected_and_rank_marked(tmp_path):
    s = _session(n=2, n_params=848_645_257)
    s.log_dir = str(tmp_path)
    code = ("import os,sys;"
            "r=int(os.environ['TGAI_DP_RANK']);"
            "sys.exit(1) if r==0 else None")
    # rank0 输出"显存不足"并以非 0 退出
    code = ("import os,sys;"
            "r=int(os.environ['TGAI_DP_RANK']);"
            "(print('显存不足：本次申请 0.01 GiB'), sys.exit(1)) if r==0 else None")
    r = s.run([sys.executable, "-c", code])
    assert r.ok is False
    assert r.failed_rank == 0
    assert 0 in r.oom_ranks
    assert "OOM" in r.summary()
    s.stop()


def test_timeout_kills_everything(tmp_path):
    s = _session(n=2, n_params=848_645_257)
    s.log_dir = str(tmp_path)
    s.timeout_s = 1.0
    t0 = time.time()
    r = s.run([sys.executable, "-c", "import time;time.sleep(60)"])
    assert time.time() - t0 < 20
    assert r.killed is True
    s.stop()


def test_stop_is_idempotent():
    s = _session(n=2, n_params=848_645_257)
    s.stop()
    s.stop()
    assert s._closed is True


def test_run_after_stop_is_refused():
    s = _session(n=2, n_params=848_645_257)
    s.stop()
    with pytest.raises(D.DuoRefused) as e:
        s.run([sys.executable, "-c", "print(1)"])
    assert "已经关闭" in str(e.value)


def test_empty_command_is_refused():
    s = _session(n=2, n_params=848_645_257)
    with pytest.raises(D.DuoRefused) as e:
        s.run([])
    assert "command 是空的" in str(e.value)
    s.stop()


# ---------------- 数据切分（墙 ②） ----------------
def test_shard_indices_are_disjoint_and_complete():
    for n in (10, 7, 3, 1):
        for world in (1, 2, 3):
            parts = [D._shard_indices(n, world, r) for r in range(world)]
            flat = [x for p in parts for x in p]
            assert flat == list(range(n)), (n, world, parts)
            assert sum(len(p) for p in parts) == n


def test_shard_indices_balances():
    parts = [D._shard_indices(10, 3, r) for r in range(3)]
    assert [len(p) for p in parts] == [4, 3, 3]      # 余数给前面的 rank
