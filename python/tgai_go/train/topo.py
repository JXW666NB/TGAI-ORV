# -*- coding: utf-8 -*-
"""本机拓扑与资源探测：**决定"哪张卡和哪张卡近"**。

为什么要单独一个模块：`T.DUO()` 要接管整机资源，第一件事就是知道主机长什么样 ——
几张卡、每张多少显存、谁和谁之间有 NVLink/PIX/PXB/PHB/SYS、几个 NUMA 节点、
多少 CPU 核、多少内存。**这些都不是猜的，全部来自 `nvidia-smi` 的实测输出。**

## 为什么拓扑重要（实测数字）

同一份 3.16 GiB 梯度归约，在 2×5090 的 `SYS` 拓扑（跨 NUMA）上是 **140.78 ms**
（NCCL，48.2 GB/s 收发合计）。如果两张卡在同一个 PCIe switch 下（`PXB`），
或者有 NVLink（`NV#`），这个数字会差一个数量级。所以：

* `closest_pairs()` 给出"哪两张卡最近" ⇒ 起 worker 时优先把通信最重的 rank 放过去
* `has_nvlink` / `has_p2p` 喂给 `plan.py`，让规划器知道 ZeRO-3 到底值不值得
"""
import os
import subprocess

__all__ = ["DeviceInfo", "HostTopology", "probe", "parse_topo",
           "parse_gpu_query", "link_rank"]

#: 拓扑距离分级（数字越小越近）。`NV#` 最优先，`SYS` 最差。
_LINK_ORDER = {"NV": 0, "PIX": 1, "PXB": 2, "PHB": 3, "NODE": 4, "SYS": 5}


def link_rank(link):
    """把 `NV4` / `PIX` / `SYS` 这类字符串映射成分级数字（越小越近）。"""
    s = str(link or "").strip().upper()
    for key in ("NV", "PIX", "PXB", "PHB", "NODE", "SYS"):
        if s.startswith(key):
            return _LINK_ORDER[key]
    return 9                      # 未知一律当最差，别乐观


class DeviceInfo(object):
    """一张卡。`index` 是**物理**序号（`CUDA_VISIBLE_DEVICES` 之前）。"""

    __slots__ = ("index", "name", "total_gib", "pci_bus", "numa")

    def __init__(self, index, name="device", total_gib=0.0, pci_bus="", numa=None):
        self.index = int(index)
        self.name = str(name)
        self.total_gib = float(total_gib)
        self.pci_bus = str(pci_bus or "")
        self.numa = numa

    @property
    def free_gib(self):
        """可用上限按 95% 算（驱动/上下文/碎片总要留一点）。"""
        return self.total_gib * 0.95

    def __repr__(self):
        return "DeviceInfo(%d, %s, %.1f GiB)" % (self.index, self.name, self.total_gib)


class HostTopology(object):
    """本机资源快照 + 卡间连接矩阵。"""

    __slots__ = ("devices", "links", "cpu_count", "ram_gib", "numa_nodes", "source")

    def __init__(self, devices, links=None, cpu_count=None, ram_gib=None,
                 numa_nodes=None, source="nvidia-smi"):
        self.devices = list(devices)
        #: `links[i][j]` 是原始字符串（`NV4`/`PIX`/…），对角线为 `X`
        self.links = links or []
        self.cpu_count = int(cpu_count or _cpu_count())
        self.ram_gib = float(ram_gib if ram_gib is not None else _ram_gib())
        self.numa_nodes = numa_nodes
        self.source = source

    # ---- 事实查询 ----
    def count(self):
        return len(self.devices)

    def has_nvlink(self):
        return any(link_rank(x) == 0 for row in self.links for x in row if x)

    def has_p2p(self):
        """同一 PCIe 域内（PIX/PXB/PHB）可以点对点；`SYS` 跨 NUMA 通常不行。

        注意这是**拓扑推断**，不是运行时能力：消费卡即使 PCIe 同域，驱动也可能禁用 P2P。
        真正的判据是 `cudaDeviceCanAccessPeer`，本模块取不到就如实返回 None。
        """
        for row in self.links:
            for x in row:
                if x and link_rank(x) in (1, 2, 3):
                    return True
        return False

    def is_heterogeneous(self):
        names = {d.name for d in self.devices}
        return len(names) > 1

    def closest_pairs(self):
        """按"连接最近"排序的所有卡对：`[(rank, i, j, link), ...]`。"""
        out = []
        n = len(self.devices)
        for i in range(n):
            for j in range(i + 1, n):
                link = self.link(i, j)
                out.append((link_rank(link), i, j, link))
        out.sort()
        return out

    def link(self, i, j):
        try:
            return self.links[int(i)][int(j)]
        except Exception:                                # noqa: BLE001
            return ""

    def summary(self):
        lines = ["[duo] 本机 %d 张卡 / %d 核 / %.0f GiB 内存%s"
                 % (self.count(), self.cpu_count, self.ram_gib,
                    "" if self.numa_nodes is None else " / %d NUMA" % self.numa_nodes)]
        for d in self.devices:
            lines.append("   #%d %-24s %5.1f GiB  numa=%s"
                         % (d.index, d.name, d.total_gib, d.numa))
        if self.count() > 1 and self.links:
            worst = self.closest_pairs()[-1]
            best = self.closest_pairs()[0]
            lines.append("   最近的一对 #%d↔#%d = %s；最远的一对 #%d↔#%d = %s"
                         % (best[1], best[2], best[3], worst[1], worst[2], worst[3]))
            if not self.has_nvlink():
                lines.append("   无 NVLink；跨卡通信走 PCIe（实测 48.2 GB/s 收发合计）")
        if self.is_heterogeneous():
            lines.append("   ⚠️ 异构卡：同步数据并行会被最慢的那张卡拖住（木桶效应）")
        return "\n".join(lines)


def _cpu_count():
    try:
        return len(os.sched_getaffinity(0))          # 尊重 taskset/cgroup
    except Exception:                                # noqa: BLE001
        return os.cpu_count() or 1


def _ram_gib():
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / 1024.0 / 1024.0
    except Exception:                                # noqa: BLE001
        pass
    try:
        import ctypes

        class _MemStat(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _MemStat()
        st.dwLength = ctypes.sizeof(_MemStat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))   # type: ignore[attr-defined]
        return float(st.ullTotalPhys) / (1 << 30)
    except Exception:                                # noqa: BLE001
        return 0.0


def parse_gpu_query(text):
    """解析 `nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id --format=csv,noheader`。"""
    devs = []
    for line in str(text or "").strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 3:
            continue
        try:
            idx = int(f[0])
            gib = float(f[2]) / 1024.0
        except ValueError:
            continue
        devs.append(DeviceInfo(idx, f[1], gib, f[3] if len(f) > 3 else ""))
    return devs


def parse_topo(text):
    """解析 `nvidia-smi topo -m` 的连接矩阵。

    返回 `(links, numa_nodes)`：`links[i][j]` 是原始字符串，`numa_nodes` 取表里
    `NUMA Affinity` 列的取值个数（取不到就 None）。
    """
    rows = []
    for line in str(text or "").splitlines():
        if not line.strip().startswith("GPU"):
            continue
        parts = line.split()
        # 表头行也以 "GPU" 开头（`GPU0  GPU1  ...  CPU Affinity`），
        # 必须靠"有没有 X（自连）"把它区分出去，否则整张矩阵会错位一行。
        if "X" not in parts:
            continue
        if len(parts) < 2:
            continue
        rows.append(parts)
    if not rows:
        return [], None
    n = len([p for p in rows[0][1:] if p.upper().startswith("GPU") and False]) or None
    # 连接列：从第 2 列开始，直到出现 CPU 亲和性（含逗号或形如 0-51）
    links = []
    numas = set()
    for parts in rows:
        cells = []
        for tok in parts[1:]:
            if tok.upper().startswith("GPU") or tok in ("X", "SYS", "PHB", "PXB", "PIX",
                                                       "NODE") or tok.upper().startswith("NV"):
                cells.append(tok)
            else:
                break
        links.append(cells)
    # NUMA：每行最后几个字段里会出现 NUMA 亲和性（单个数字）
    for parts in rows:
        tail = parts[-3:]
        for tok in tail:
            if tok.isdigit():
                numas.add(int(tok))
                break
    return links, (len(numas) if numas else None)


def probe(visible=None):
    """真正去问本机。**任何一步失败都不编造**，只是少给信息。"""
    devs = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,pci.bus_id",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
        devs = parse_gpu_query(out)
    except Exception:                                # noqa: BLE001
        devs = []
    links, numas = [], None
    if len(devs) > 1:
        try:
            out = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                                 text=True, timeout=15).stdout
            links, numas = parse_topo(out)
        except Exception:                            # noqa: BLE001
            links, numas = [], None
    if visible:
        keep = [int(x) for x in visible]
        devs = [d for d in devs if d.index in keep]
        if links:
            links = [[links[i][j] for j in keep if j < len(links)]
                     for i in keep if i < len(links)]
    return HostTopology(devs, links, numa_nodes=numas,
                        source="nvidia-smi" if devs else "none")
