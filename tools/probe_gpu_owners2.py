# -*- coding: utf-8 -*-
"""找出占显存进程的父进程链（ASCII 输出，便于远程读取）。"""
import subprocess


def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                           encoding="gbk", errors="replace")
        return (p.stdout or "") + (p.stderr or "")
    except Exception as exc:
        return "ERR %s" % exc


def gpu_pids():
    out = run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
    return [x.strip() for x in out.splitlines() if x.strip().isdigit()]


def ps_table():
    tbl = {}
    for line in run(["wmic", "process", "get", "ProcessId,ParentProcessId,Name,CommandLine",
                     "/format:csv"]).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6 and parts[3].isdigit():
            tbl[parts[3]] = (parts[2], parts[3], parts[4], ",".join(parts[5:]))
    return tbl


def main():
    pids = gpu_pids()
    print("GPU pids: %s" % (pids,))
    tbl = ps_table()
    for pid in pids:
        cur = pid
        chain = []
        for _ in range(4):
            info = tbl.get(cur)
            if not info:
                chain.append("%s(unknown)" % cur)
                break
            _, cpid, name, cmd = info
            chain.append("%s[%s] %s" % (cpid, name, (cmd or "")[:90]))
            ppid = None
            for k, v in tbl.items():
                if v[1] == cpid:
                    pass
            # 直接用 wmic 查 ppid
            q = run(["wmic", "process", "where", "ProcessId=%s" % cur,
                     "get", "ParentProcessId", "/format:csv"])
            for ln in q.splitlines():
                f = [x.strip() for x in ln.split(",")]
                if len(f) >= 2 and f[-1].isdigit() and f[-1] != cur:
                    ppid = f[-1]
            if not ppid or ppid == "0":
                break
            cur = ppid
        print(" --- ")
        for c in chain:
            print("   " + c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
