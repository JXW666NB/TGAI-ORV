# -*- coding: utf-8 -*-
"""查占卡进程的父进程链 + 自启动项 + 服务（只读）。"""
import subprocess


def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                           encoding="gbk", errors="replace")
        return (p.stdout or "") + (p.stderr or "")
    except Exception as exc:
        return "ERR %s" % exc


def ps(script):
    return run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])


def main():
    gpu = run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
    pids = [x.strip() for x in gpu.splitlines() if x.strip().isdigit()]
    print("GPU pids: %s" % pids)
    # 逐级打印父进程
    for pid in pids:
        cur = int(pid)
        print("--- chain from %d ---" % cur)
        for _ in range(5):
            out = ps("$p=Get-CimInstance Win32_Process -Filter 'ProcessId=%d'; "
                     "if($p){'{0}|{1}|{2}' -f $p.ProcessId,$p.ParentProcessId,"
                     "$p.CommandLine}else{'-'}" % cur)
            line = out.strip()
            print("   " + line[:160])
            if "|" not in line:
                break
            try:
                cur = int(line.split("|")[1])
            except Exception:
                break
            if cur in (0, 4):
                break
    print("--- 自启动项 ---")
    print(ps("Get-CimInstance Win32_StartupCommand | "
             "ForEach-Object { '{0}|{1}|{2}' -f $_.Name,$_.Location,$_.Command }")[:2500])
    print("--- 含 tgai 的服务 ---")
    print(ps("Get-CimInstance Win32_Service | Where-Object {$_.PathName -match 'tgai|11450'} | "
             "ForEach-Object { '{0}|{1}|{2}' -f $_.Name,$_.State,$_.PathName }")[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
