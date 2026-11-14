# -*- coding: utf-8 -*-
"""V100 系统体检（**只读**）：计划任务、用户进程、自启动项、浏览器主页劫持。

输出刻意保持 ASCII 分隔符（|），避免中文经远程管道乱码。
"""
import csv
import io
import json
import os
import subprocess
import sys


def run(cmd, shell=False):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                           encoding="gbk", errors="replace", shell=shell)
        return (p.stdout or "") + (p.stderr or "")
    except Exception as exc:
        return "ERR %s" % exc


def ps(cmd):
    return run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd])


SYSTEM_HINTS = ("\\microsoft\\", "windows", "system32", "syswow64", "programdata\\microsoft",
                "defender", "onedrive", "microsoftedge", "nvidia", "intel", "realtek")


def norm(s):
    return (s or "").lower().replace("/", "\\")


def _section(which):
    """只跑某一节：2=自启动，3=进程，4=浏览器。"""
    if which == "2":
        print("### 2. 自启动项（注册表 Run + 启动文件夹）")
        print(ps("Get-CimInstance Win32_StartupCommand | "
                 "ForEach-Object { '{0}|{1}|{2}' -f $_.Name,$_.Location,$_.Command }"))
        return 0
    if which == "3":
        print("### 3. 占内存最多的用户进程")
        print(ps("Get-Process | Where-Object {$_.Path -and $_.Path -notmatch 'Windows\\\\'} | "
                 "Sort-Object WS -Descending | Select-Object -First 25 | "
                 "ForEach-Object { '{0}|{1}|{2}MB|{3}' -f $_.Id,$_.ProcessName,"
                 "[int]($_.WS/1MB),$_.Path }"))
        return 0
    if which == "4":
        return _section4()
    return 0


def _section4():
    print("### 4. 浏览器主页 / 快捷方式检查（2345）")
    for reg in (r"HKCU\Software\Microsoft\Internet Explorer\Main",
                r"HKLM\SOFTWARE\Policies\Microsoft\Edge",
                r"HKCU\Software\Policies\Microsoft\Edge",
                r"HKLM\SOFTWARE\Policies\Google\Chrome"):
        print("--- %s ---" % reg)
        print(run(["reg", "query", reg, "/v"]).strip()[:1200])
    users = os.path.expanduser("~")
    for tag, rel in (("Chrome", ("AppData", "Local", "Google", "Chrome", "User Data",
                                 "Default", "Preferences")),
                     ("Edge", ("AppData", "Local", "Microsoft", "Edge", "User Data",
                               "Default", "Preferences"))):
        path = os.path.join(users, *rel)
        if not os.path.exists(path):
            print("%s: 未安装或无配置" % tag)
            continue
        try:
            d = json.load(io.open(path, encoding="utf-8", errors="replace"))
            print("%s: homepage=%r startup_urls=%r"
                  % (tag, d.get("homepage", ""),
                     d.get("session", {}).get("startup_urls", "")))
        except Exception as exc:
            print("%s: 读取失败 %s" % (tag, exc))
    for base in (os.path.join(users, "Desktop"),
                 os.path.join(users, "AppData", "Roaming", "Microsoft", "Windows",
                              "Start Menu", "Programs")):
        for root, _, files in os.walk(base):
            for f in files:
                if not f.lower().endswith(".lnk"):
                    continue
                p = os.path.join(root, f)
                try:
                    if b"2345" in io.open(p, "rb").read():
                        print("命中 2345 快捷方式: %s" % p)
                except Exception:
                    pass
    return 0


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only not in ("all", "1"):
        return _section(only)
    print("### 1. 计划任务（非 Microsoft 路径）")
    out = run(["schtasks", "/query", "/fo", "csv", "/v"])
    rows = list(csv.reader(io.StringIO(out)))
    if rows:
        hdr = [h.strip().lower() for h in rows[0]]
        def idx(*names):
            for n in names:
                if n in hdr:
                    return hdr.index(n)
            return -1
        i_name = idx("taskname", "任务名")
        i_path = idx("taskpath", "任务路径")
        i_run = idx("task to run", "要运行的任务")
        i_stat = idx("status", "状态")
        i_next = idx("next run time", "下次运行时间")
        seen = set()
        for r in rows[1:]:
            if len(r) <= max(i_name, i_run):
                continue
            name = r[i_name].strip()
            path = r[i_path].strip() if i_path >= 0 and i_path < len(r) else ""
            task = (path + "\\" + name) if path else name
            if "microsoft" in norm(task):
                continue
            runto = r[i_run].strip() if i_run >= 0 and i_run < len(r) else ""
            stat = r[i_stat].strip() if i_stat >= 0 and i_stat < len(r) else ""
            key = (task, runto)
            if key in seen:
                continue
            seen.add(key)
            print("%-42s | %-8s | %s" % (task[:42], stat[:8], runto[:110]))
    print()
    if only not in ("all", "2"):
        return 0
    print("### 2. 自启动项（注册表 Run + 启动文件夹）")
    print(ps("Get-CimInstance Win32_StartupCommand | "
             "ForEach-Object { '{0}|{1}|{2}' -f $_.Name,$_.Location,$_.Command }")[:4000])
    print()
    if only not in ("all", "3"):
        return 0
    print("### 3. 占内存最多的用户进程")
    print(ps("Get-Process | Where-Object {$_.Path -and $_.Path -notmatch 'Windows\\\\'} | "
             "Sort-Object WS -Descending | Select-Object -First 25 | "
             "ForEach-Object { '{0}|{1}|{2}MB|{3}' -f $_.Id,$_.ProcessName,"
             "[int]($_.WS/1MB),$_.Path }")[:4000])
    print()
    if only not in ("all", "4"):
        return 0
    print("### 4. 浏览器主页 / 快捷方式检查（2345）")
    for reg in (r"Software\Microsoft\Internet Explorer\Main",
                r"Software\Microsoft\Edge\BLBeacon"):
        print(run(["reg", "query", "HKCU\\" + reg])[:800])
    users = os.path.expanduser("~")
    chrome = os.path.join(users, "AppData", "Local", "Google", "Chrome", "User Data",
                          "Default", "Preferences")
    edge = os.path.join(users, "AppData", "Local", "Microsoft", "Edge", "User Data",
                        "Default", "Preferences")
    for tag, path in (("Chrome", chrome), ("Edge", edge)):
        if not os.path.exists(path):
            print("%s: 未安装或无配置" % tag)
            continue
        try:
            d = json.load(io.open(path, encoding="utf-8", errors="replace"))
            home = d.get("homepage", "")
            urls = d.get("session", {}).get("startup_urls", "")
            print("%s: homepage=%r startup_urls=%r" % (tag, home, urls))
        except Exception as exc:
            print("%s: 读取失败 %s" % (tag, exc))
    # 桌面/开始菜单里带 2345 的快捷方式
    for base in (os.path.join(users, "Desktop"),
                 os.path.join(users, "AppData", "Roaming", "Microsoft", "Windows",
                              "Start Menu", "Programs")):
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith(".lnk"):
                    p = os.path.join(root, f)
                    try:
                        raw = io.open(p, "rb").read()
                        if b"2345" in raw:
                            print("命中 2345 快捷方式: %s" % p)
                    except Exception:
                        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
